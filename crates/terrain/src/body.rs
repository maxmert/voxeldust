//! ★ THE BODY DEFINITION — everything the recipe needs about one round body, drawn ONCE from the
//! body's seed and its radius: the ladder, the sea level, the relief, the octave table, the strata,
//! the biome and the cave parameters.
//!
//! ★ THE CHARTER IS INTEGERS (ruling F7, 2026-09-12). Every number a kernel reads is a word in the
//! recipe's own fixed-point formats — the octaves as [`vd_recipe::height::Octave`], the radius and
//! the sea in gap steps, the cave band in gap steps, the cavern wavelength's RECIPROCAL so no kernel
//! divides, the cell-count reciprocal of every rung so no kernel divides there either. The DRAW is
//! the one float left in the generator: the seed's shares, caps and weights are read as fenced
//! floats ([`crate::gf::Gf`], IEEE-exact on every target) and ROUNDED ONCE into the charter, here,
//! on the CPU, once per body. A kernel never sees a float. (Ruling V13 L12 owes the next step: the
//! realm STORES the charter and STATES it in its surface statement, so the draw itself runs once in
//! the world's life rather than once per process.)
//!
//! **Example.** The home planet's seed draws its relief, its sea level, its octave table from a long
//! wave down to a 30 m ripple, its topsoil over subsoil over sediment over bedrock, and its cave band.
//! Every host that holds the seed holds this table as the same words (`crate::home` states the planet;
//! the bench `terrain_cost` prints its numbers).

use crate::gf::Gf;
use crate::strata::{Bedrock, StrataTable, Stratum};
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use vd_recipe::Gi;
use vd_recipe::bend::inv_n_of;
use vd_recipe::height::{
    AMP_BITS, BiomeCharter, OCTAVE_COARSE, OCTAVE_FINE, OCTAVE_RIDGED, OCTAVE_SMOOTH, Octave,
    Roughness,
};
use vd_recipe::noise::NOISE_BITS;
use vd_recipe::plan::PlanCharter;
use vd_recipe::root::recip_pow2;
use vd_recipe::terrace::{TERRACE_RECIP_BITS, Terrace};
use vd_seed::bend::Face;
use vd_seed::ladder::{Ladder, cell_m};
use vd_seed::rng::{SplitMix64, child_seed};

/// The most octaves a body can have; the table is sized for it. The recipe holds the same cap: a
/// GPU shell keeps the octaves in a fixed-size table of it (`vd_recipe::height::relief_of_table`).
pub const OCTAVES: usize = 16;
const _: () = assert!(OCTAVES == vd_recipe::height::OCTAVES_CAP);
/// The most rungs a body can have: the address's five bits and one (`vd_seed::ladder::RUNG_MAX`).
pub const RUNGS: usize = 22;
/// The coarsest wavelength any body draws, in metres, and the wavelength below which no octave is
/// added. The table can never overflow: the cap halves to under the floor within `OCTAVES` steps.
pub const LONG_WAVE_CAP_M: u64 = 400_000;
pub const SHORT_WAVE_M: u64 = 30;
const _: () = assert!((LONG_WAVE_CAP_M >> OCTAVES) < SHORT_WAVE_M);
const _: () = assert!(RUNGS == vd_seed::ladder::RUNG_MAX as usize + 1);

/// ★ THE SLOPE SPECTRUM — slice 8a, stage 1 (`slice_8a_design.md` §1.1 and §2.1; `03_erosion_rivers.md`
/// §4.2.3; ruling V13 L5). The FINE octaves stop being a geometric ladder. Each one states an RMS
/// SLOPE drawn from a rational bump in the octave INDEX, and its amplitude follows from that slope
/// and its own wavelength:
///
/// ```text
///    s(o) = S_PEAK / (1 + ((o − O_PEAK)·ln 2)² / SIGMA²)      the octave's RMS slope
///    a(o) = s(o) · λ(o) / τ                                   a sine of amplitude a and wavelength λ
/// ```
///
/// The COARSE octaves keep the geometric ladder: 8c's macro solve replaces them, so 8a does not
/// touch them. The relief law stays where it is (ruling T6 ask 1 moved it to 8b).
///
/// **Example.** The home planet's 3.13 km octave sits at the bump's peak, so it carries the steepest
/// slope of the table — 0.40 — and 198.8 m of amplitude; the 25 km octave beside it carries 0.125 and
/// 496.1 m. Together the ten fine octaves make 1 532 m of relief at an RMS slope of 0.700, which is
/// the tangent of the angle of repose.
///
/// THE OCTAVE INDEX THE BUMP PEAKS AT — the 3.13 km octave on a 400 km table
/// (`03_erosion_rivers.md` §4.2.3).
pub const O_PEAK: i64 = 7;
/// THE BUMP'S WIDTH, in the same unit as `x`: the octave index times `ln 2`, which is the LOG of the
/// wavelength, so the bump is symmetric in the wavelength and not in the index
/// (`03_erosion_rivers.md` §4.2.3).
pub const SIGMA: f64 = 1.4;
/// The natural logarithm of two, and a whole turn: CONSTANTS, never calls — the float fence offers
/// no transcendental (`crate::gf`).
const LN_2: f64 = std::f64::consts::LN_2;
const TAU: f64 = std::f64::consts::TAU;
/// ★ THE ANCHOR that fixes the bump's peak. The fine octaves' slopes, taken as a root sum of squares,
/// equal `TALUS_RMS × TAN_REPOSE`: loose talus stands at its angle of repose, 35°, and a surface whose
/// fine roughness is one times that tangent is a hillside at the angle rock actually rests at
/// (`03_erosion_rivers.md` §4.2.3; ruling V13 L5). `tan 35° = 0.700 207 5`, stated to four places as
/// the arc states it, because the fence has no tangent to call.
pub const TALUS_RMS: f64 = 1.0;
pub const TAN_REPOSE: f64 = 0.70;
/// ★ WHICH OCTAVES ARE FINE, in METRES. `03_erosion_rivers.md` §4.2.1 makes an octave COARSE while
/// `λ(o) ≥ MACRO_SAMPLES_PER_WAVE · macro_node_m` — the solve reads it and 8c's macro field `Z`
/// replaces it. 8a HAS NO MACRO NODE, so it states the same threshold as one number:
/// `4 × 8 224 m = 32 896 m`, the macro node `03` computes for the home planet (`03` §4.7: the macro
/// lattice is 640 nodes a face edge, so a node is 8 224 m) times the four samples per wavelength
/// `03` §4.2.1 requires. On the home planet it selects octaves 4..13 — 25 km down to 49 m — which is
/// the band every number in the arc was computed on. **8c replaces the constant with
/// `4 · macro_node_m` and the band does not move on the home planet, by construction.**
pub const MACRO_NODE_M: u64 = 8_224;
pub const MACRO_SAMPLES_PER_WAVE: u64 = 4;
pub const FINE_ABOVE_M: u64 = MACRO_SAMPLES_PER_WAVE * MACRO_NODE_M;
/// A BODY ALWAYS HOLDS AT LEAST ONE FINE OCTAVE, so the constraint below never divides by zero: the
/// table halves until the wavelength is at or under [`SHORT_WAVE_M`], so the finest octave's
/// wavelength is at most twice that — far under the fine threshold.
const _: () = assert!(2 * SHORT_WAVE_M < FINE_ABOVE_M);

/// ★ THE RIDGED BAND — slice 8a, stage 2 (`slice_8a_design.md` §1.3 and §2.2; `04_detail_rungs.md`
/// §4.4 rule 2; ruling V13 L5). In the middle of the spectrum an octave stops being a smooth wave
/// and becomes a CREST: `1 − |n|`, re-centred. A smooth octave makes a lump; a ridged one makes a
/// LINE, which is what a spur and a gully are.
///
/// ★ **THE BAND IS STATED IN METRES, NEVER IN INDICES**, so it does not move when the table's ends
/// move: an octave is ridged while its wavelength lies in `[RIDGE_LO_M, RIDGE_HI_M]`. A spur is
/// about a kilometre long and a range's flank about ten, wherever you stand.
///
/// **THE PAIR, AND WHERE IT COMES FROM.** The band is the SPECTRUM'S OWN WIDTH: the octaves whose
/// bump argument `x = (o − O_PEAK)·ln 2` stands within ONE [`SIGMA`] of the peak. That is
/// `|o − O_PEAK| ≤ SIGMA / ln 2 = 2.02`, so it is the peak octave and the two either side of it —
/// and in WAVELENGTH it is the peak's own wavelength times and divided by `e^SIGMA = 4.0552`. The
/// peak's wavelength on a table that clamps is `LONG_WAVE_CAP_M >> O_PEAK = 3 125 m`, so:
///
/// ```text
///    RIDGE_HI_M = 3 125 × 4.0552 = 12 672 m        RIDGE_LO_M = 3 125 ÷ 4.0552 = 771 m
/// ```
///
/// On the home planet that selects octaves 5..=9 — 12 500 m, 6 250 m, 3 125 m, 1 562.5 m and
/// 781.25 m — which is EXACTLY the band `RIDGED_BAND = 5..=9` the M1b-2 measurement was taken on
/// (`09_step0_results.md`), so what the owner sees is the field that instrument predicted. The
/// margin is the same at both ends by construction: 12 500 m stands 1.38 % inside the top and
/// 781.25 m stands 1.38 % inside the bottom, while the octaves next out (25 000 m and 390.6 m) stand
/// a whole octave clear.
///
/// **What 8a does NOT ship.** `04` §4.4 stretches a ridged octave's sample frame along the flow
/// direction, which is 8c's solve. 8a ships the ridge ISOTROPIC — no stretch — which is what M1b-2
/// measured. 8c adds the stretch and moves no address.
///
/// ★ **THE LIFT.** Ridged noise is ONE-SIDED: its mean stands above zero, so the ground RISES by
/// roughly the band's own summed amplitude where the crests stand. That is not an error; it is what
/// makes a crest. The band of the ladder still holds it, because `|ridged| ≤ 1` keeps every octave
/// inside `[−a, +a]` and the ladder is sized on `Σ|a|` (stage 1, §2.6).
pub const RIDGE_HI_M: u64 = 12_672;
pub const RIDGE_LO_M: u64 = 771;
const _: () = assert!(RIDGE_LO_M < RIDGE_HI_M);
/// The ridged band stands INSIDE the fine band: the macro solve replaces the coarse octaves whole,
/// so a ridged coarse octave would be work 8c throws away.
const _: () = assert!(RIDGE_HI_M < FINE_ABOVE_M);

/// ★ THE SAMPLING RULE PER OCTAVE KIND — ruling T7 rule 1 (owner, 2026-09-17), the cure for the
/// MEASURED rung disagreement of 1.28 px at the pairs that drop a CREST.
///
/// **What a rung drops, and why.** A rung hands the ground to the next coarser rung where its own
/// cell falls under a pixel, and the coarser rung draws the same ground with the fine octaves left
/// out (ruling V9). The step the handover makes is exactly the octaves the coarser rung drops, and
/// the ladder's tolerance for that step is ONE CELL OF THE RUNG THAT TAKES OVER — one pixel at that
/// rung's own switch distance, which is the line `rung_disagreement` measures on.
///
/// **The rule, ONE FUNCTION of a wavelength, a kind and a cell** ([`octave_survives`], stated once
/// for the final table at slice 8a stage 5). An octave survives on rung `L` while its wavelength
/// covers [`survival_cells_per_wave`] cells of that rung; a RIDGED octave survives while it covers
/// HALF that many, so it lives ONE RUNG LONGER and is dropped where its crest stands under a cell of
/// the rung that takes over. NOTHING ELSE SELECTS AN OCTAVE ANYWHERE IN THE CRATE: the draw walks
/// the table once with that function and stores the answer per rung, and every reader — the charter,
/// the bounds, the client's crossfade — reads the stored row.
///
/// ★ **WHERE THE TWO COMES FROM — a bound, not a taste.** The recipe folds a ridged octave as
/// `1 − |n|`, doubled and re-centred: `ridged = 1 − 2|n|` (`vd_recipe::height::octave_term`). Read
/// the fold's own `2` twice over:
///
/// 1. **The first derivative doubles.** `d(1 − 2|n|) = ∓2·dn`, so at one wavelength and one
///    amplitude a crest is twice as steep as the round wave it is made from, and it sweeps the whole
///    of `[−a, a]` twice per wavelength instead of once.
/// 2. **The residual doubles where the noise lives.** A value noise is ZERO on every lattice point
///    of its own octave and small over most of the ground between them; the fold maps that same
///    small `|n|` onto the crest's own peak. Term for term the two residuals obey
///    `r_ridged ≥ a − 2·r_smooth`: wherever the smooth term stands under a quarter of its amplitude
///    — which the noise does over most of the ground — the crest's term stands over half of it, at
///    least TWICE the smooth one. Nothing but the fold's `2` enters that line.
///
/// A cell doubles from rung to rung, so twice the residual needs exactly ONE MORE RUNG of life to
/// stand under one cell again. That is the whole of the rule.
///
/// **Example.** The home planet's 3 125 m crest carries 198.8 m of amplitude. Under the count rule
/// it was dropped between rung 6 and rung 7, where a rung-7 cell is 128 m — the step stood 1.55
/// cells high and the pilot saw the hillside jump as she flew out. It now lives on rung 7 and is
/// dropped between rung 7 and rung 8, where a cell is 256 m: 0.78 of a cell, under the line.
///
/// **THE CELLS PER WAVELENGTH ARE READ FROM THE TABLE, NEVER TYPED.** The count rule this replaces
/// (keep `octave_count − rung` octaves) IS a wavelength rule at one constant, because the table
/// halves its wavelength per index exactly as the ladder halves its cell per rung: the finest live
/// octave at every rung covers `λ(finest) / cell_m(0)` cells of it. On the home planet the finest
/// octave is 48.83 m and a rung-0 cell is one metre, so the constant is 48.83 — the "49 cells per
/// wavelength" `slice_8a_design.md` §1.0 measured. A body whose table ends elsewhere states its own
/// constant, and no rung below the crest's band moves one byte.
///
/// ★ **AND STAGE 5 KEEPS IT.** `slice_8a_design.md` §1.7 item 1, written before ruling T7, proposed
/// replacing this constant with a typed `LADDER_SAMPLES_PER_WAVE = 4` — Nyquist with a factor of two
/// of margin. Ruling T7 (owner, 2026-09-17) is NEWER and derives the number from the table instead,
/// and the judge MEASURED that rule at 0.64 px against a line of 1 px.
///
/// The arithmetic of the difference: 48.83 / 4 is 12.2, which is 3.6 halvings, so a typed 4 would
/// hold EVERY octave about three and a half rungs longer than it is held today. The step a handover
/// makes is the dropped octaves' own amplitude, and the ladder's tolerance for it is one cell of the
/// rung that takes over — which is exactly the line T7 rule 1 was written to keep. A rule that hands
/// over three rungs late makes that step many cells high: the defect T7 cured, put back by a typed
/// number. (The 0.64 px is a measurement; what a typed 4 would read is UNMEASURED, and no reason to
/// measure it — the owner ruled the derived rule.) So the table states the constant, and stage 5
/// changes the SHAPE of the rule — one function, and the alias made explicit — and not the number.
fn survival_cells_per_wave(waves: &[Gf; OCTAVES], count: usize) -> Gf {
    waves[count - 1] / Gf::from_i64(i64::from(cell_m(0)))
}

/// ★ NYQUIST, IN CELLS PER WAVELENGTH — the line the ALIAS row is judged against, and the only
/// number here that is not read from the table. A sampled wave needs one cell for its crest and one
/// for its trough, so TWO cells per wavelength is the floor under which a lattice draws a beat of
/// its own spacing and not the wave. The world never reads it: the survival rule asks for 48.83 and
/// stops long before it. It is what says how far past honest the top rungs' one kept octave stands
/// (`live_octaves_at`, the owner's ruling T6 ask 3).
pub const NYQUIST_CELLS_PER_WAVE: f64 = 2.0;

/// ★ THE SURVIVAL RULE ITSELF — ONE FUNCTION OF (wavelength, kind, cell), and the only place in the
/// crate that decides whether an octave lives on a rung (slice 8a stage 5).
///
/// A smooth octave reaches its own wavelength; a CREST reaches twice it, which is exactly one rung
/// of the ladder, for the two reasons the module doc above derives from the fold's own `2`. The test
/// is `reach ≥ cells · cell`, so an octave that covers exactly the required cells LIVES.
///
/// **Example.** The home planet's crest at 3 125 m reaches 6 250 m. At rung 7 a cell is 128 m and
/// the table asks 48.83 cells, so the threshold is 6 250 m: the crest lives, by nothing to spare.
/// At rung 8 the threshold doubles to 12 500 m and the crest is dropped — where the step it makes
/// stands at 0.78 of a rung-8 cell, under the ladder's own tolerance.
fn octave_survives(wave_m: Gf, kind: Gi, cell_m: Gf, cells_per_wave: Gf) -> bool {
    let reach = if kind == OCTAVE_RIDGED {
        wave_m * Gf::from_i64(2)
    } else {
        wave_m
    };
    reach >= cells_per_wave * cell_m
}

/// ★ WHAT THE SURVIVAL RULE ANSWERS FOR A WHOLE BODY: one row per rung, drawn once
/// ([`live_octaves_at`]).
struct RungOctaves {
    /// How many octaves each rung keeps — a PREFIX of the table, never fewer than one.
    live: [u8; RUNGS],
    /// ★ WHETHER THAT RUNG'S ONE OCTAVE IS AN ALIAS: true where the rule kept NOTHING and the rung
    /// was given the widest octave anyway (the owner's T6 ask 3).
    alias: [bool; RUNGS],
    /// ★ THE RATIO THE OWNER ASKED TO SEE PRINTED: the cells of that rung one wavelength of its
    /// FINEST LIVE octave covers, as a word at the noise's fraction bits. Under two, the field
    /// aliases.
    cells_per_wave: [Gi; RUNGS],
}

/// ★ THE LIVE OCTAVE COUNT AT EVERY RUNG, drawn ONCE with the body (the survival rule above).
///
/// The answer is a PREFIX of the table at every rung, and it stays one under the per-kind rule: the
/// table halves its wavelength per index, so a ridged octave's doubled wavelength is exactly the
/// wavelength of the octave BEFORE it, and the test's own answer never rises as the index grows.
/// The scan therefore stops at the first octave that fails.
///
/// ★ **THE ALIAS AT THE RUNGS PAST THE TABLE** (`slice_8a_design.md` §1.7 item 2, and THE OWNER'S
/// OWN DECISION — ruling T6 ask 3, 2026-09-16: *"let's try that and change if quality will not be
/// good or we can improve"* — "keep ONE aliased octave at the rungs past the table, the ratio
/// printed, 8c named as the fix").
///
/// Past the table the rule keeps NOTHING: on the home planet a rung-14 cell is 16 384 m and the
/// widest octave is 400 km, which covers 24.41 cells where the table asks 48.83. The rung is then
/// given EXACTLY ONE octave — THE WIDEST, which is index 0, because the table is drawn coarsest
/// first — and the row records that the octave is an ALIAS and how many cells its wavelength covers.
/// On the home planet that is rungs 14 to 18, at 24.41, 12.21, 6.10, 3.05 and 1.53 cells per
/// wavelength — MEASURED by the row's own test. The last one stands under two cells, which is under
/// Nyquist ([`NYQUIST_CELLS_PER_WAVE`]) and so is not a wave the lattice can carry at all; the four
/// over it are drawn honestly but at a sixteenth to a half of the sampling the rule asks for. Either
/// way the globe seen from orbit carries the blotch pattern of ONE 400 km octave where it wants
/// continents.
///
/// `04_detail_rungs.md` §4.3 says the right count there is ZERO. Zero is right only once 8c's macro
/// field `Z` carries the far shape; until then a bare sphere is a VISIBLE wrong and one aliased
/// octave is a known, bounded and — since this row — MEASURED one. **8c IS THE FIX**, and
/// [`BodyDefinition::aliases_at`] and [`BodyDefinition::cells_per_wave`] are what an instrument, a
/// picture's stamp and a report print until it lands.
fn live_octaves_at(
    waves: &[Gf; OCTAVES],
    octaves: &[Octave; OCTAVES],
    count: usize,
) -> RungOctaves {
    let cells = survival_cells_per_wave(waves, count);
    let mut rows = RungOctaves {
        live: [1u8; RUNGS],
        alias: [false; RUNGS],
        cells_per_wave: [Gi::ZERO; RUNGS],
    };
    let mut rung = 0usize;
    while rung < RUNGS {
        let cell = Gf::from_i64(i64::from(cell_m(rung as u8)));
        let mut keep = 0usize;
        let mut o = 0usize;
        while o < count {
            if !octave_survives(waves[o], octaves[o].kind, cell, cells) {
                break;
            }
            keep = o + 1;
            o += 1;
        }
        // ★ THE ALIAS: nothing qualified, so the rung keeps the WIDEST octave and SAYS SO.
        rows.alias[rung] = keep == 0;
        let kept = if keep == 0 { 1 } else { keep };
        rows.live[rung] = kept as u8;
        rows.cells_per_wave[rung] = q28_of(waves[kept - 1] / cell);
        rung += 1;
    }
    rows
}

/// ★ THE PER-COLUMN ROUGHNESS FACTOR — slice 8a, stage 3 (`slice_8a_design.md` §1.2 and §2.3;
/// `03_erosion_rivers.md` §4.2.4; ruling V13 L5). ONE downward-only multiplier per column on the
/// FINE octaves alone. It is what makes a plain flat and a range rough with ONE table: a craton's
/// fine RMS slope is `0.70 × M_MIN`, and the orogen next to it keeps the whole 0.70.
///
/// `M_MIN = 0.06` gives a craton `0.70 × 0.06 = 0.042`, which is 2.4° — Earth's plains
/// (`03_erosion_rivers.md` §4.2.4). The word is `round(0.06 · 2²⁸) = 16 106 127`, DERIVED here by
/// the same `share_of` every other share goes through, never typed.
pub const M_MIN: f64 = 0.06;
/// ★ THE PLACEHOLDER'S BAND (`slice_8a_design.md` §1.2). `03` reads the factor from the GRADIENT of
/// 8c's macro field `Z`. 8a has no `Z`, so its raw reading is ONE MORE SLOW OCTAVE of the recipe's
/// own noise, from its own salt, at the continental wavelength the biome noises already use — 20 km
/// to 120 km. 8c swaps the one function for the macro field and MOVES NO ADDRESS, because the
/// ladder's band is derived from `Σ|a|` at the factor's ceiling of one, which both fields obey.
pub const ROUGH_WAVE_LO_M: u64 = 20_000;
pub const ROUGH_WAVE_HI_M: u64 = 120_000;
const _: () = assert!(ROUGH_WAVE_LO_M < ROUGH_WAVE_HI_M);
/// The placeholder's field is a CONTINENTAL swell: its wavelength stands well over the coarsest
/// octave the fine band holds, so the factor is a property of a region and never of a hillside.
const _: () = assert!(ROUGH_WAVE_LO_M > RIDGE_HI_M);

/// ★ THE CAP-ROCK BENCH — slice 8a, stage 4 (`slice_8a_design.md` §1.4 and §2.4;
/// `03_erosion_rivers.md` §7.3 rule 2). The SHAPE alone: the surface is pulled toward the nearest
/// BED TOP, and a bed top stands at a FIXED RADIUS. The strata table is untouched — a substance
/// still reads its DEPTH under the surface, and 8d's rock map is what gives a bed its own hardness.
///
/// ★ **THE BED SPACING IS DERIVED, NEVER DRAWN.** A bench stands where a soft bed lies over a hard
/// one, and the body already draws exactly one bed thickness: `StrataTable::sediment_m`, 20 m to
/// 80 m, which is the same sediment `03` §7.3 rule 2 names. `04` §4.5 states the spacing between two
/// hard tops as "two thicknesses or more" — a soft bed and the hard bed beneath it — so the spacing
/// is TWO of the body's own bed. On the home planet the sediment is 69 m, so the beds stand 138 m
/// apart and a walker crosses one every 138 m of altitude.
pub const BEDS_PER_SPACING: u64 = 2;
/// ★ HOW MANY CELLS A BENCH'S TREAD MUST COVER before the terrace is drawn at a rung: FOUR, which is
/// Nyquist ([`NYQUIST_CELLS_PER_WAVE`]) with a factor of two of margin. A tread under four cells is
/// a ripple the rung cannot draw, and drawing it is a step the eye catches.
///
/// ★ **IT IS THE BENCH'S OWN NUMBER, AND NOT THE OCTAVE TABLE'S** (corrected at slice 8a stage 5).
/// `slice_8a_design.md` §1.7 item 1 proposed four for the octave survival rule as well; ruling T7
/// (owner, 2026-09-17, and newer than the design) derives that one FROM THE TABLE instead — 48.83
/// cells on the home planet ([`survival_cells_per_wave`]) — and stage 5 kept it. The two questions
/// are different: a tread is a STEP the extractor must stand on a cell boundary, and an octave is a
/// WAVE whose dropping must make a step under one cell of the rung that takes over. Two questions,
/// two derivations, neither number typed twice.
///
/// ★ THE RUNG RULE, DERIVED FROM THE ANGLE OF REPOSE. Half a spacing of ALTITUDE becomes
/// `(S/2) / tan θ` of GROUND on a hillside, and the spectrum's own anchor says a hillside stands at
/// the angle of repose ([`TAN_REPOSE`]). So the tread covers
/// `TERRACE_SAMPLES_PER_TREAD` cells while `S ≥ 2 · TAN_REPOSE · TERRACE_SAMPLES_PER_TREAD · cell`,
/// which on the home planet's 138 m beds is 5.6 cells of the rung: full strength to rung 3, fading
/// through rung 4, and gone by rung 5 (32 m cells). The draw computes it; nothing here is typed but
/// the two counts above, and both are stated with their own derivation.
///
/// The FADE is linear over the last rung, as `S/cell` falls from twice the threshold to it, so the
/// bench never appears or vanishes between two rungs and its own share of a handover's step is the
/// fade's step and not the whole bench (`04` §4.5).
pub const TERRACE_SAMPLES_PER_TREAD: u64 = 4;
/// ★ THE FALLOFF'S OWN MAXIMUM PULL, as a share of half a spacing: the largest value of
/// `u · (1 − fade(u))` over `u ∈ [0, 1]`, which stands at the root of
/// `g(u) = 1 − 36u⁵ + 75u⁴ − 40u³` (the derivative of `u · f(u)`), `u = 0.398 126`. The maximum is
/// 0.273 032 098; the constant below is rounded UP at the sixth decimal so it is an UPPER BOUND, and
/// the stage's own test finds the maximum by a dense scan and refuses a constant under it.
pub const TERRACE_PULL_SHARE: f64 = 0.273_033;
/// ★ THE TERRACE'S LIPSCHITZ SLOPE, RE-DERIVED FOR THE QUINTIC and NOT copied from `04` §4.5's
/// cubic. With `f = 1 − fade`, `d(terrace)/dh = 1 − q · g(u)` where `g(u) = d(u·f(u))/du =
/// 1 − 36u⁵ + 75u⁴ − 40u³`. `g'(u) = −60u²(3u − 2)(u − 1)`, so the only interior extreme is at
/// `u = 2/3`, where `g = −7/9` EXACTLY. The terrace therefore amplifies every variation under it by
/// at most `1 + (7/9)·q`.
///
/// ★ **THAT IS LARGER THAN `04`'s 0.6875, not smaller** — `slice_8a_design.md` §8 item 9 expected
/// the quintic's constant to be smaller, and it is not. A bound copied from `04` would have been a
/// bound quietly short, which is why the stage's test scans the real first difference against both.
pub const TERRACE_LIP_NUM: i64 = 7;
pub const TERRACE_LIP_DEN: i64 = 9;
/// ★ THE STRENGTH'S CEILING: `1 − M_MIN`. At a bed top the terrace's own slope factor is exactly
/// `1 − q`, so a strength of `1 − M_MIN` leaves the tread standing at the same share of its
/// hillside that the world's flattest ground stands at ([`M_MIN`], a craton). It cannot be more: a
/// strength of one folds the surface over, and an overhang is 8f's question and not 8a's.
///
/// ★ **AND THE LADDER USUALLY BINDS FIRST.** The draw SOLVES the strength, as stage 1 solves
/// `S_PEAK`: the largest strength at which EVERY handover's step still stands under one cell of the
/// rung that takes over (ruling T7's own line). On the home planet that is 0.560, well under the
/// ceiling — and inside the `[0.4, 0.7)` band `slice_8a_design.md` §2.4 proposed as a draw, which is
/// a cross-check that could have failed. A stronger bench would push a rung pair over the ladder's
/// tolerance and wake ruling T7's rules 2 and 3; the world carries the strongest bench it can carry
/// without a seam.
pub const TERRACE_STRENGTH_CEILING: f64 = 1.0 - M_MIN;
/// THE MARGIN the solve leaves for the bound's own roundings: TWO GAP STEPS, one for each rounded
/// side of the step bound (the two pull words and the Lipschitz product). It is 15.6 mm against a
/// cell of metres, and it is what makes the integer bound's promise survive the float solve.
pub const TERRACE_MARGIN_STEPS: i64 = 2;
/// The units of the length format a step bound adds for its own roundings: one for the difference of
/// two upward-rounded pull words, one for the truncation of the Lipschitz product. Together they are
/// 5.8 × 10⁻¹¹ m.
const TERRACE_BOUND_MARGIN: i64 = 2;
/// ★ THE UNITS THE INTEGER FALLOFF MAY BE OUT BY, and why a bound derived from the REAL terrace owes
/// an allowance. `f = 1 − fade(u)` is computed in whole words: `u` is truncated once (one unit,
/// which the quintic's own slope of at most 15/8 turns into two), and `fade` truncates the four
/// products it shifts back. Six units at the noise's bits, together.
///
/// MEASURED: a dense scan of the integer terrace's own first difference over four beds of the home
/// planet reads 1.435 834 against the real bound's 1.435 801 — a hundred-thousandth, which is this
/// allowance and not a broken Lipschitz constant.
const TERRACE_FADE_ROUND_UNITS: i64 = 6;

/// ★ THE CRUST'S LONG-TERM YIELD STRESS, in pascals — the ONE constant of THE RELIEF LAW (the
/// landform arc, slice 8b stage 3; ruling T6 ask 1; `02_planet_layout.md` §4.3).
///
/// A mountain of height `h` presses its own root with `ρ_c·g·h`. Past the crust's long-term yield
/// stress the root flows and the mountain spreads under its own weight, so the tallest relief a body
/// can carry is `σ_y / (ρ_c·g)`.
///
/// **THE PUBLISHED SOURCE, stated honestly.** `σ_y` is a CALIBRATION ON ONE BODY — Everest:
/// `2 800 kg/m³ × 9.81 m/s² × 8 850 m = 243.1 MPa` (COMPUTED, `02_planet_layout.md` §4.3). Its value
/// lands in the same order of magnitude as laboratory rock strengths, and that agreement is a sanity
/// check and NOT a validation: an unconfined compressive strength and the ductile flow limit under a
/// mountain root are two different properties. What is physical, and what this law uses, is the
/// `1/g` TREND. The spread over the three bodies that have actually built mountains is Earth 100 %,
/// Mars 94 %, Venus 112 %.
pub const CRUST_YIELD_STRESS_PA: f64 = 243.1e6;
/// ★ THE CRUST'S DENSITY AS A SHARE OF THE BODY'S BULK DENSITY — **ASSUMED, PENDING THE OWNER**
/// (`slice_8b_design.md` §8 ask 3, recommendation (b), taken as assumed so stage 3 can be built).
///
/// The census holds a body's BULK density and never its crust's. Earth's continental crust is
/// 2 800 kg/m³ against the bulk density of a 1.000 M⊕ body at the home planet's census radius,
/// 5 514.7 kg/m³, so the share is `2 800 / 5 514.7 = 0.507 74` (COMPUTED). A denser body then gets a
/// denser crust, which is the trend the alternative — a flat 2 800 kg/m³ for every body — misses:
/// `02_planet_layout.md` §4.3's own super-earth row says the flat constant OVERSTATES that body's
/// ceiling by exactly this term. On the home planet the two answers agree to four figures, so
/// nothing visible turns on the choice today and the law is right for the next body.
pub const CRUST_DENSITY_SHARE: f64 = 0.507_74;
/// ★ THE SHAPE BOUND'S SHARE OF THE RADIUS, cited to an OBSERVED SMALL BODY and never to Earth
/// (`06_laws_integration.md` §3.4): Vesta carries about 20 km of relief on a 260 km radius, which is
/// `0.077 R`.
///
/// It is the arm that binds on a SMALL body, where the strength bound is enormous — COMPUTED for a
/// 100 km moon at 3 000 kg/m³: `g = 0.084 m/s²` and the strength bound is 883 km, EIGHT TIMES THE
/// MOON'S OWN RADIUS. A strength-only law would ask for a crust deeper than the body, and
/// `Ladder::for_radius` refuses a crust that reaches the centre — which DELETES every small round
/// body from the world. COMPUTED in `06` §3.4: the two arms cross at a radius of about 939 km, so a
/// planet is strength-limited and a moon is shape-limited.
pub const SHAPE_RELIEF_SHARE: f64 = 0.077;

/// The fraction bits of the RADIUS's reciprocal ([`BodyDefinition::radius_recip`]): a whole word less
/// the headroom a two-word product needs, so the column bound's divide is one multiply.
pub const RADIUS_RECIP_BITS: u32 = 62;
/// The fraction bits of the CAVERN WAVELENGTH's reciprocal: the lattice point it makes is under 2⁵⁰
/// at the largest legal body, and the wavelength's own error is then under one part in 2⁴³.
pub const CAVERN_RECIP_BITS: u32 = 48;
/// The tube carvers' region edge, in metres. A POWER OF TWO, so a point's region index is a shift and
/// not a division (the charter holds the shift).
pub const TUBE_REGION_M: u32 = 512;
const _: () = assert!(TUBE_REGION_M.is_power_of_two());

/// The salts of the seed tree: each part of the body draws from its own stream, so adding a draw to
/// one part never moves another.
pub(crate) mod salt {
    pub const OCTAVES: u64 = 0x5e_ed_01;
    pub const SEA: u64 = 0x5e_ed_02;
    pub const STRATA: u64 = 0x5e_ed_03;
    pub const BIOME: u64 = 0x5e_ed_04;
    pub const CAVERN: u64 = 0x5e_ed_05;
    pub const TUBES: u64 = 0x5e_ed_06;
    pub const NOISE: u64 = 0x5e_ed_07;
    /// ★ THE ROUGHNESS FIELD's own stream (slice 8a stage 3). A NEW salt never moves an existing
    /// draw: every octave, sea, stratum, biome and cave the body already drew reads the same words
    /// it read before this one existed.
    pub const ROUGHNESS: u64 = 0x5e_ed_08;
    /// ★ THE CAP-ROCK BENCH's own stream (slice 8a stage 4): the bed stack's PHASE, and nothing
    /// else — the spacing is derived from the strata and the strength is solved from the ladder. A
    /// NEW salt never moves an existing draw.
    pub const BENCH: u64 = 0x5e_ed_09;
    /// ★ THE INITIAL LAND's stream (slice 8c stage C2): the plate sites, their drifts, their
    /// affinities and ages, the crust's named scatter. Read by the solve once per body, never by
    /// the draw of the body itself, so no chunk byte moved when it was added.
    pub const LAND: u64 = 0x5e_ed_0a;
}

/// The cave parameters, as the kernels read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caves {
    /// The cavern field's lattice wavelength in whole metres (every fourth cell at rung 0 is 4 m; the
    /// field is smooth across many cells). The rung test reads it as metres.
    pub(crate) cavern_wavelength_m: u32,
    /// `floor(2^CAVERN_RECIP_BITS / wavelength_m)`: the lattice point is one two-word product.
    pub(crate) cavern_recip: Gi,
    /// The value above which a cell is hollow, at the noise's fraction bits.
    pub(crate) cavern_threshold: Gi,
    /// How many GAP STEPS of hollow one unit of the field above the threshold opens.
    pub(crate) cavern_scale_steps: Gi,
    /// The depth band caves live in, in metres under the surface.
    pub(crate) min_depth_m: u32,
    pub(crate) max_depth_m: u32,
    /// The tube carvers' region edge in metres, the same edge in gap steps, the shift from gap steps
    /// to a region index, and their seed.
    pub(crate) tube_region_m: u32,
    pub(crate) tube_region_steps: Gi,
    pub(crate) tube_region_shift: u32,
    pub(crate) tube_seed: u64,
    /// A tube's radius in gap steps.
    pub(crate) tube_radius_steps: Gi,
}

/// The biome field's parameters. The two slow noises are stated as OCTAVES of the one octave sum, with
/// the amplitude of one noise unit, so the biome reads the field through the same kernel the hills do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiomeField {
    pub(crate) temperature: Octave,
    pub(crate) humidity: Octave,
    /// Above this height over the sea, in gap steps at [`LENGTH_BITS`], a column is highland whatever
    /// its climate.
    pub(crate) highland_above: Gi,
    /// `floor(2^RADIUS_RECIP_BITS / (highland_above_steps + one metre))`: the height share the
    /// temperature reads is one multiply, never a divide.
    pub(crate) highland_recip: Gi,
}

/// ★ THE BODY'S PHYSICAL FACTS, AS WHOLE NUMBERS — what the generator reads of the charter the
/// body's OWN REALM authored (the landform arc, slice 8b; ruling V13 L12; SL10 as ruling V13 L12
/// widened it: the static shape is a function of the seed, the address AND THE BODY'S CHARTER).
///
/// It is the SUBSET of `vd_core::look::BodyCharter` that a draw or a kernel actually reads, and it
/// carries WHOLE NUMBERS only. The author floors them ONCE, in the census's own crate, and every
/// reader — the body's own shard and every client that draws that body — reads the same integer.
/// That is what makes the no-drift gate hold: the client's body is built from the same two integers
/// the shard used, so its chunks equal the shard's byte for byte.
///
/// **This crate names no motion crate** (SL10 clause 2, and the workspace's dependency rule), so
/// the caller does the copy: `BodyFacts::new(charter.gravity_mm_s2, charter.bulk_density_kgm3)`.
/// The census's own full-precision row is `vd_physics::worldgen::BodyFacts`, and the door between
/// the two is `vd_physics::worldgen::quantise_u32`. The two records never meet, and this one never
/// holds a float.
///
/// **Example.** The home planet states 9 818 mm/s² and 5 513 kg/m³. The relief law reads those two
/// numbers and caps the planet's mountains at 8 840 m, where the old lottery let them reach 16 454 m.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyFacts {
    /// Surface gravity, whole mm/s². One step is 1 part in 9 818 on the home planet, which moves the
    /// relief cap by under a metre. ZERO IS REFUSED: a body with no weight has no strength bound,
    /// and a division by zero inside the fence is a defect, never a bound.
    pub gravity_mm_s2: u32,
    /// Bulk density `M / ((4/3)πR³)`, whole kg/m³. ZERO IS REFUSED, for the same reason.
    pub bulk_density_kgm3: u32,
}

impl BodyFacts {
    /// The two words, as the realm's charter states them.
    #[must_use]
    pub const fn new(gravity_mm_s2: u32, bulk_density_kgm3: u32) -> BodyFacts {
        BodyFacts {
            gravity_mm_s2,
            bulk_density_kgm3,
        }
    }

    /// Whether the law can read these facts at all: neither word is zero. A body whose realm states
    /// a zero is REFUSED by [`BodyDefinition::from_seed`], exactly as a client refuses a surface
    /// that arrives with no charter — the lane never invents a gravity, and neither does the draw.
    #[must_use]
    fn usable(self) -> bool {
        (self.gravity_mm_s2 > 0) & (self.bulk_density_kgm3 > 0)
    }
}

/// ★ THE FACTS OF THE CRATE'S OWN UNNAMED TEST BODIES, stated once so every module reads the same
/// rock (slice 8b stage 3). Each `g` is COMPUTED from the body's own bulk density and radius by
/// `g = (4/3)πGρR` — the same relation the census uses — and floored to whole mm/s². A test may
/// divide; the shipped path never computes a gravity, it is told one.
///
/// A 3 km rock at 3 000 kg/m³: `g = 2.516 mm/s²`. Its shape arm is 231 m and its strength arm is
/// 63 800 km, so the rock is shape-limited, which is what a rock is.
#[cfg(test)]
pub(crate) const ROCK_3KM_FACTS: BodyFacts = BodyFacts::new(2, 3_000);
/// A 40 000 km giant at Jupiter's bulk density, 1 326 kg/m³: `g = 14.83 m/s²`.
#[cfg(test)]
pub(crate) const GIANT_FACTS: BodyFacts = BodyFacts::new(14_830, 1_326);

/// ★ THE RELIEF LAW's TWO ARMS, in metres: `(σ_y / (ρ_c·g), 0.077·R)` (ruling T6 ask 1;
/// `slice_8b_design.md` §2.1). The CAP is the lesser of them, and the relief is a draw in
/// `[0.5, 1.0)` of that cap — a share of a bound, so a bound is a bound.
///
/// ONE SOURCE (HR3): the draw calls this and [`BodyDefinition::relief_arms_m`] reads it back, so a
/// measurement can never print a different law from the one the world is built on.
///
/// **The door** (`slice_8b_design.md` §2.0): the charter's two whole numbers enter here and become
/// fenced floats ONCE, at the draw, on the CPU, once per body. No kernel sees them and no card runs
/// them.
fn relief_arms(facts: BodyFacts, radius_m: Gf) -> (Gf, Gf) {
    // mm/s² to m/s², and the bulk density to the crust's — the two steps that turn a stated integer
    // into the law's own units.
    let gravity = Gf::from_i64(i64::from(facts.gravity_mm_s2)) / Gf::from_i64(1_000);
    let crust_density =
        Gf::from_i64(i64::from(facts.bulk_density_kgm3)) * Gf::from_f64(CRUST_DENSITY_SHARE);
    (
        Gf::from_f64(CRUST_YIELD_STRESS_PA) / (crust_density * gravity),
        radius_m * Gf::from_f64(SHAPE_RELIEF_SHARE),
    )
}

/// One round body, fully described. Every field is read inside the crate only (the refuter's
/// finding 18): a body is DRAWN from a seed by [`BodyDefinition::from_seed`] and never assembled from
/// numbers computed elsewhere, so no number from an unfenced crate can enter the recipe as a body.
/// What a host outside the crate may read is behind the accessors below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyDefinition {
    /// The body's own seed (the realm's).
    pub(crate) seed: u64,
    /// ★ THE FACTS THE BODY'S REALM STATED ABOUT ITSELF (slice 8b stage 3), as whole numbers. The
    /// relief law reads them; every later stage of the arc reads more of the charter through this
    /// same word.
    pub(crate) facts: BodyFacts,
    /// The grid on the ladder.
    pub(crate) ladder: Ladder,
    /// The radius on the ladder, in gap steps at [`LENGTH_BITS`].
    pub(crate) radius: Gi,
    /// The same radius in WHOLE gap steps, which the reciprocal below is the reciprocal of.
    pub(crate) radius_steps: Gi,
    /// `floor(2^RADIUS_RECIP_BITS / radius_steps)`: the column bound's one divide, done once.
    pub(crate) radius_recip: Gi,
    /// ★ THE RELIEF THE SEED DREW UNDER THE LAW (slice 8b stage 3), in gap steps at
    /// [`LENGTH_BITS`]: `draw × min(σ_y/(ρ_c·g), 0.077·R)`. It is NOT the band — the band is
    /// `relief_bound_m(0)`, which the spectrum's own amplitudes size — and it is stored because a
    /// measurement must be able to read the law's answer rather than re-derive it.
    pub(crate) relief: Gi,
    /// The sea's radius: the ladder radius plus the seed's sea offset, in gap steps at
    /// [`LENGTH_BITS`].
    pub(crate) sea_radius: Gi,
    /// The cell-count reciprocal of every rung (`vd_recipe::bend::inv_n_of`), so a direction costs no
    /// divide. Entries past the body's own rungs are zero and never read.
    pub(crate) inv_n: [Gi; RUNGS],
    /// The octaves, coarsest first; only the first `octave_count` are live.
    pub(crate) octaves: [Octave; OCTAVES],
    pub(crate) octave_count: u8,
    /// ★ HOW MANY OCTAVES EACH RUNG KEEPS (ruling T7 rule 1, `live_octaves_at`): the survival rule
    /// read once per body, per RUNG, never per column. A rung past the body's own reads the last
    /// entry, which is the top rung's one octave.
    pub(crate) live_at: [u8; RUNGS],
    /// ★ WHICH RUNGS DRAW AN ALIAS (the owner's ruling T6 ask 3): true where the survival rule kept
    /// nothing and the rung was given the widest octave anyway. 8c is the fix; until then the body
    /// states the wrong instead of hiding it.
    pub(crate) alias_at: [bool; RUNGS],
    /// ★ THE RATIO THE OWNER ASKED TO SEE PRINTED, per rung: the cells of that rung one wavelength
    /// of its finest live octave covers, at the noise's fraction bits. Under two, the field aliases.
    pub(crate) cells_per_wave_at: [Gi; RUNGS],
    /// ★ THE PER-COLUMN ROUGHNESS FACTOR's words (slice 8a stage 3): the placeholder's slow octave,
    /// the factor's floor, and the index the FINE band begins at. Every column pass reads them.
    pub(crate) roughness: Roughness,
    /// ★ THE CAP-ROCK BENCH's words (slice 8a stage 4): the datum radius, the bed spacing and its
    /// two reciprocals. The `strength` member carries rung 0's, and [`BodyDefinition::terrace_at`]
    /// swaps in the rung's own from the row below.
    pub(crate) terrace: Terrace,
    /// ★ THE BENCH'S STRENGTH AT EVERY RUNG, faded to zero where a tread falls under
    /// [`TERRACE_SAMPLES_PER_TREAD`] cells. Drawn once per body, read per box, never per column.
    pub(crate) strength_at: [Gi; RUNGS],
    /// ★ THE BENCH'S OWN VALUE BOUND at every rung, in gap steps at [`LENGTH_BITS`]: the furthest
    /// the terrace can carry a surface, `strength × (S/2) × TERRACE_PULL_SHARE`, ROUNDED UP, plus
    /// the integer falloff's own allowance ([`BodyDefinition::terrace_slack`]).
    pub(crate) pull_at: [Gi; RUNGS],
    /// ★ HOW FAR THE INTEGER TERRACE MAY STAND FROM THE REAL ONE at one column, in units of the
    /// length format: `(S/2) × TERRACE_FADE_ROUND_UNITS` for the falloff's own roundings and two for
    /// the pull's. About a millionth of a metre on the home planet — and stated, not assumed.
    pub(crate) terrace_round: Gi,
    /// ★ THE BENCH'S LIPSCHITZ WORD at every rung, at the noise's bits: `1 + (7/9)·strength`,
    /// ROUNDED UP. Every DIFFERENCE bound under the terrace is multiplied by it.
    pub(crate) lip_at: [Gi; RUNGS],
    /// The strata and the caves and the biomes.
    pub(crate) strata: StrataTable,
    pub(crate) caves: Caves,
    pub(crate) biome: BiomeField,
}

/// A draw in `[lo, hi)` metres as a whole number.
fn draw_m(rng: &mut SplitMix64, lo: u64, hi: u64) -> u32 {
    rng.range_u64(lo, hi) as u32
}

/// A draw in `[0, 1)` as a fenced float: the top 53 bits over 2^53, exact.
pub(crate) fn draw_unit(rng: &mut SplitMix64) -> Gf {
    Gf::from_i64((rng.next_u64() >> 11) as i64) / Gf::from_i64(1 << 53)
}

/// A fenced float ROUNDED to a whole number, half upward: the one rounding between the draw and the
/// charter. The draw is never negative where this is used.
fn rounded(v: Gf) -> i64 {
    (v + Gf::HALF).floor().to_i64_floor()
}

/// A fenced float rounded UP to a whole number — the direction a BOUND rounds, so an integer word
/// can never stand under the real number it bounds. A TRUE ceiling, not `floor + 1`: the Lipschitz
/// word multiplies a bound of eight kilometres, so one spare unit at the noise's bits would add
/// thirty micrometres to it and break the promise that a column's span nests inside the body's own
/// band at a rung where every term of the span stands at its ceiling.
fn rounded_up(v: Gf) -> i64 {
    let whole = v.floor();
    if whole < v {
        whole.to_i64_floor() + 1
    } else {
        whole.to_i64_floor()
    }
}

/// A length the draw states in metres, as a word in gap steps at [`LENGTH_BITS`].
fn length_of(metres: Gf) -> Gi {
    Gi::new(rounded(metres * Gf::from_i64(STEPS_PER_M << LENGTH_BITS)))
}

/// A NUMBER the draw states as a word at the noise's fraction bits. The one rounding site for every
/// unitless quantity the body carries — a share in `[0, 1)` and the survival rule's cells per
/// wavelength, which stands over one.
fn q28_of(count: Gf) -> Gi {
    Gi::new(rounded(count * Gf::from_i64(1 << NOISE_BITS)))
}

/// A share the draw states in `[0, 1)`, as a word at the noise's fraction bits.
fn share_of(unit: Gf) -> Gi {
    q28_of(unit)
}

/// One SLOW NOISE of the body, as an octave of unit amplitude: the wavelength in metres turns into
/// the frequency the one octave sum reads (cells per unit direction = the radius over the wavelength).
/// The biome's two climate noises are drawn this way, and so is the roughness field's placeholder
/// (slice 8a stage 3) — one shape, so a slow swell is a slow swell wherever the body reads one.
fn biome_octave(seed: u64, radius_m: Gf, wavelength_m: u32) -> Octave {
    let (frequency_int, frequency_frac) =
        frequency_of(radius_m / Gf::from_i64(i64::from(wavelength_m)));
    // SMOOTH always: a climate is a slow swell, never a crest, and the one-octave sum reads the
    // noise itself back out of it.
    Octave::smooth(seed, frequency_int, frequency_frac, Gi::new(1 << AMP_BITS))
}

/// ★ AN OCTAVE'S FREQUENCY as a real number — cells per unit direction, for a host outside the recipe
/// (a caption, the skyline march's spectrum table). The recipe itself reads the pair.
#[must_use]
pub fn octave_frequency(o: &Octave) -> f64 {
    (Gf::from_i64(o.frequency_int.raw())
        + Gf::from_i64(o.frequency_frac.raw()) / Gf::from_i64(1 << NOISE_BITS))
    .to_f64()
}

/// ★ AN OCTAVE'S AMPLITUDE in metres, for a host outside the recipe. The recipe reads gap steps at
/// [`AMP_BITS`] below the step.
#[must_use]
pub fn octave_amplitude_m(o: &Octave) -> f64 {
    crate::units::metres_of_fixed(o.amplitude, AMP_BITS)
}

/// ★ THE FIRST FINE OCTAVE: the first whose wavelength is under [`FINE_ABOVE_M`]. The wavelengths
/// halve, so the coarse octaves are a PREFIX and one pass over the table names the boundary. A body
/// with no coarse octave at all — a rock whose longest wave is already under the threshold — answers
/// zero, and the whole of its table is the spectrum's.
fn first_fine_octave(waves: &[Gf; OCTAVES], count: usize) -> usize {
    let threshold = Gf::from_i64(FINE_ABOVE_M as i64);
    let mut first = 0usize;
    let mut o = 0;
    while o < count {
        if waves[o] >= threshold {
            first = o + 1;
        }
        o += 1;
    }
    first
}

/// The bump at UNIT peak: `1 / (1 + x² / σ²)` with `x = (o − O_PEAK)·ln 2`. Four operations of the
/// fence and no call.
fn spectrum_base(o: usize) -> Gf {
    let sigma = Gf::from_f64(SIGMA);
    let x = Gf::from_i64(o as i64 - O_PEAK) * Gf::from_f64(LN_2);
    Gf::ONE / (Gf::ONE + x * x / (sigma * sigma))
}

/// ★ `S_PEAK`, SOLVED FROM THE ANCHOR, never typed: the fine octaves' slopes, taken as a root sum of
/// squares, equal `TALUS_RMS × TAN_REPOSE`. A table whose ends move re-solves it, so the promise "the
/// fine roughness stands at the angle of repose" survives a change to the table rather than quietly
/// drifting off it. On the home planet's ten fine octaves it lands at 0.399 78, which the stage's own
/// test pins.
fn spectrum_peak(first_fine: usize, count: usize) -> Gf {
    let mut squares = Gf::ZERO;
    let mut o = first_fine;
    while o < count {
        let base = spectrum_base(o);
        squares += base * base;
        o += 1;
    }
    Gf::from_f64(TALUS_RMS) * Gf::from_f64(TAN_REPOSE) / squares.sqrt()
}

/// One fine octave's amplitude in metres: its slope times its wavelength over a whole turn.
fn spectrum_amplitude_m(s_peak: Gf, o: usize, wave_m: Gf) -> Gf {
    s_peak * spectrum_base(o) * wave_m / Gf::from_f64(TAU)
}

/// A frequency the draw states as a fenced float, as the recipe's pair: the integer part and the
/// fraction at the noise's fraction bits. MEASURED (the integer bench, part 1): a frequency rounded
/// to 2⁻⁸ moved the coarsest lattice point by a ten-thousandth of a cell, which eight kilometres of
/// amplitude turned into metres — so the fraction carries the noise's own 28 bits.
pub(crate) fn frequency_of(frequency: Gf) -> (Gi, Gi) {
    let whole = frequency.floor();
    (
        Gi::new(whole.to_i64_floor()),
        Gi::new(rounded((frequency - whole) * Gf::from_i64(1 << NOISE_BITS))),
    )
}

impl BodyDefinition {
    /// The body's own seed (the realm's).
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The grid on the ladder.
    #[must_use]
    pub const fn ladder(&self) -> &Ladder {
        &self.ladder
    }

    /// How many octaves the seed drew (the live prefix of the table).
    #[must_use]
    pub const fn octave_count(&self) -> u8 {
        self.octave_count
    }

    /// The body's radius on the ladder, in metres — for a host outside the recipe (a caption, a
    /// census). The recipe itself reads the word.
    #[must_use]
    pub fn radius_m(&self) -> f64 {
        crate::units::metres_of_q28(self.radius)
    }

    /// The sea's radius in metres, for a host outside the recipe.
    #[must_use]
    pub fn sea_radius_m(&self) -> f64 {
        crate::units::metres_of_q28(self.sea_radius)
    }

    /// ★ THE FACTS THE BODY'S REALM STATED (slice 8b stage 3) — the two whole numbers the relief law
    /// read. A measurement prints them beside the relief they capped.
    #[must_use]
    pub fn facts(&self) -> BodyFacts {
        self.facts
    }

    /// ★ THE RELIEF THE LAW CAPPED AND THE SEED DREW, in metres. It is NOT the band: the band is
    /// [`BodyDefinition::relief_bound_m`] at rung 0, which the spectrum's own amplitudes size and
    /// which stands a little over this number.
    #[must_use]
    pub fn relief_m(&self) -> f64 {
        crate::units::metres_of_q28(self.relief)
    }

    /// ★ THE RELIEF LAW's TWO ARMS in metres, `(strength, shape)` — what a measurement prints to say
    /// WHICH arm binds on this body. The lesser of the two is the cap the draw multiplies; the same
    /// function computes both here and in the draw, so the two can never part company.
    #[must_use]
    pub fn relief_arms_m(&self) -> (f64, f64) {
        let (strength, shape) = relief_arms(self.facts, Gf::from_f64(self.ladder.radius_m()));
        (strength.to_f64(), shape.to_f64())
    }

    /// The cell-count reciprocal of a rung: what a direction needs instead of a divide. A rung past
    /// the body's own reads the top rung's.
    #[must_use]
    pub(crate) fn inv_n(&self, rung: u8) -> Gi {
        self.inv_n[usize::from(rung.min(self.ladder.rungs - 1))]
    }

    /// The body for `seed` at `look_radius_m`, under the FACTS its own realm states about it;
    /// `None` where the ladder refuses the radius, or where the facts carry a zero. The band — the
    /// crust under the surface and the room over it — is DERIVED from the relief the law caps and
    /// the seed draws, the strata and the caves, so the surface, every stratum and every cave fit
    /// inside the grid.
    ///
    /// ★ **THE CHARTER IS AN ARGUMENT, NOT AN OPTION** (slice 8b stage 3, §4.1). A body cannot be
    /// built without one: the client's lane refuses a surface that arrives with no charter and
    /// counts the refusal, the shard states no surface it cannot state a charter for, and here the
    /// type itself refuses. Both hosts therefore read the SAME two integers, which is what makes the
    /// no-drift gate a measurement rather than a hope.
    #[must_use]
    pub fn from_seed(seed: u64, look_radius_m: f64, facts: BodyFacts) -> Option<BodyDefinition> {
        // ★ THE REFUSAL AT THE DRAW. A zero gravity or a zero density is not a fact a realm can
        // state about a body the law can cap: the strength arm would be a division by zero, and the
        // fence calls that a defect, never a bound. The body is refused, and its realm then states
        // no ground at all rather than ground nobody drew.
        if !facts.usable() {
            return None;
        }
        // The ladder's edge count depends on the radius alone; the band is fixed once the relief is
        // known, below, and the ladder is taken again with it (the same `n` both times).
        let radius_m = Gf::from_f64(Ladder::for_radius(look_radius_m, 1, 1)?.radius_m());

        // ★ THE RELIEF LAW (ruling T6 ask 1, moved here from 8a; `slice_8b_design.md` §2.1). The
        // relief is a DRAW INSIDE A BOUND — `draw × min(strength bound, shape bound)` with the draw
        // in [0.5, 1.0) — where it used to be a share of the radius times a factor in [0.5, 1.5),
        // which could stand half again over its own cap. The strength arm reads the body's stated
        // gravity and density; the shape arm reads its radius. A planet is strength-limited and a
        // moon is shape-limited, and the two cross at about 939 km of radius.
        //
        // The RNG takes exactly ONE draw here, as it always did, so every draw after this one —
        // the coarsest wavelength, the roughness, the sea, the strata — reads the stream at the
        // same position it read before.
        let mut octave_rng = SplitMix64::new(child_seed(seed, salt::OCTAVES, 0));
        let (strength_m, shape_m) = relief_arms(facts, radius_m);
        let relief_cap = strength_m.lesser(shape_m);
        let relief_m = relief_cap * (Gf::HALF + draw_unit(&mut octave_rng) * Gf::HALF);
        // The coarsest wavelength: a quarter to a half of the radius, at least 20 km, at most 400 km.
        let long_share = Gf::from_f64(0.25) + draw_unit(&mut octave_rng) * Gf::from_f64(0.25);
        let long_wave_m = (radius_m * long_share)
            .clamp(Gf::from_i64(20_000), Gf::from_i64(LONG_WAVE_CAP_M as i64));
        // The roughness: how much each finer octave keeps of the one before, in [0.45, 0.55).
        let k_rough = Gf::from_f64(0.45) + draw_unit(&mut octave_rng) * Gf::from_f64(0.10);
        // Halve the wavelength until it is under 30 m or the table is full. The COARSE octaves' weights
        // are the geometric ladder; the FINE octaves' amplitudes come from the slope spectrum below.
        let mut waves = [Gf::ZERO; OCTAVES];
        let mut weights = [Gf::ZERO; OCTAVES];
        let mut count = 0usize;
        let mut wave_m = long_wave_m;
        let mut weight = Gf::ONE;
        let mut weight_sum = Gf::ZERO;
        // Never past the table: the compile-time assertion above proves the cap halves to under the
        // floor within `OCTAVES` steps.
        while wave_m > Gf::from_i64(SHORT_WAVE_M as i64) {
            waves[count] = wave_m;
            weights[count] = weight;
            weight_sum += weight;
            weight *= k_rough;
            wave_m *= Gf::HALF;
            count += 1;
        }
        // ★ THE SLOPE SPECTRUM (slice 8a, stage 1). The fine octaves' amplitudes read their own slope
        // and their own wavelength and never `relief_m`; the coarse ones keep the geometric ladder
        // 8c replaces. THE WHOLE SPECTRUM IS A DRAW: `Octave::amplitude` is the unit the charter
        // already carries, so not one kernel line changes and the card pays nothing for it.
        let first_fine = first_fine_octave(&waves, count);
        let s_peak = spectrum_peak(first_fine, count);
        let mut amplitudes_m = [Gf::ZERO; OCTAVES];
        let mut fine_sum_m = Gf::ZERO;
        let mut o = 0;
        while o < count {
            amplitudes_m[o] = if o < first_fine {
                weights[o] * relief_m / weight_sum
            } else {
                let a = spectrum_amplitude_m(s_peak, o, waves[o]);
                fine_sum_m += a;
                a
            };
            o += 1;
        }
        // ★ THE CONSTRAINT, as ONE `lesser`: what the SPECTRUM spends may not stand over the body's
        // relief. Spending less is lawful, so it is an inequality and never a normalisation — which is
        // exactly why `Σ|a|` and `relief_m` part company and why the band below is re-derived. On the
        // home planet the fine sum is 1 532 m against a relief of 16 454 m, so the scale is ONE and the
        // arm is never taken; a small rock, whose whole table is fine, takes it.
        let fine_scale = Gf::ONE.lesser(relief_m / fine_sum_m);
        // THE ONE ROUNDING into the charter: the frequency as an integer part and a 28-bit fraction,
        // the amplitude in gap steps at `AMP_BITS` below the step (1/32 768 m at the metre rung).
        let mut octaves = [Octave::smooth(0, Gi::ZERO, Gi::ZERO, Gi::ZERO); OCTAVES];
        let mut o = 0;
        while o < count {
            let (frequency_int, frequency_frac) = frequency_of(radius_m / waves[o]);
            let scale = if o < first_fine { Gf::ONE } else { fine_scale };
            let amplitude_m = amplitudes_m[o] * scale;
            // ★ THE RIDGED BAND, BY WAVELENGTH IN METRES (stage 2). The word is an arithmetic MASK
            // the kernel `and`s, never a flag it branches on.
            let ridged = (waves[o] >= Gf::from_i64(RIDGE_LO_M as i64))
                & (waves[o] <= Gf::from_i64(RIDGE_HI_M as i64));
            octaves[o] = Octave {
                seed: child_seed(seed, salt::NOISE, o as u64),
                frequency_int,
                frequency_frac,
                amplitude: Gi::new(rounded(amplitude_m * Gf::from_i64(STEPS_PER_M << AMP_BITS))),
                kind: if ridged { OCTAVE_RIDGED } else { OCTAVE_SMOOTH },
                // ★ THE FINE MASK (stage 3): the per-column roughness factor multiplies this
                // octave, or it does not. The KERNEL reads this word and never compares an index
                // against a split point — the card MEASURED that comparison wrong.
                fine: if o < first_fine {
                    OCTAVE_COARSE
                } else {
                    OCTAVE_FINE
                },
                pad: [Gi::ZERO; 2],
            };
            o += 1;
        }

        // ★ THE SURVIVAL RULE (ruling T7 rule 1), drawn once: which octaves each rung keeps. It
        // reads the table and the ladder's cell sizes alone, so it stands here, in front of the
        // bench's own solve, which reads what each rung drops.
        let rung_octaves = live_octaves_at(&waves, &octaves, count);
        let live_at = rung_octaves.live;
        // ★ Σ|a| — the sum of the DRAWN amplitudes (slice 8a stage 1, §2.6), and the same sum at
        // every rung, in metres. The band below is sized on the first; the bench's solve reads the
        // differences of the rest.
        let amp_unit = Gf::from_i64(STEPS_PER_M << AMP_BITS);
        let mut amplitude_sum = Gi::ZERO;
        let mut o = 0;
        while o < count {
            amplitude_sum += octaves[o].amplitude;
            o += 1;
        }
        let amplitude_sum_m = Gf::from_i64(amplitude_sum.raw()) / amp_unit;
        let mut sum_at_m = [Gf::ZERO; RUNGS];
        let mut rung = 0usize;
        while rung < RUNGS {
            let keep = usize::from(live_at[rung]);
            let mut live_sum = Gf::ZERO;
            let mut o = 0usize;
            while o < keep {
                live_sum += Gf::from_i64(octaves[o].amplitude.raw());
                o += 1;
            }
            sum_at_m[rung] = live_sum / amp_unit;
            rung += 1;
        }
        let radius = length_of(radius_m);
        let radius_steps = radius >> LENGTH_BITS;

        // ★ THE ROUGHNESS FIELD (slice 8a, stage 3) — the PLACEHOLDER of `slice_8a_design.md` §1.2:
        // one more slow octave of the recipe's own noise, SMOOTH, of unit amplitude, from its own
        // salt at a continental wavelength. `one_octave` reads the noise itself back out of a unit
        // amplitude, the recipe maps it to `[0, 1)` and passes it through the quintic `fade` the
        // noise already holds, so no new polynomial enters the fence.
        let mut rough_rng = SplitMix64::new(child_seed(seed, salt::ROUGHNESS, 0));
        let rough_seed = rough_rng.next_u64();
        let rough_wavelength_m = draw_m(&mut rough_rng, ROUGH_WAVE_LO_M, ROUGH_WAVE_HI_M);
        let roughness = Roughness {
            octave: biome_octave(rough_seed, radius_m, rough_wavelength_m),
            m_min: share_of(Gf::from_f64(M_MIN)),
            first_fine: Gi::new(first_fine as i64),
        };

        // The sea: between 40 % of the relief below the ladder radius and 30 % above it.
        // ★ RULING T8 (a), slice 8b stage 6: the owning realm SOLVES the sea's level from the
        // body's water inventory (`crate::sea`) and states it in its charter, but the recipe keeps
        // THIS draw until 8c gives the ground its second hump — the solved level would put almost
        // the whole one-humped globe under water, and a picture with a sea level nothing draws is a
        // lie. 8c's switch is one line: this word reads the stated offset.
        let mut sea_rng = SplitMix64::new(child_seed(seed, salt::SEA, 0));
        let sea_offset =
            (draw_unit(&mut sea_rng) * Gf::from_f64(0.7) - Gf::from_f64(0.4)) * relief_m;
        let sea_radius = length_of(radius_m + sea_offset.floor());

        // The strata.
        let mut strata_rng = SplitMix64::new(child_seed(seed, salt::STRATA, 0));
        let sediments = [Stratum::Sandstone, Stratum::Limestone, Stratum::Shale];
        let strata = StrataTable {
            topsoil_m: draw_m(&mut strata_rng, 1, 4),
            subsoil_m: draw_m(&mut strata_rng, 2, 8),
            sediment_m: draw_m(&mut strata_rng, 20, 80),
            sediment: sediments[strata_rng.range_u64(0, 3) as usize],
            bedrock: Bedrock::ALL[strata_rng.range_u64(0, 5) as usize],
        };

        // The biomes.
        let mut biome_rng = SplitMix64::new(child_seed(seed, salt::BIOME, 0));
        let temperature_seed = biome_rng.next_u64();
        let temperature_wavelength_m = draw_m(&mut biome_rng, 30_000, 120_000);
        let humidity_seed = biome_rng.next_u64();
        let humidity_wavelength_m = draw_m(&mut biome_rng, 20_000, 80_000);
        let highland_above = length_of(relief_m * Gf::from_f64(0.55));
        let biome = BiomeField {
            temperature: biome_octave(temperature_seed, radius_m, temperature_wavelength_m),
            humidity: biome_octave(humidity_seed, radius_m, humidity_wavelength_m),
            highland_above,
            // One metre of floor under the share's divisor, as the float recipe had it, so a body of
            // no relief still reads a share.
            highland_recip: Gi::new(recip_pow2(
                ((highland_above >> LENGTH_BITS) + Gi::new(STEPS_PER_M)).raw() as u64,
                RADIUS_RECIP_BITS,
            ) as i64),
        };

        // The caves: a cavern field in a depth band, and tube carvers in regions.
        let mut cave_rng = SplitMix64::new(child_seed(seed, salt::CAVERN, 0));
        let cave_min_depth_m = draw_m(&mut cave_rng, 8, 30);
        let cavern_wavelength_m = draw_m(&mut cave_rng, 24, 48);
        // ONE rounding: the draw's whole expression, then the charter's word.
        let cavern_threshold =
            share_of(Gf::from_f64(0.62) + draw_unit(&mut cave_rng) * Gf::from_f64(0.08));
        let cavern_scale_steps = Gi::new(20 * STEPS_PER_M);
        let max_depth_m = cave_min_depth_m + draw_m(&mut cave_rng, 200, 400);
        let tube_radius_m = draw_m(&mut cave_rng, 2, 5);
        let tube_region_steps = i64::from(TUBE_REGION_M) * STEPS_PER_M;
        let caves = Caves {
            cavern_wavelength_m,
            cavern_recip: Gi::new(
                recip_pow2(u64::from(cavern_wavelength_m), CAVERN_RECIP_BITS) as i64,
            ),
            cavern_threshold,
            cavern_scale_steps,
            min_depth_m: cave_min_depth_m,
            max_depth_m,
            tube_region_m: TUBE_REGION_M,
            tube_region_steps: Gi::new(tube_region_steps),
            tube_region_shift: tube_region_steps.trailing_zeros(),
            tube_seed: child_seed(seed, salt::TUBES, 0),
            tube_radius_steps: Gi::new(i64::from(tube_radius_m) * STEPS_PER_M),
        };

        // ★ THE CAP-ROCK BENCH (slice 8a, stage 4). THREE numbers and one drawn phase:
        //   * the SPACING is DERIVED from the body's own soft bed ([`BEDS_PER_SPACING`]);
        //   * the DATUM stands under the deepest surface the body can reach, so the bed index is
        //     never negative and the reciprocal's truncation IS the floor the kernel needs;
        //   * the PHASE is the only draw, from the bench's own salt;
        //   * the STRENGTH is SOLVED from the ladder, below.
        let mut bench_rng = SplitMix64::new(child_seed(seed, salt::BENCH, 0));
        let spacing_m = BEDS_PER_SPACING * u64::from(strata.sediment_m);
        let spacing_steps = spacing_m as i64 * STEPS_PER_M;
        // The half spacing is EXACT: a metre is 128 gap steps, so the half is a shift.
        let half_spacing_steps = spacing_m as i64 * (STEPS_PER_M >> 1);
        let phase_steps = bench_rng.range_u64(0, spacing_steps as u64) as i64;
        let deepest_steps = Gi::new(rounded(amplitude_sum_m * Gf::from_i64(STEPS_PER_M)) + 1);
        let datum_steps = radius_steps - deepest_steps - Gi::new(phase_steps);

        // ★ THE RUNG FADE: full strength while the tread covers its cells, linear to zero over the
        // last rung, nothing below (`04_detail_rungs.md` §4.5; [`TERRACE_SAMPLES_PER_TREAD`]).
        let tread_cells =
            Gf::TWO * Gf::from_f64(TAN_REPOSE) * Gf::from_i64(TERRACE_SAMPLES_PER_TREAD as i64);
        let mut fade_at = [Gf::ZERO; RUNGS];
        let mut rung = 0usize;
        while rung < RUNGS {
            let cells =
                Gf::from_i64(spacing_m as i64) / Gf::from_i64(i64::from(cell_m(rung as u8)));
            fade_at[rung] = ((cells - tread_cells) / tread_cells).clamp(Gf::ZERO, Gf::ONE);
            rung += 1;
        }

        // ★ THE STRENGTH, SOLVED FROM THE LADDER — the largest the world can carry without a seam.
        // At the handover from rung `L` the picture's step is the octaves rung `L + 1` drops,
        // AMPLIFIED by the terrace (`1 + (7/9)·q·φ`), plus the bench's own fade step
        // (`q·(φ_L − φ_{L+1})·(S/2)·share`); ruling T7's line is that the step stands under ONE CELL
        // of the rung that takes over. Every term is linear in `q`, so the largest lawful `q` is one
        // division per rung pair and a `lesser` — the same shape stage 1 solves `S_PEAK` with. A
        // rung pair the bench does not live on states nothing, which is why the coarse pairs (whose
        // steps are the biggest) never bind.
        let lip_slope = Gf::from_i64(TERRACE_LIP_NUM) / Gf::from_i64(TERRACE_LIP_DEN);
        let pull_unit_m =
            Gf::from_i64(spacing_m as i64) * Gf::HALF * Gf::from_f64(TERRACE_PULL_SHARE);
        let margin_m = Gf::from_i64(TERRACE_MARGIN_STEPS) / Gf::from_i64(STEPS_PER_M);
        let mut strength = Gf::from_f64(TERRACE_STRENGTH_CEILING);
        let mut rung = 0usize;
        while rung + 1 < RUNGS {
            let dropped = sum_at_m[rung] - sum_at_m[rung + 1];
            let per_q = lip_slope * fade_at[rung] * dropped
                + (fade_at[rung] - fade_at[rung + 1]) * pull_unit_m;
            if per_q > Gf::ZERO {
                let room = Gf::from_i64(i64::from(cell_m((rung + 1) as u8))) - dropped - margin_m;
                strength = strength.lesser(room / per_q);
            }
            rung += 1;
        }
        let strength = strength.greater(Gf::ZERO);

        // The rung rows. THE STRENGTH ROUNDS DOWN, so the kernel can never pull harder than the
        // solve allowed; the two BOUNDS read that same word back and round UP, so neither can ever
        // stand under what the kernel does.
        let noise_one = Gf::from_i64(1 << NOISE_BITS);
        let bound_unit = Gf::from_i64(STEPS_PER_M << LENGTH_BITS);
        let terrace_round =
            Gi::new(half_spacing_steps * TERRACE_FADE_ROUND_UNITS + TERRACE_BOUND_MARGIN);
        let mut strength_at = [Gi::ZERO; RUNGS];
        let mut pull_at = [Gi::ZERO; RUNGS];
        let mut lip_at = [Gi::ZERO; RUNGS];
        let mut rung = 0usize;
        while rung < RUNGS {
            let word = (strength * fade_at[rung] * noise_one).floor();
            strength_at[rung] = Gi::new(word.to_i64_floor());
            let share = word / noise_one;
            // ★ THE SLACK GOES IN TWICE. A VALUE bound needs it once — one column. It carries it
            // twice so the COLUMN bound, which reads two columns and therefore carries it twice,
            // still nests inside this one at a coarse rung where every one of its terms stands at
            // its ceiling and the two sums are otherwise equal.
            pull_at[rung] =
                Gi::new(rounded_up(share * pull_unit_m * bound_unit)) + (terrace_round << 1);
            lip_at[rung] = Gi::new(rounded_up((Gf::ONE + lip_slope * share) * noise_one));
            rung += 1;
        }
        let terrace = Terrace {
            datum: datum_steps << LENGTH_BITS,
            spacing: Gi::new(spacing_steps),
            spacing_recip: Gi::new(recip_pow2(spacing_steps as u64, TERRACE_RECIP_BITS) as i64),
            half_recip: Gi::new(recip_pow2(half_spacing_steps as u64, TERRACE_RECIP_BITS) as i64),
            strength: strength_at[0],
            pad: [Gi::ZERO; 3],
        };

        // ★ THE BAND IS SIZED ON THE SUM OF DRAWN AMPLITUDES (slice 8a, stage 1, §2.6), not on
        // `relief_m`. Reading `relief_m` was correct only while the amplitudes were NORMALISED to it.
        // Under the spectrum the fine octaves author their own amplitudes, so `Σ|a|` stands over the
        // relief — on the home planet by 393 m, MEASURED — and a band sized on the relief would leave the
        // surface outside the grid, which is a chunk with no ground in it. This is the number
        // `relief_bound(0)` already computes, in whole metres.
        //
        // The band is taken at the roughness factor's CEILING (`m = 1`), which is what lets 8c swap the
        // per-column factor for its macro field without moving one address (§1.2).
        //
        // ★ AND ON THE BENCH'S OWN REACH (slice 8a, stage 4). The terrace carries a surface at most
        // `strength × (S/2) × TERRACE_PULL_SHARE` from where the octaves put it — 10.6 m on the home
        // planet — and that is ADDED, never multiplied: the Lipschitz constant bounds a DIFFERENCE
        // between two surfaces, and the band bounds a VALUE. Multiplying `Σ|a|` by 1.44 would buy
        // 7 400 m of empty grid to hold 10.6 m of bench.
        //
        // The relief on both sides, the strata and the caves below, and room to stand on the highest
        // peak above. Whole metres, so the ladder floors them to whole cells at every rung.
        let pull_whole_m = Gf::from_i64(pull_at[0].raw()) / bound_unit;
        let relief_whole = (amplitude_sum_m + pull_whole_m).floor().to_i64_floor() as u32 + 1;
        let crust_m = relief_whole + strata.max_depth_m() + caves.max_depth_m + 64;
        let above_m = relief_whole + 64;
        let ladder = Ladder::for_radius(look_radius_m, crust_m, above_m)?;

        let mut inv_n = [Gi::ZERO; RUNGS];
        let mut rung = 0usize;
        while rung < usize::from(ladder.rungs) {
            inv_n[rung] = inv_n_of(ladder.cells_per_edge(rung as u8));
            rung += 1;
        }
        Some(BodyDefinition {
            seed,
            facts,
            ladder,
            radius,
            radius_steps,
            radius_recip: Gi::new(recip_pow2(radius_steps.raw() as u64, RADIUS_RECIP_BITS) as i64),
            relief: length_of(relief_m),
            sea_radius,
            inv_n,
            octaves,
            octave_count: count as u8,
            live_at,
            alias_at: rung_octaves.alias,
            cells_per_wave_at: rung_octaves.cells_per_wave,
            roughness,
            terrace,
            terrace_round,
            strength_at,
            pull_at,
            lip_at,
            strata,
            caves,
            biome,
        })
    }

    /// ★ THE BIOME CHARTER — the biome field's own numbers as the recipe's kernel reads them. One
    /// source: `crate::height::biome_of` hands this row to `vd_recipe::height::biome_of`, and so
    /// does the card's column pass, so the shard and the picture name one biome per column.
    #[must_use]
    pub fn biome_charter(&self) -> BiomeCharter {
        BiomeCharter {
            sea_radius: self.sea_radius,
            highland_above: self.biome.highland_above,
            highland_recip: self.biome.highland_recip,
            highland_shift: Gi::new(i64::from(RADIUS_RECIP_BITS - NOISE_BITS)),
            temperature: self.biome.temperature,
            humidity: self.biome.humidity,
        }
    }

    /// ★ THE PLAN CHARTER at a rung, for a chunk of `key_face` — everything the COLUMN PASS and the
    /// NODE PASS read that is not a column's or a node's own address (`vd_recipe::plan`). The host
    /// draws it once per box and the card runs both passes from it, so a box's request carries its
    /// key and this row and nothing else.
    ///
    /// **Example.** The home planet at rung 0 for a chunk on face `+X`: fourteen live octaves, the
    /// cell-count reciprocal of that rung, the cavern wavelength's reciprocal, and `+X`'s own index
    /// — which is the basis a corner phantom of that chunk stands on.
    #[must_use]
    pub fn plan_charter(&self, rung: u8, key_face: Face) -> PlanCharter {
        PlanCharter {
            seed: self.seed,
            cavern_recip: self.caves.cavern_recip,
            // The point is 128 times the metres, so the shift takes those seven bits back out along
            // with the reciprocal's own — the same shift `crate::carve::cavern_value` states.
            cavern_shift: Gi::new(i64::from(
                CAVERN_RECIP_BITS - NOISE_BITS + crate::units::STEPS_PER_M.trailing_zeros(),
            )),
            inv_n: self.inv_n(rung),
            radius: self.radius,
            octave_count: Gi::new(self.octaves_at(rung).len() as i64),
            key_face: Gi::new(i64::from(key_face.index())),
            biome: self.biome_charter(),
            roughness: self.roughness,
            terrace: self.terrace_at(rung),
            octaves: self.octaves,
        }
    }

    /// ★ THE LIVE OCTAVES AT RUNG `rung` — the survival rule of ruling T7 rule 1, read from the row
    /// the draw already computed ([`live_octaves_at`]): an octave lives while its wavelength covers
    /// the table's own cells per wavelength, and a RIDGED octave while it covers half that many, so
    /// a crest lives one rung longer than the round wave beside it. A rung past the body's own reads
    /// the last row, which keeps ONE octave (8c is the fix for that one; see [`live_octaves_at`]).
    #[must_use]
    pub fn octaves_at(&self, rung: u8) -> &[Octave] {
        let keep = usize::from(self.live_at[usize::from(rung).min(RUNGS - 1)]);
        &self.octaves[..keep]
    }

    /// ★ WHETHER A RUNG DRAWS AN ALIASED OCTAVE — the owner's ruling T6 ask 3, as a fact the body
    /// STATES instead of a wrong it hides. True where the survival rule kept nothing and the rung
    /// was given the widest octave anyway; false everywhere the rule itself answered.
    ///
    /// **Example.** The home planet aliases at rungs 14 to 18 — the globe a pilot sees from orbit —
    /// and at no rung a walker or a hull in the atmosphere ever draws. **8c is the fix**
    /// (`live_octaves_at`).
    #[must_use]
    pub fn aliases_at(&self, rung: u8) -> bool {
        self.alias_at[usize::from(rung).min(RUNGS - 1)]
    }

    /// ★ THE RATIO, PRINTED: how many cells of rung `rung` one wavelength of its FINEST LIVE octave
    /// covers. The table asks for `survival_cells_per_wave` of them — 48.83 on the home planet — and
    /// a crest is content with half. The top rungs stand far under both until 8c lands: 3.05 cells
    /// at rung 17, and 1.53 at rung 18, which is under Nyquist ([`NYQUIST_CELLS_PER_WAVE`]).
    #[must_use]
    pub fn cells_per_wave(&self, rung: u8) -> f64 {
        crate::units::share_of_q28(self.cells_per_wave_at[usize::from(rung).min(RUNGS - 1)])
    }

    /// ★ THE BENCH AT ONE RUNG: the body's own bed stack with the RUNG's own strength in it — zero
    /// where the rung's cells are too coarse to carry a tread, so the kernel answers its argument
    /// unchanged and pays one multiply by zero for saying so.
    #[must_use]
    pub fn terrace_at(&self, rung: u8) -> Terrace {
        Terrace {
            strength: self.strength_at[usize::from(rung).min(RUNGS - 1)],
            ..self.terrace
        }
    }

    /// ★ THE TERRACE'S OWN VALUE BOUND at a rung, in gap steps at [`LENGTH_BITS`]: the furthest the
    /// bench can carry a surface from where the octaves put it.
    #[must_use]
    pub fn pull_bound(&self, rung: u8) -> Gi {
        self.pull_at[usize::from(rung).min(RUNGS - 1)]
    }

    /// ★ THE INTEGER TERRACE'S OWN SLACK at one column, in gap steps at [`LENGTH_BITS`]: how far the
    /// kernel's whole-word falloff may stand from the real one ([`TERRACE_FADE_ROUND_UNITS`]). Every
    /// bound derived from the REAL terrace adds it — once for a VALUE, twice for a DIFFERENCE,
    /// because a difference reads two columns.
    #[must_use]
    pub fn terrace_slack(&self) -> Gi {
        self.terrace_round
    }

    /// ★ THE TERRACE'S LIPSCHITZ WORD at a rung, at the noise's bits: `1 + (7/9)·strength`. Every
    /// DIFFERENCE bound under the bench is multiplied by it, because the terrace amplifies every
    /// variation below it ([`TERRACE_LIP_NUM`]).
    #[must_use]
    pub fn lip(&self, rung: u8) -> Gi {
        self.lip_at[usize::from(rung).min(RUNGS - 1)]
    }

    /// THE SUM OF THE LIVE AMPLITUDES at a rung, in gap steps at [`LENGTH_BITS`]. An exact bound on
    /// the octave sum alone, because the noise is in `[−1, 1]` — the bench stands on top of it.
    fn amplitude_bound(&self, rung: u8) -> Gi {
        let mut sum = Gi::ZERO;
        for o in self.octaves_at(rung) {
            sum += o.amplitude;
        }
        sum << (LENGTH_BITS - AMP_BITS)
    }

    /// The most the surface can rise or fall from the ladder radius at rung `rung`, in gap steps at
    /// [`LENGTH_BITS`]: the live amplitudes PLUS the bench's own reach.
    ///
    /// ★ **THE BENCH IS ADDED, NOT MULTIPLIED** (slice 8a stage 4). This bounds a VALUE — how far
    /// the surface stands from the ladder radius — and `|T(h) − R| ≤ |h − R| + |pull|` is the
    /// honest line. The Lipschitz constant belongs to [`BodyDefinition::step_bound`], which bounds a
    /// DIFFERENCE. `slice_8a_design.md` §2.7 asks for a multiply on all three; on the home planet
    /// that would buy 7 400 m of empty grid to hold 10.6 m of bench.
    #[must_use]
    pub fn relief_bound(&self, rung: u8) -> Gi {
        self.amplitude_bound(rung) + self.pull_bound(rung)
    }

    /// ★ THE STEP ONE HANDOVER MAKES, in gap steps at [`LENGTH_BITS`]: the most the surface at rung
    /// `rung` can differ from the surface at the rung above it. Ruling T7's rules 2 and 3 read this
    /// ONE number, and so does [`BodyDefinition::dropped_bound`].
    ///
    /// **The derivation.** `T_L(h_L) − T_{L+1}(h_{L+1})` splits into `T_L(h_L) − T_L(h_{L+1})`,
    /// which the terrace's Lipschitz constant bounds by `Lip(L) · |h_L − h_{L+1}|`, and
    /// `T_L(h_{L+1}) − T_{L+1}(h_{L+1})`, which is the two rungs' benches at ONE height and stands
    /// under the difference of their two pull bounds. The first factor is the dropped amplitudes;
    /// the second is the fade's own step and never the whole bench.
    #[must_use]
    pub fn step_bound(&self, rung: u8) -> Gi {
        let next = rung.saturating_add(1);
        let dropped = self.amplitude_bound(rung) - self.amplitude_bound(next);
        // The two pull words carry the slack each, and subtracting them takes it away; a DIFFERENCE
        // reads two columns, so it goes back twice over.
        dropped.mul_shr(self.lip(rung), NOISE_BITS)
            + (self.pull_bound(rung) - self.pull_bound(next))
            + (self.terrace_round << 1)
            + Gi::new(TERRACE_BOUND_MARGIN)
    }

    /// The most the surface at rung `rung` can differ from the surface at rung 0, in gap steps at
    /// [`LENGTH_BITS`]: the handover steps between them, added up. Before the bench that was the
    /// plain difference of two amplitude sums; the terrace amplifies each step by its own rung's
    /// constant, so the chain is walked instead of subtracted.
    #[must_use]
    pub fn dropped_bound(&self, rung: u8) -> Gi {
        let mut sum = Gi::ZERO;
        let mut r = 0u8;
        while r < rung {
            sum += self.step_bound(r);
            r += 1;
        }
        sum
    }

    /// ★ THE ROUGHNESS FACTOR's own words, for this crate's height field and for an instrument that
    /// measures the planet's histogram of it (the slope histogram M-C).
    #[must_use]
    pub const fn roughness(&self) -> &Roughness {
        &self.roughness
    }

    /// The relief bound in metres, for a host outside the recipe.
    #[must_use]
    pub fn relief_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.relief_bound(rung))
    }

    /// The dropped-octave bound in metres, for a host outside the recipe (the client's crossfade sink
    /// reads it).
    #[must_use]
    pub fn dropped_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.dropped_bound(rung))
    }

    /// ★ THE STEP ONE HANDOVER MAKES, in metres — what ruling T7's rules 2 and 3 size a crossfade
    /// band and a switch distance from (`vd_client::ladder_view::handover_step_m`). The body states
    /// it; the client never re-derives it from two bounds.
    #[must_use]
    pub fn step_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.step_bound(rung))
    }

    /// The bench's own reach at a rung, in metres, for a host outside the recipe.
    #[must_use]
    pub fn pull_bound_m(&self, rung: u8) -> f64 {
        crate::units::metres_of_q28(self.pull_bound(rung))
    }

    /// The bench's strength at a rung as a real number in `[0, 1)`, for a host outside the recipe —
    /// an instrument reads it to say which rungs carry a bench.
    #[must_use]
    pub fn terrace_strength(&self, rung: u8) -> f64 {
        crate::units::share_of_q28(self.strength_at[usize::from(rung).min(RUNGS - 1)])
    }

    /// The bench's bed spacing in metres, for a host outside the recipe.
    #[must_use]
    pub fn bed_spacing_m(&self) -> f64 {
        crate::units::metres_of_q28(self.terrace.spacing << LENGTH_BITS)
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;
    use crate::units::metres_of_q28;
    use vd_recipe::noise::NOISE_ONE;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    /// The frequency of one octave, reassembled from the charter's pair, at the noise's bits.
    fn frequency_q(o: &Octave) -> Gi {
        (o.frequency_int << NOISE_BITS) + o.frequency_frac
    }

    /// ★ A 400 km BODY, drawn with the facts a 400 km body at 3 000 kg/m³ has: COMPUTED by
    /// `g = (4/3)πGρR`, `g = 0.3355 m/s²`, so 335 mm/s². The world may hold no body this small on
    /// the ladder near the home system, so the shape arm needs a body stated here.
    const SMALL_BODY_FACTS: BodyFacts = BodyFacts::new(335, 3_000);
    const SMALL_BODY_RADIUS_M: f64 = 400_000.0;

    /// ★ THE RELIEF LAW ON THE HOME PLANET (the landform arc, slice 8b stage 3; ruling T6 ask 1).
    ///
    /// Four statements, each of which could fail. (1) The STRENGTH arm binds on a planet — the
    /// shape arm stands fifty times higher. (2) The cap is the number the law's own constants give:
    /// `σ_y / (ρ_c · g)` with the body's OWN stated gravity and density. (3) The drawn relief is a
    /// share of that cap inside `[0.5, 1.0)`, so a bound is a bound. (4) The band the ladder is
    /// taken with follows the relief down, and the floor and the radial index follow the band.
    ///
    /// RED before the stage: the relief was `min(0.004·R, [200, 12 000]) × (0.5 + u)` — a lottery
    /// that read no fact of the body and could stand half again over its own cap.
    #[test]
    fn the_relief_law_caps_the_home_planet_by_its_crusts_own_strength() {
        let m = home();
        assert_eq!(
            m.facts(),
            BodyFacts::new(9_818, 5_513),
            "the home planet's own stated facts"
        );
        let (strength, shape) = m.relief_arms_m();
        // The cap is the lesser arm through the fence's own `lesser` — the same operation the draw
        // takes — so the print names no branch a passing test never walks (HR5).
        let cap = Gf::from_f64(strength).lesser(Gf::from_f64(shape)).to_f64();
        println!(
            "[relief_law] home: strength {strength:.3} m, shape {shape:.3} m, \
             cap {cap:.3} m, relief {:.3} m, band {} m, floor {} m",
            m.relief_m(),
            m.ladder.band_m,
            m.ladder.floor_m,
        );
        // (1) A PLANET IS STRENGTH-LIMITED. The two arms cross at about 939 km of radius, and this
        // body is six thousand kilometres.
        assert!(
            strength < shape,
            "the strength arm binds on a planet: {strength} against {shape}"
        );
        // (2) THE CAP, to the millimetre, from the law's own constants and the body's own facts.
        assert_eq!(
            (strength * 1_000.0).round() as i64,
            8_845_707,
            "the home planet's relief cap in millimetres"
        );
        // (3) THE DRAW IS INSIDE THE BOUND.
        assert_eq!(
            (m.relief_m() * 1_000.0).round() as i64,
            8_275_877,
            "the home planet's drawn relief in millimetres"
        );
        assert!(m.relief_m() >= 0.5 * strength, "the draw's floor");
        assert!(m.relief_m() < strength, "the draw never reaches its cap");
        // (4) THE BAND FOLLOWS. `Σ|a|` is the sum the band is sized on, and it tracks the relief
        // because the coarse octaves carry it.
        assert_eq!(
            (m.relief_bound_m(0) * 1_000.0).round() as i64,
            9_245_598,
            "the value bound the band is sized on (Σ|a| plus the bench's own pull), in millimetres"
        );
        assert_eq!(
            (m.ladder.band_m, m.ladder.floor_m),
            (271_454, 6_079_526),
            "the ladder's band and floor"
        );
        assert_eq!(m.ladder.rungs, 19, "the ladder's rungs do not move");
        assert_eq!(m.ladder.n, 9_961_472, "nor its edge count");
    }

    /// ★ THE SHAPE ARM, ON A BODY SMALL ENOUGH TO NEED IT (slice 8b stage 3; `06` §3.4).
    ///
    /// A strength-only law DELETES every small round body from the world: at 400 km and
    /// 3 000 kg/m³ the strength arm asks for 476 km of relief on a body of 400 km radius, and
    /// `Ladder::for_radius` refuses a crust that reaches the centre. The shape arm — `0.077·R`,
    /// cited to Vesta's own observed relief — is what keeps the moon in the world.
    #[test]
    fn a_four_hundred_kilometre_body_takes_the_shape_bound_arm() {
        let moon = BodyDefinition::from_seed(11, SMALL_BODY_RADIUS_M, SMALL_BODY_FACTS)
            .expect("a 400 km body is on the ladder");
        let (strength, shape) = moon.relief_arms_m();
        println!(
            "[relief_law] a 400 km body: strength {strength:.1} m, shape {shape:.1} m, \
             relief {:.1} m, radius {:.1} m",
            moon.relief_m(),
            moon.radius_m(),
        );
        assert!(
            shape < strength,
            "a small body is shape-limited: {shape} against {strength}"
        );
        // The shape arm is exactly the share of the body's OWN ladder radius.
        assert!(
            (shape - SHAPE_RELIEF_SHARE * moon.ladder.radius_m()).abs() < 1.0e-6,
            "the shape arm is 0.077 R: {shape}"
        );
        // The strength arm on this body asks for more relief than the body has radius — which is
        // the measurement that says why the shape arm must exist.
        assert!(
            strength > moon.radius_m(),
            "a strength-only law would ask for a crust past the centre: {strength}"
        );
        assert!(moon.relief_m() >= 0.5 * shape, "the draw's floor");
        assert!(moon.relief_m() < shape, "the draw never reaches its cap");
    }

    /// ★ A BODY WHOSE REALM STATES NO USABLE FACTS IS NOT DRAWN (slice 8b stage 3, §4.2 rule 2).
    ///
    /// The refusal stage 1 built on the client's lane — a surface with no charter is counted and
    /// gets no body — now stands AT THE DRAW as well: the charter is an argument and not an option,
    /// so a body with no charter cannot be built at all, and a charter carrying a zero where a
    /// physical fact belongs is refused rather than clamped. A zero gravity would make the strength
    /// arm a division by zero, and the fence calls that a defect, never a bound.
    #[test]
    fn a_body_whose_facts_carry_a_zero_is_refused_at_the_draw() {
        let r = f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS);
        assert_eq!(
            BodyDefinition::from_seed(crate::home::HOME_PLANET_SEED, r, BodyFacts::new(0, 5_513)),
            None,
            "a body with no weight has no strength bound"
        );
        assert_eq!(
            BodyDefinition::from_seed(crate::home::HOME_PLANET_SEED, r, BodyFacts::new(9_818, 0)),
            None,
            "a body of no substance has no crust to yield"
        );
        assert_eq!(
            BodyDefinition::from_seed(crate::home::HOME_PLANET_SEED, r, BodyFacts::new(0, 0)),
            None
        );
        // And the same seed and radius WITH the facts its realm states does draw a body.
        assert!(
            BodyDefinition::from_seed(crate::home::HOME_PLANET_SEED, r, crate::home::home_facts())
                .is_some()
        );
    }

    /// ★ THE LAW READS THE BODY'S OWN FACTS AND NOTHING ELSE (slice 8b stage 3). Two bodies of the
    /// same seed and the same radius, stating different gravities, get different mountains — which
    /// is what makes the charter an input to the shape and not a caption on it.
    #[test]
    fn two_bodies_of_one_seed_and_two_gravities_get_two_reliefs() {
        let r = f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS);
        let heavy = BodyDefinition::from_seed(
            crate::home::HOME_PLANET_SEED,
            r,
            BodyFacts::new(19_636, 5_513),
        )
        .expect("on the ladder");
        let light = crate::home::home_planet();
        // Twice the gravity, half the cap: the law's `1/g` trend, which is the physical part.
        assert!(
            (heavy.relief_arms_m().0 * 2.0 - light.relief_arms_m().0).abs() < 1.0e-6,
            "the cap falls as 1/g"
        );
        assert!(heavy.relief_m() < light.relief_m());
        assert_ne!(heavy, light, "the charter moves the body");
    }

    /// ★ THE SLOPE SPECTRUM, AND THE BAND THAT HOLDS ITS SUM (slice 8a, stage 1).
    ///
    /// Five statements, each of which could fail. (1) Every FINE octave's amplitude is its own slope
    /// times its own wavelength over a whole turn, within ONE gap step. (2) The fine slopes, as a root
    /// sum of squares, stand at the angle of repose. (3) `S_PEAK` lands on its pin, so a table whose
    /// ends move is a red test and not a surprise. (4) What the spectrum spends stays under the body's
    /// relief — and on a rock whose whole table is fine, the constraint BINDS. (5) The ladder's band
    /// holds `Σ|a|`, which on this body stands OVER the relief, which is why the band's one line moved.
    ///
    /// RED before the stage: the fine amplitudes were `k_rough^o · relief_m / Σ k^o` and the band read
    /// `relief_m`.
    #[test]
    fn the_fine_octaves_carry_the_slope_spectrum_and_the_band_holds_their_sum() {
        let m = home();
        let r = m.radius_m();
        let live = m.octaves_at(0);
        let count = live.len();
        let wavelength = |o: &Octave| r / octave_frequency(o);
        // (0) WHICH OCTAVES ARE FINE, read from the body's own charter and not from the draw.
        let mut first_fine = 0usize;
        for (i, o) in live.iter().enumerate() {
            if wavelength(o) >= FINE_ABOVE_M as f64 {
                first_fine = i + 1;
            }
        }
        assert_eq!(count, 14, "the home planet's table");
        assert_eq!(first_fine, 4, "the fine band starts at the 25 km octave");
        assert!(wavelength(&live[3]) >= FINE_ABOVE_M as f64);
        assert!(wavelength(&live[4]) < FINE_ABOVE_M as f64);

        // (3) `S_PEAK` AT ITS PIN: solved from the anchor over this body's own fine band.
        let s_peak = spectrum_peak(first_fine, count).to_f64();
        assert!((s_peak - 0.3998).abs() < 5e-5, "S_PEAK {s_peak}");

        // (1) EVERY FINE AMPLITUDE IS ITS SLOPE TIMES ITS WAVELENGTH over a whole turn, within one
        // gap step at `AMP_BITS` (1/32 768 m), which is THE ONE ROUNDING the charter carries.
        let one_gap_step_m = 1.0 / ((STEPS_PER_M << AMP_BITS) as f64);
        let mut fine_sum_m = 0.0;
        let mut slope_squares = 0.0;
        for (i, o) in live.iter().enumerate().skip(first_fine) {
            let lambda = wavelength(o);
            let want = s_peak * spectrum_base(i).to_f64() * lambda / std::f64::consts::TAU;
            let got = octave_amplitude_m(o);
            assert!(
                (got - want).abs() <= one_gap_step_m,
                "octave {i}: {got} against {want}"
            );
            let slope = got * std::f64::consts::TAU / lambda;
            slope_squares += slope * slope;
            fine_sum_m += got;
        }
        // (2) THE ANCHOR: the fine RMS slope is the tangent of the angle of repose.
        let fine_rms = slope_squares.sqrt();
        assert!(
            (fine_rms - TALUS_RMS * TAN_REPOSE).abs() < 1e-4,
            "the fine RMS slope {fine_rms}"
        );
        assert!(
            (fine_sum_m - 1_532.0).abs() < 1.0,
            "the fine octaves sum {fine_sum_m} m"
        );

        // (4) THE CONSTRAINT IS SLACK HERE. `relief_m` is not stored, so it is RECOVERED from the
        // coarse octaves, which still carry the geometric ladder: `a(o) = k^o · relief / Σ k^o`, so
        // `k = a(1)/a(0)` and `relief = a(0) · Σ k^o`.
        let k = octave_amplitude_m(&live[1]) / octave_amplitude_m(&live[0]);
        let mut weight_sum = 0.0;
        let mut weight = 1.0;
        for _ in 0..count {
            weight_sum += weight;
            weight *= k;
        }
        let relief_m = octave_amplitude_m(&live[0]) * weight_sum;
        assert!(
            fine_sum_m <= relief_m,
            "the spectrum spends {fine_sum_m} m of {relief_m} m"
        );

        // (5) THE BAND HOLDS `Σ|a|`, WHICH STANDS OVER THE RELIEF — the whole reason the band's line
        // moved off `relief_m`.
        let whole_sum_m = m.relief_bound_m(0);
        assert!(
            whole_sum_m > relief_m,
            "the drawn amplitudes {whole_sum_m} m against the relief {relief_m} m"
        );
        let floor_m = f64::from(m.ladder.floor_m);
        let top_m = floor_m + f64::from(m.ladder.band_m);
        // The two numbers the band must cover, computed BEFORE the assertions: an expression written
        // inside an `assert!` message runs only on failure, which is a region no green test can cover.
        let highest_peak_m = r + whole_sum_m;
        let deepest_cave_m =
            r - whole_sum_m - f64::from(m.strata.max_depth_m()) - f64::from(m.caves.max_depth_m);
        assert!(
            top_m >= highest_peak_m,
            "the band's top {top_m} m against the highest peak {highest_peak_m} m"
        );
        assert!(
            floor_m <= deepest_cave_m,
            "the band's floor {floor_m} m against the deepest cave {deepest_cave_m} m"
        );

        // (4b) THE `lesser` ARM, DRIVEN: a rock whose longest wave is already under the fine
        // threshold has NO coarse octave, and its spectrum asks for more relief than the body has, so
        // every amplitude stands strictly under the unconstrained spectrum's.
        let rock = BodyDefinition::from_seed(5, 3_000.0, ROCK_3KM_FACTS).expect("a rock");
        let rock_r = rock.radius_m();
        let rock_live = rock.octaves_at(0);
        let rock_count = rock_live.len();
        assert!(
            rock_r / octave_frequency(&rock_live[0]) < FINE_ABOVE_M as f64,
            "the rock's whole table is fine"
        );
        let rock_peak = spectrum_peak(0, rock_count).to_f64();
        for (i, o) in rock_live.iter().enumerate() {
            let lambda = rock_r / octave_frequency(o);
            let want = rock_peak * spectrum_base(i).to_f64() * lambda / std::f64::consts::TAU;
            let got = octave_amplitude_m(o);
            assert!(
                got < want,
                "the rock's octave {i}: {got} is not under {want}"
            );
        }
    }

    /// The direction of a face position, through the recipe's own bend: the cell nearest `(a, b)`
    /// of a face at rung 0 (a test names positions, the recipe names cells).
    fn dir(face: Face, a: f64, b: f64) -> [Gi; 3] {
        let m = home();
        let n_l = m.ladder().cells_per_edge(0);
        let i = vd_seed::ladder::index_of(a, n_l);
        let j = vd_seed::ladder::index_of(b, n_l);
        vd_seed::bend::direction_q(face, i, j, m.inv_n(0))
    }

    /// ★ A PLAIN COLUMN AND A RANGE COLUMN DIFFER BY THE FACTOR ALONE (slice 8a stage 3;
    /// `slice_8a_design.md` §1.2, §2.3 and §6 stage 3).
    ///
    /// Four statements, each of which could fail. (1) The factor stands inside `[M_MIN, 1]` over a
    /// scan of the home planet's own columns, and the scan MEETS a plain and a range — a factor that
    /// modulated nothing would meet one hump only. (2) On EVERY column the shaped field is the raw
    /// field with the FINE half multiplied once: `shaped − raw = fine·m − fine`, word for word, so
    /// the factor touches the fine octaves and nothing else. (3) The COARSE half is untouched: a
    /// column's coarse sum is the same word whatever its factor. (4) The CONTRAST is real: the
    /// lowest-factor column of the scan keeps under a fifth of the fine relief the highest-factor
    /// column keeps, on the same table.
    ///
    /// RED before the stage: `roughness_at` and `relief_shaped` did not exist, and `height` read the
    /// raw sum, so (2) held trivially with `m = 1` and (1) and (4) could not be stated at all.
    #[test]
    fn a_plain_column_and_a_range_column_differ_by_the_factor_alone() {
        use vd_recipe::height::{relief, relief_shaped, roughness_factor};
        let m = home();
        let live = m.octaves_at(0);
        let first_fine = m.roughness.first_fine.raw() as usize;
        assert_eq!(first_fine, 4, "the fine band starts at the 25 km octave");
        let floor = m.roughness.m_min;
        assert_eq!(floor, Gi::new(16_106_127), "round(0.06 · 2^28)");
        // ★ THE MASK AND THE INDEX ARE ONE SPLIT. The KERNEL reads each octave's own `fine` mask,
        // never an index compared inside its loop (the card measured that comparison wrong); the
        // roughness row's `first_fine` is the DRAW's record of the same split. A table whose masks
        // disagreed with it would give a surface the host's own accounting could not reproduce.
        for (i, o) in m.octaves[..usize::from(m.octave_count)].iter().enumerate() {
            let want = if i < first_fine {
                OCTAVE_COARSE
            } else {
                OCTAVE_FINE
            };
            assert_eq!(o.fine, want, "the mask of octave {i}");
        }
        let (mut plains, mut ranges) = (0, 0);
        let (mut lowest, mut highest) = (NOISE_ONE, Gi::ZERO);
        let (mut low_dir, mut high_dir) = (None, None);
        let mut i = 0u32;
        while i < 600 {
            let a = -1.0 + f64::from(i % 25) / 12.5;
            let b = -1.0 + f64::from((i / 25) % 24) / 12.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let factor = roughness_factor(&m.roughness, d);
            // (1) THE BAND.
            assert!(factor >= floor, "under the floor at {i}: {factor:?}");
            assert!(factor <= NOISE_ONE, "over the ceiling at {i}: {factor:?}");
            plains += i32::from(factor <= NOISE_ONE >> 2);
            ranges += i32::from(factor >= (NOISE_ONE >> 2) * Gi::new(3));
            if factor < lowest {
                lowest = factor;
                low_dir = Some(d);
            }
            if factor > highest {
                highest = factor;
                high_dir = Some(d);
            }
            // (2) THE FINE HALF ALONE CARRIES IT.
            let raw = relief(live, d);
            let shaped = relief_shaped(live, d, m.roughness());
            let fine = relief(&live[first_fine..], d);
            assert_eq!(
                shaped - raw,
                fine.mul_shr(factor, NOISE_BITS) - fine,
                "the factor moved something other than the fine half at {i}"
            );
            // (3) THE COARSE HALF IS UNTOUCHED.
            assert_eq!(
                shaped - fine.mul_shr(factor, NOISE_BITS),
                relief(&live[..first_fine], d),
                "the coarse sum moved at {i}"
            );
            i += 1;
        }
        assert!(plains > 0, "the scan meets a plain: {plains}");
        assert!(ranges > 0, "and a range: {ranges}");
        // (4) THE CONTRAST, on the two columns the scan itself named.
        let low = low_dir.expect("the scan named a plain");
        let high = high_dir.expect("the scan named a range");
        let low_fine = metres_of_q28(relief(&live[first_fine..], low).mul_shr(lowest, NOISE_BITS));
        let high_fine =
            metres_of_q28(relief(&live[first_fine..], high).mul_shr(highest, NOISE_BITS));
        assert!(
            lowest.raw() * 5 < highest.raw(),
            "the plain's factor {lowest:?} is not a fifth of the range's {highest:?}"
        );
        // Two statements and no branch of the test's own: the range column keeps real fine relief,
        // and the plain column keeps less of it. (`f64::max` is banned — SL10 clause 4 — and a
        // conditional here would leave an arm no green run walks.)
        assert!(
            high_fine.abs() > 1.0,
            "the range keeps only {high_fine} m of fine relief"
        );
        assert!(
            low_fine.abs() < high_fine.abs(),
            "the plain keeps {low_fine} m of fine relief against the range's {high_fine} m"
        );
    }

    /// ★ THE BAND IS SIZED AT THE FACTOR'S CEILING (slice 8a stage 3; `slice_8a_design.md` §1.2).
    ///
    /// This is the property that lets 8c swap the placeholder for the macro field WITHOUT MOVING ONE
    /// ADDRESS: the ladder's `floor_m` and `band_m` are derived from `Σ|a|` at `m = 1`, the upper
    /// bound BOTH fields obey, so a field that answers differently column by column still fits the
    /// grid the body already has.
    ///
    /// Three statements. (1) `Σ|a|` reads the AMPLITUDES and nothing else, so the factor does not
    /// move it: it stands at the same 16 846.70 m stage 1 recorded, and the value bound is that sum
    /// PLUS the cap-rock bench's own reach (slice 8a stage 4). (2) The
    /// ladder's crust and room hold that sum. (3) Over a scan the shaped surface stands strictly
    /// INSIDE the bound, and it stands where the factor puts it — under what the raw sum would give
    /// wherever the factor is under one, never over it, because the factor is DOWNWARD ONLY.
    ///
    /// RED before the stage: the third statement cannot be written without the factor.
    #[test]
    fn the_band_is_sized_at_the_factors_ceiling() {
        use vd_recipe::height::{relief, relief_shaped};
        let m = home();
        // (1) `Σ|a|` DOES NOT MOVE: the factor is not in it, and neither is the bench.
        let mut amplitudes_m = 0.0;
        for o in m.octaves_at(0) {
            amplitudes_m += octave_amplitude_m(o);
        }
        assert!(
            (amplitudes_m - 9_235.04).abs() < 0.01,
            "Σ|a| {amplitudes_m} m — the roughness factor may not move it"
        );
        let sum_m = m.relief_bound_m(0);
        assert!(
            (sum_m - amplitudes_m - m.pull_bound_m(0)).abs() < 1e-6,
            "the value bound is `Σ|a|` plus the bench's reach: {sum_m} m"
        );
        // (2) THE LADDER HOLDS IT.
        let r = m.radius_m();
        let floor_m = f64::from(m.ladder.floor_m);
        let top_m = floor_m + f64::from(m.ladder.band_m);
        assert!(top_m >= r + sum_m, "the band's top {top_m} m");
        assert!(floor_m <= r - sum_m, "the band's floor {floor_m} m");
        // (3) THE SURFACE STANDS INSIDE THE CEILING, AND THE FACTOR ONLY EVER LOWERS THE FINE HALF.
        let live = m.octaves_at(0);
        let first_fine = m.roughness.first_fine.raw() as usize;
        let mut lowered = 0;
        let mut raised = 0;
        let mut i = 0u32;
        while i < 400 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 10.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let shaped = relief_shaped(live, d, m.roughness());
            assert!(
                Gi::new(shaped.unsigned_abs() as i64) <= m.relief_bound(0),
                "the shaped surface left the ceiling at {i}"
            );
            // The fine half's MAGNITUDE never grows: `|fine · m| ≤ |fine|`, which is what "downward
            // only" means and what the band is sized on.
            let fine = relief(&live[first_fine..], d);
            let scaled = shaped - relief(&live[..first_fine], d);
            assert!(
                scaled.unsigned_abs() <= fine.unsigned_abs(),
                "the factor raised the fine half at {i}"
            );
            lowered += i32::from(scaled.unsigned_abs() < fine.unsigned_abs());
            raised += i32::from(scaled.unsigned_abs() > fine.unsigned_abs());
            i += 1;
        }
        assert!(lowered > 0, "the factor lowers some column: {lowered}");
        assert_eq!(raised, 0, "and raises none");
    }

    /// The raw surface of a column — the octave sum WITHOUT the bench — so a test can measure what
    /// the terrace moved.
    fn raw_height(m: &BodyDefinition, d: [Gi; 3], rung: u8) -> Gi {
        m.radius + vd_recipe::height::relief_shaped(m.octaves_at(rung), d, m.roughness())
    }

    /// ★ FAILING FIRST (slice 8a stage 4): THE CAP-ROCK BENCH PULLS THE GROUND TOWARD A BED TOP AT A
    /// FIXED RADIUS, AND NEVER PAST IT.
    ///
    /// Six statements, each of which could fail. (1) The bed spacing is TWICE the body's own soft
    /// bed, derived and never drawn. (2) The datum stands under the deepest surface the body can
    /// reach, which is what makes the kernel's bed index a floor. (3) Over 600 real columns the
    /// pull never passes the body's own pull bound, and it REACHES a real share of it. (4) Every
    /// terraced column stands NEARER its own bed top than the raw column did — never farther, and
    /// strictly nearer on most. (5) The bench is a function of the RADIUS alone: two columns on
    /// different faces whose raw surfaces stand within a gap step of each other are moved by the
    /// same amount, which is what "one height along a whole hillside" means and what a depth-draped
    /// stratum can never do. (6) The pull is zero exactly at a bed top.
    ///
    /// RED before the stage: `height` was the octave sum and nothing else, so the pull was zero on
    /// every column and statement (3)'s "reaches a share of the bound" and (4)'s "strictly nearer on
    /// most" both refuse.
    #[test]
    fn the_bench_pulls_the_ground_toward_a_bed_top_at_a_fixed_radius() {
        let m = home();
        // (1) THE SPACING IS THE BODY'S OWN BED, TWICE.
        assert_eq!(
            m.bed_spacing_m(),
            (BEDS_PER_SPACING * u64::from(m.strata.sediment_m)) as f64,
            "the spacing is two of the body's own sediment"
        );
        assert_eq!(m.bed_spacing_m(), 138.0, "the home planet's beds");
        // (2) THE DATUM STANDS UNDER THE DEEPEST RAW SURFACE — which is the radius less `Σ|a|`,
        // because the bed index reads the height the octaves give and never the terraced one.
        let datum_m = metres_of_q28(m.terrace.datum);
        let mut amplitude_sum_m = 0.0;
        for o in m.octaves_at(0) {
            amplitude_sum_m += octave_amplitude_m(o);
        }
        assert!(
            datum_m <= m.radius_m() - amplitude_sum_m,
            "the datum {datum_m} over the deepest surface"
        );
        let spacing_m = m.bed_spacing_m();
        let pull_bound = m.pull_bound_m(0);
        assert!(pull_bound > 0.0, "the bench has a reach: {pull_bound}");
        let mut worst_pull = 0.0f64;
        let mut nearer = 0;
        let mut farther = 0;
        let mut same_height: Vec<(f64, f64)> = Vec::new();
        let mut i = 0u32;
        while i < 600 {
            let a = -1.0 + f64::from(i % 25) / 12.5;
            let b = -1.0 + f64::from(i / 25) / 12.5;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let raw_q = raw_height(&m, d, 0);
            assert!(
                !(raw_q - m.terrace.datum).is_negative(),
                "column {i} stands under the datum"
            );
            let raw = metres_of_q28(raw_q);
            let got = metres_of_q28(crate::height::height(&m, d, 0));
            // (3) THE PULL STANDS INSIDE THE BODY'S OWN BOUND.
            let pull = (got - raw).abs();
            assert!(pull <= pull_bound, "column {i}: pulled {pull} m");
            if pull > worst_pull {
                worst_pull = pull;
            }
            // (4) NEARER ITS BED TOP, NEVER FARTHER.
            let gap = |h: f64| {
                let x = (h - datum_m) / spacing_m;
                ((x - x.floor()) - (x - x.floor()).round()).abs() * spacing_m
            };
            let (before, after) = (gap(raw), gap(got));
            assert!(after <= before + 1e-6, "column {i}: {before} -> {after}");
            nearer += i32::from(after < before - 1e-6);
            farther += i32::from(after > before + 1e-6);
            same_height.push((raw, got - raw));
            i += 1;
        }
        assert!(
            worst_pull > pull_bound * 0.5,
            "the pull reaches its bound: {worst_pull} of {pull_bound}"
        );
        assert!(nearer > 500, "columns pulled toward a bed top: {nearer}");
        assert_eq!(farther, 0, "and none pushed away");
        // (5) THE BENCH READS THE RADIUS AND NOTHING ELSE: two columns of the same raw height are
        // moved by the same amount, whatever face they stand on.
        same_height.sort_by(|x, y| x.0.partial_cmp(&y.0).expect("finite"));
        let mut pairs = 0;
        let mut k = 1;
        while k < same_height.len() {
            if (same_height[k].0 - same_height[k - 1].0).abs() < 0.01 {
                assert!(
                    (same_height[k].1 - same_height[k - 1].1).abs() < 0.01,
                    "two columns of one height moved apart"
                );
                pairs += 1;
            }
            k += 1;
        }
        assert!(
            pairs > 0,
            "the scan meets two columns of one height: {pairs}"
        );
        // (6) AT A BED TOP the pull is nothing.
        let top = crate::units::q28_of_metres(datum_m + spacing_m * 40.0);
        assert_eq!(
            vd_recipe::terrace::terrace(&m.terrace_at(0), top),
            top,
            "a column standing on a bed top"
        );
    }

    /// ★ FAILING FIRST (slice 8a stage 4): THE BENCH AMPLIFIES BY THE BODY'S OWN LIPSCHITZ WORD, and
    /// that word is the QUINTIC's `1 + (7/9)·q` and NOT `04_detail_rungs.md` §4.5's cubic
    /// `1 + 0.6875·q`. A DENSE SCAN of the terrace's own first difference over four whole beds of
    /// the home planet, against both numbers.
    ///
    /// RED if the bound is copied from `04`: the scan measures 1.4358 where the cubic promises
    /// 1.3852, so a bound taken from `04` stands 3.7 % SHORT — and a short bound on the column span
    /// is a chunk missed, which is a hole.
    #[test]
    fn the_bench_amplifies_by_the_bodys_own_lipschitz_word_and_not_the_cubics() {
        let m = home();
        let q = m.terrace_strength(0);
        let word = crate::units::share_of_q28(m.lip(0));
        let t = m.terrace_at(0);
        let base = m.radius_m();
        let step_m = 1.0 / (STEPS_PER_M as f64);
        let mut worst = 0.0f64;
        let mut i = 0i64;
        while i < 4 * 138 * STEPS_PER_M {
            let h = crate::units::q28_of_metres(base + (i as f64) * step_m);
            let next = crate::units::q28_of_metres(base + ((i + 1) as f64) * step_m);
            let a = metres_of_q28(vd_recipe::terrace::terrace(&t, h));
            let b = metres_of_q28(vd_recipe::terrace::terrace(&t, next));
            let slope = (b - a) / step_m;
            if slope > worst {
                worst = slope;
            }
            i += 1;
        }
        let cubic = 1.0 + 0.6875 * q;
        // The scan reads the INTEGER terrace, whose falloff is whole words; the body's own slack
        // says how far that can stand from the real shape at one column, and a secant over one gap
        // step reads twice it (`terrace_slack`, and the derivation at `TERRACE_FADE_ROUND_UNITS`).
        let slack = 2.0 * metres_of_q28(m.terrace_slack()) / step_m;
        assert!(
            worst <= word + slack,
            "the body's word {word} (+{slack}) under the scan {worst}"
        );
        assert!(
            worst > cubic,
            "the cubic's constant {cubic} is BROKEN by {worst}"
        );
        assert!(
            (worst - word).abs() < 0.01,
            "the word is tight: {word} against {worst}"
        );
        // And the word is what the COLUMN BOUND multiplies by, at every rung.
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            let lip = crate::units::share_of_q28(m.lip(rung));
            assert!(lip >= 1.0, "rung {rung}: {lip}");
            assert!(lip <= 1.0 + 7.0 / 9.0, "rung {rung}: {lip}");
            rung += 1;
        }
    }

    /// ★ FAILING FIRST (slice 8a stage 4): THE BENCH FADES OUT OVER ITS LAST RUNG, so it never
    /// appears or vanishes between two rungs. On the home planet its 138 m beds need 5.6 cells of a
    /// tread, so the bench stands at full strength to rung 3 (8 m cells), fades through rung 4
    /// (16 m) and is gone by rung 5 (32 m).
    ///
    /// The third statement is the one that matters to the picture: the step between the last rung
    /// that carries a bench and the first that does not is the FADE's own step, not the whole bench.
    #[test]
    fn the_bench_fades_out_over_its_last_rung() {
        let m = home();
        let full = m.terrace_strength(0);
        assert!(full > 0.0, "the bench exists: {full}");
        // Full strength while the tread covers its cells.
        let mut rung = 0u8;
        while rung <= 3 {
            assert_eq!(
                m.terrace_strength(rung),
                full,
                "rung {rung} at full strength"
            );
            rung += 1;
        }
        // One rung of fade, then nothing.
        let fading = m.terrace_strength(4);
        assert!(fading > 0.0, "rung 4 fades: {fading}");
        assert!(fading < full, "rung 4 fades: {fading}");
        let mut rung = 5u8;
        while rung < RUNGS as u8 {
            assert_eq!(
                m.terrace_strength(rung),
                0.0,
                "rung {rung} carries no bench"
            );
            rung += 1;
        }
        // The fade's own step, and not the whole bench.
        let whole = m.pull_bound_m(0);
        let fade_step = m.pull_bound_m(4) - m.pull_bound_m(5);
        assert!(fade_step > 0.0, "the fade makes a step: {fade_step}");
        assert!(
            fade_step < whole * 0.75,
            "the step is the fade's, not the bench's: {fade_step} of {whole}"
        );
        // And a rung that carries no bench answers its argument word for word.
        let h = crate::units::q28_of_metres(m.radius_m() + 321.0);
        assert_eq!(vd_recipe::terrace::terrace(&m.terrace_at(9), h), h);
        assert!(m.terrace_at(0).strength > m.terrace_at(4).strength);
    }

    /// ★ FAILING FIRST (slice 8a stage 4): THE BOUNDS HOLD THE BENCH'S WORST CASE, and the ladder's
    /// band holds it too.
    ///
    /// Four statements. (1) The ladder's crust and its room over the surface hold `Σ|a|` PLUS the
    /// bench's own reach — the band grew by 10.6 m and not by `Σ|a| × 1.44`, because a band bounds a
    /// VALUE. (2) Over 400 real columns at every rung the surface stands inside `relief_bound`. (3)
    /// Over the same columns every coarser rung stands inside `dropped_bound`, which is now the
    /// handover steps added up rather than a difference of two sums. (4) The column bound carries the
    /// terrace's Lipschitz word, so a chunk's span is never short.
    ///
    /// RED before the stage: the band read `Σ|a|` alone and `dropped_bound` was a plain subtraction,
    /// so the terraced surface stood outside both.
    #[test]
    fn the_band_and_the_bounds_hold_the_benchs_worst_case() {
        let m = home();
        let r = m.radius_m();
        // (1) THE BAND HOLDS Σ|a| + the bench's reach.
        let mut amplitude_sum_m = 0.0;
        for o in m.octaves_at(0) {
            amplitude_sum_m += octave_amplitude_m(o);
        }
        let need = amplitude_sum_m + m.pull_bound_m(0);
        let floor_m = f64::from(m.ladder.floor_m);
        let top_m = floor_m + f64::from(m.ladder.band_m);
        assert!(
            r - need >= floor_m,
            "the crust holds {need} m under the radius"
        );
        assert!(r + need <= top_m, "the room holds {need} m over the radius");
        let value_bound = m.relief_bound_m(0);
        assert!(
            value_bound > amplitude_sum_m,
            "the value bound carries the bench: {value_bound} over {amplitude_sum_m}"
        );
        // (2) and (3): every column, every rung.
        let mut i = 0u32;
        while i < 400 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 10.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h0 = crate::height::height(&m, d, 0);
            assert!(
                Gi::new((h0 - m.radius).unsigned_abs() as i64) <= m.relief_bound(0),
                "column {i} inside the band"
            );
            let mut rung = 1u8;
            while rung < m.ladder.rungs {
                let hl = crate::height::height(&m, d, rung);
                assert!(
                    Gi::new((hl - h0).unsigned_abs() as i64) <= m.dropped_bound(rung),
                    "column {i} rung {rung}: {hl:?} against {h0:?}"
                );
                assert!(
                    Gi::new((hl - m.radius).unsigned_abs() as i64) <= m.relief_bound(rung),
                    "column {i} rung {rung} inside the band"
                );
                rung += 1;
            }
            i += 1;
        }
        // (4) THE COLUMN BOUND carries the word.
        assert!(
            crate::digest::column_bound(&m, 0) > Gi::ZERO,
            "the column bound is real"
        );
        assert_eq!(m.dropped_bound(0), Gi::ZERO, "rung 0 drops nothing");
    }

    /// ★ FAILING FIRST (slice 8a stage 4): THE STRENGTH IS SOLVED FROM THE LADDER, NEVER TYPED.
    ///
    /// (1) It stands strictly under its ceiling `1 − M_MIN`, so the LADDER is what binds on this
    /// body and not the fold. (2) It stands inside the `[0.4, 0.7)` band `slice_8a_design.md` §2.4
    /// proposed as a DRAW — a cross-check that could have failed, since nothing in the solve reads
    /// that band. (3) The pair it binds on is the one the solve names: at the binding pair the step
    /// stands within the solve's own margin of a whole cell, and at no pair over it.
    #[test]
    fn the_benchs_strength_is_solved_from_the_ladder_and_stands_under_its_ceiling() {
        let m = home();
        let q = m.terrace_strength(0);
        assert!(q > 0.0, "the ladder can carry a bench: {q}");
        assert!(
            q < TERRACE_STRENGTH_CEILING,
            "under the fold's ceiling: {q}"
        );
        assert!((0.4..0.7).contains(&q), "inside `04`'s proposed band: {q}");
        let margin_m = TERRACE_MARGIN_STEPS as f64 / (STEPS_PER_M as f64);
        let mut binding = 0u8;
        let mut worst = 0.0f64;
        let mut rung = 0u8;
        while rung + 1 < m.ladder.rungs {
            let ratio = m.step_bound_m(rung) / f64::from(cell_m(rung + 1));
            if ratio > worst {
                worst = ratio;
                binding = rung;
            }
            rung += 1;
        }
        assert!(worst <= 1.0, "no pair over a cell: {worst} at {binding}");
        assert!(
            worst >= 1.0 - margin_m,
            "the solve is tight: {worst} at pair {binding}"
        );
        assert_eq!(binding, 3, "the bench's own fade rung binds");
    }

    #[test]
    fn a_body_is_drawn_once_from_its_seed_and_its_octaves_sum_to_its_relief() {
        let m = home();
        assert_eq!(m, home(), "the same seed, the same body");
        assert_ne!(
            m.octaves[0].seed,
            BodyDefinition::from_seed(
                m.seed + 1,
                f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS),
                crate::home::home_facts()
            )
            .expect("on the ladder")
            .octaves[0]
                .seed
        );
        assert!(
            m.octave_count >= 8,
            "a 20 km wave halves to 30 m in at least eight steps"
        );
        assert!(usize::from(m.octave_count) <= OCTAVES);
        let relief = m.relief_bound_m(0);
        assert!(relief >= 100.0, "{relief}");
        assert!(relief <= 18_000.0, "{relief}");
        // Amplitudes fall and frequencies double, coarsest first.
        let live = m.octaves_at(0);
        let mut o = 1;
        while o < live.len() {
            assert!(live[o].amplitude < live[o - 1].amplitude);
            assert!(frequency_q(&live[o]) > frequency_q(&live[o - 1]));
            o += 1;
        }
        assert!((m.sea_radius_m() - m.radius_m()).abs() <= relief);
        assert!(m.strata.max_depth_m() < 100);
        assert!(m.caves.min_depth_m < m.caves.max_depth_m);
        assert_eq!(m.ladder.rungs, 19, "the home planet's ladder");
        // The band holds the whole relief, the strata and the caves.
        let relief_whole = relief as u32 + 1;
        assert!(m.ladder.band_m >= 2 * relief_whole + m.strata.max_depth_m() + m.caves.max_depth_m);
        assert!(u64::from(m.ladder.floor_m) + u64::from(relief_whole) < m.radius_m() as u64);
        assert_eq!(
            BodyDefinition::from_seed(1, f64::NAN, crate::home::home_facts()),
            None,
            "the ladder refuses"
        );
        assert_eq!(
            BodyDefinition::from_seed(1, 100.0, crate::home::home_facts()),
            None,
            "a body smaller than its own crust"
        );
    }

    /// ★ THE CHARTER'S OWN GATE (ruling F7): every word the kernels read is the ROUNDED draw, and the
    /// two reciprocals are the exact floors of the divisions they stand for. MEASURED, not argued.
    #[test]
    fn the_integer_charter_is_the_rounded_draw_and_its_reciprocals_are_exact() {
        let m = home();
        // The radius: the word is the metres to within half a step's 2⁻²⁸, and the whole-step twin is
        // its floor.
        let radius_m = m.radius_m();
        assert!((radius_m - 6_341_670.0).abs() < 1.0, "{radius_m}");
        assert_eq!(m.radius_steps, m.radius >> LENGTH_BITS);
        // The two reciprocals, against the divisions they replace.
        let radius_steps = m.radius_steps.raw();
        assert_eq!(
            i128::from(m.radius_recip.raw()),
            (1i128 << RADIUS_RECIP_BITS) / i128::from(radius_steps)
        );
        assert_eq!(
            i128::from(m.caves.cavern_recip.raw()),
            (1i128 << CAVERN_RECIP_BITS) / i128::from(m.caves.cavern_wavelength_m)
        );
        let d = (m.biome.highland_above >> LENGTH_BITS).raw() + STEPS_PER_M;
        assert_eq!(
            i128::from(m.biome.highland_recip.raw()),
            (1i128 << RADIUS_RECIP_BITS) / i128::from(d)
        );
        // The tube region's edge is a power of two, so the region index is a shift.
        assert_eq!(m.caves.tube_region_m, TUBE_REGION_M);
        assert_eq!(
            m.caves.tube_region_steps,
            Gi::new(i64::from(TUBE_REGION_M) * STEPS_PER_M)
        );
        assert_eq!(
            Gi::ONE << m.caves.tube_region_shift,
            m.caves.tube_region_steps
        );
        // The threshold sits inside the noise's unit interval, and the scale is twenty metres.
        assert!(m.caves.cavern_threshold > Gi::ZERO);
        assert!(m.caves.cavern_threshold < NOISE_ONE);
        assert_eq!(m.caves.cavern_scale_steps, Gi::new(20 * STEPS_PER_M));
        // The two exits an investigation host reads: the frequency and the amplitude as real numbers.
        let coarse = &m.octaves_at(0)[0];
        let f = octave_frequency(coarse);
        assert!((f - (coarse.frequency_int.raw() as f64)).abs() < 1.0, "{f}");
        assert!(f > 1.0, "{f}");
        assert_eq!(
            octave_amplitude_m(coarse),
            metres_of_q28(Gi::new(coarse.amplitude.raw()) << (LENGTH_BITS - AMP_BITS))
        );
        // The biome's noises carry one noise unit of amplitude, so the octave sum reads the noise.
        assert_eq!(m.biome.temperature.amplitude, Gi::new(1 << AMP_BITS));
        assert_eq!(m.biome.humidity.amplitude, Gi::new(1 << AMP_BITS));
        assert_ne!(m.biome.temperature.seed, m.biome.humidity.seed);
        // Every rung of the body has its cell-count reciprocal; none past it does.
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            assert_eq!(
                m.inv_n(rung),
                inv_n_of(m.ladder.cells_per_edge(rung)),
                "rung {rung}"
            );
            rung += 1;
        }
        assert_eq!(m.inv_n(m.ladder.rungs), m.inv_n(m.ladder.rungs - 1));
        assert_eq!(m.inv_n[usize::from(m.ladder.rungs)], Gi::ZERO);
    }

    /// M-16 restated, part 1, as an EXACT property (the refuter's finding 22): the coarse answer at
    /// every rung sums strictly fewer octaves than the rung below it, down to one. The timing in the
    /// bench is a measurement; this is the gate.
    /// What a host outside the crate may read of a body: the seed, the ladder and the octave count,
    /// and nothing it could write (the refuter's finding 18).
    #[test]
    fn a_body_shows_its_seed_its_ladder_and_its_octave_count_and_nothing_writable() {
        let m = home();
        assert_eq!(m.seed(), crate::home::HOME_PLANET_SEED);
        assert_eq!(m.ladder(), &m.ladder);
        assert_eq!(m.octave_count(), m.octave_count);
        assert_eq!(usize::from(m.octave_count()), m.octaves_at(0).len());
        assert_eq!(m.radius_m(), metres_of_q28(m.radius));
        assert_eq!(m.sea_radius_m(), metres_of_q28(m.sea_radius));
    }

    #[test]
    fn every_rung_sums_no_more_octaves_than_the_rung_below_it() {
        let m = home();
        for o in m.octaves_at(0) {
            assert!(frequency_q(o) > Gi::ZERO);
            assert!(o.amplitude > Gi::ZERO);
        }
        // ★ THE LADDER IS LONGER THAN THE TABLE since 2026-09-15: the home planet has nineteen rungs
        // and fourteen octaves, so a rung drops octaves until ONE is left and then keeps that one.
        //
        // ★ AND A RUNG MAY REPEAT ITS NEIGHBOUR'S COUNT since ruling T7 rule 1: a RIDGED octave
        // lives one rung longer than its wavelength alone would buy, so the rung that would have
        // dropped it keeps it and the count stands still for exactly that one rung. The count never
        // RISES, and it stands still only where the octave at the table's live edge is a crest.
        let mut rung = 1u8;
        while rung < m.ladder.rungs {
            let below = m.octaves_at(rung - 1).len();
            let here = m.octaves_at(rung).len();
            assert!(here <= below, "rung {rung}: {here} vs {below}");
            if below > 1 {
                assert!(here + 2 >= below, "rung {rung}: {here} vs {below}");
                if here == below {
                    assert_eq!(
                        m.octaves[below - 1].kind,
                        OCTAVE_RIDGED,
                        "rung {rung} keeps {here}: only a crest buys the extra rung"
                    );
                }
            } else {
                assert_eq!(here, 1, "rung {rung}: the last octave stays");
            }
            rung += 1;
        }
        assert!(m.octaves_at(m.ladder.rungs - 1).len() < m.octaves_at(0).len());
    }

    /// ★ THE SAMPLING RULE PER OCTAVE KIND — ruling T7 rule 1, and the MEASUREMENT that made it.
    ///
    /// Four statements, each of which could fail.
    ///
    /// 1. **The rule is the count rule for a round wave.** An octave that is not a crest lives on
    ///    exactly the rungs the old rule gave it, so no byte of the ground below the crest band
    ///    moves. The 390.6 m octave (index 10, smooth) lives on rung 3 and is gone on rung 4.
    /// 2. **A crest lives one rung longer.** The 3 125 m octave (index 7, ridged) lived to rung 6
    ///    and now lives to rung 7.
    /// 3. **The table at rung 0 does not move at all** — the ground the walker stands on is the
    ///    same ground.
    /// 4. **The rule is the wavelength test, at every rung and every octave**, never a table of
    ///    counts: the answer holds an octave exactly while its own wavelength — doubled for a
    ///    crest — covers the table's own cells per wavelength of that rung's cell.
    ///
    /// ★ RED BEFORE THE RULE (the control: [`octave_survives`] reaching `wave_m` for every kind):
    /// statement 2 read "the crest is gone on rung 7", which is the 1.28 px the judge measured.
    #[test]
    fn a_crest_lives_one_rung_longer_than_the_round_wave_beside_it() {
        let m = home();
        let count = usize::from(m.octave_count);
        // 3. The walker's own ground: every octave the seed drew.
        assert_eq!(m.octaves_at(0).len(), count);
        // 1. A round wave keeps the count rule's own life.
        assert_eq!(m.octaves[10].kind, OCTAVE_SMOOTH);
        assert!(m.octaves_at(3).len() > 10, "the round wave lives on rung 3");
        assert!(m.octaves_at(4).len() <= 10, "and is gone on rung 4");
        // 2. A crest lives one rung longer than the count rule gave it.
        assert_eq!(m.octaves[7].kind, OCTAVE_RIDGED);
        assert!(m.octaves_at(7).len() > 7, "the crest lives on rung 7");
        assert!(m.octaves_at(8).len() <= 7, "and is gone on rung 8");
        // 4. The rule itself, at every rung of the ladder and every octave of the table, stated
        //    INDEPENDENTLY of the draw: the table halves its wavelength per index and the ladder
        //    halves its cell per rung, so "the wavelength covers the table's own cells" is exactly
        //    `index + rung ≤ count − 1`, and a crest's doubled wavelength buys it ONE more index.
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            let live = m.octaves_at(rung).len();
            let mut o = 0usize;
            while o < count {
                let extra = usize::from(m.octaves[o].kind == OCTAVE_RIDGED);
                let survives = o + usize::from(rung) < count + extra;
                // Past the table nothing survives and the top rungs keep one aliased octave (8c).
                let kept = o < live;
                assert_eq!(
                    survives | (live == 1),
                    kept | (live == 1),
                    "rung {rung}, octave {o}"
                );
                o += 1;
            }
            rung += 1;
        }
    }

    /// ★ THE HOME PLANET'S TABLE, EXACTLY (slice 8a stage 5). The coarsest wavelength clamps to
    /// [`LONG_WAVE_CAP_M`] and the table halves, so `λ(o) = 400 000 / 2^o` and the count is 14. Every
    /// number here is a power-of-two division of a whole number, so it is exact in a float and the
    /// survival rule's own boundary case can be tested at equality.
    ///
    /// It is a CHECKED claim and not an assumption: the caller reads the body's own frequency back
    /// out and compares it against this table before it uses it.
    fn home_waves(m: &BodyDefinition) -> Vec<f64> {
        let count = usize::from(m.octave_count);
        assert_eq!(count, 14, "the home planet draws fourteen octaves");
        let r = m.radius_m();
        let mut waves = Vec::new();
        for (o, octave) in m.octaves_at(0).iter().enumerate() {
            let exact = LONG_WAVE_CAP_M as f64 / f64::from(1u32 << o);
            let read = r / octave_frequency(octave);
            assert!(
                (read - exact).abs() < 1.0e-6 * exact,
                "octave {o}: the body reads {read} m against the table's {exact} m"
            );
            waves.push(exact);
        }
        waves
    }

    /// ★ THE SURVIVAL RULE IS ONE FUNCTION OF (wavelength, kind, cell) — slice 8a stage 5, the shape
    /// ruling T7 rule 1 asked for and the owner's T6 ask 3 needs in order to name the alias.
    ///
    /// Four statements, each of which could fail.
    ///
    /// 1. **One function decides every rung.** Walked over every rung of the ladder and every octave
    ///    of the table, [`octave_survives`] answers exactly what the body's stored row keeps — the
    ///    alias rungs excepted, which keep an octave the rule refused and SAY SO.
    /// 2. **A crest is a round wave of twice the wavelength.** The kind enters the rule in one place
    ///    and in one way.
    /// 3. **The boundary lives.** An octave that covers EXACTLY the required cells survives; a hair
    ///    under it, it does not. The home planet's crest at 3 125 m on rung 7 is that exact case, so
    ///    the line is not a matter of taste anywhere in the world.
    /// 4. **The required cells are read from the table**, never typed: 48.83 on the home planet,
    ///    which is the finest octave over a rung-0 cell.
    ///
    /// ★ RED BEFORE THE STAGE (the control: `octave_survives` ignoring its `kind` argument, so the
    /// draw and the test read the SAME wrong rule and statement 1 still agreed with itself):
    /// statement 2 read `octave 7: a crest is a round wave of twice the wavelength, left false,
    /// right true`.
    #[test]
    fn the_survival_rule_is_one_function_of_the_wavelength_the_kind_and_the_cell() {
        let m = home();
        let waves = home_waves(&m);
        let count = waves.len();
        // 4. The constant, from the table's own finest octave over a rung-0 cell.
        let cells = waves[count - 1] / f64::from(cell_m(0));
        assert!(
            (cells - 48.828_125).abs() < 1.0e-9,
            "the table asks {cells} cells per wavelength"
        );
        let cells_g = Gf::from_f64(cells);
        // 1. One function, every rung, every octave.
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            let cell = Gf::from_f64(f64::from(cell_m(rung)));
            let live = m.octaves_at(rung).len();
            let mut o = 0usize;
            while o < count {
                let rule =
                    octave_survives(Gf::from_f64(waves[o]), m.octaves[o].kind, cell, cells_g);
                let kept = o < live;
                if m.aliases_at(rung) {
                    // The alias keeps ONE octave the rule refused, and only that one.
                    assert!(!rule, "rung {rung}, octave {o}: the rule kept something");
                    assert_eq!(live, 1, "rung {rung}: the alias keeps exactly one octave");
                } else {
                    assert_eq!(rule, kept, "rung {rung}, octave {o}");
                }
                o += 1;
            }
            rung += 1;
        }
        // 2. The kind enters once: a crest reaches twice its wavelength and nothing else moves.
        let cell = Gf::from_f64(128.0);
        let mut o = 0usize;
        while o < count {
            let w = Gf::from_f64(waves[o]);
            assert_eq!(
                octave_survives(w, OCTAVE_RIDGED, cell, cells_g),
                octave_survives(w * Gf::from_i64(2), OCTAVE_SMOOTH, cell, cells_g),
                "octave {o}: a crest is a round wave of twice the wavelength"
            );
            o += 1;
        }
        // 3. The boundary lives, and a hair under it does not. Rung 7's cell is 128 m, so the
        //    threshold is 6 250 m — the reach of the 3 125 m crest, exactly.
        // The reading is taken ONCE, into a word the assertion and its message both read: an
        // expression that only a FAILING assertion evaluates is a region no green run walks.
        let threshold_m = Gf::from_f64(cells * 128.0).to_f64();
        assert!(
            (threshold_m - 6_250.0).abs() < 1.0e-9,
            "rung 7 asks {threshold_m} m"
        );
        assert!(octave_survives(
            Gf::from_f64(3_125.0),
            OCTAVE_RIDGED,
            cell,
            cells_g
        ));
        assert!(!octave_survives(
            Gf::from_f64(3_124.999),
            OCTAVE_RIDGED,
            cell,
            cells_g
        ));
        assert!(octave_survives(
            Gf::from_f64(6_250.0),
            OCTAVE_SMOOTH,
            cell,
            cells_g
        ));
        assert!(!octave_survives(
            Gf::from_f64(6_249.999),
            OCTAVE_SMOOTH,
            cell,
            cells_g
        ));
    }

    /// ★ THE ALIAS AT THE TOP RUNGS, AS A MEASURED FACT — the owner's ruling T6 ask 3 (2026-09-16):
    /// keep ONE aliased octave at the rungs past the table, **the ratio printed**, 8c named as the
    /// fix. The body states both, so a report and a picture's stamp read them instead of deriving
    /// them.
    ///
    /// Five statements, each of which could fail.
    ///
    /// 1. **The alias rungs are exactly 14 to 18** on the home planet — the globe from orbit, and no
    ///    rung a walker or a hull in the air ever draws.
    /// 2. **Their ratios are the widest octave over their own cell**: 24.41, 12.21, 6.10, 3.05 and
    ///    1.53 cells per wavelength.
    /// 3. **ONE of them stands under Nyquist** ([`NYQUIST_CELLS_PER_WAVE`]) — rung 18, at 1.53 cells
    ///    — and rung 17 stands at 3.05, over Nyquist but at a sixteenth of the table's own ask. That
    ///    is the size of the wrong the owner agreed to carry until 8c, MEASURED and not argued.
    /// 4. **The octave they keep is the WIDEST**, index 0 of a table drawn coarsest first.
    /// 5. **No rung below them aliases**, and every one of those covers at least half the table's own
    ///    ask, which is what a crest is content with.
    ///
    /// ★ RED BEFORE THE STAGE (the control: the alias row left false everywhere): statement 5 read
    /// `rung 15: 12.20703125 cells under the table's 48.828125`, because a rung that hides its alias
    /// is judged as a rung the rule answered.
    #[test]
    fn the_top_rungs_keep_one_aliased_octave_and_the_body_states_the_ratio() {
        let m = home();
        let waves = home_waves(&m);
        let count = waves.len();
        let cells = waves[count - 1] / f64::from(cell_m(0));
        // 4. The table is drawn coarsest first, so "the widest" is index 0. Both readings are
        //    taken into words first, for the reason the test above states.
        let widest = waves[0];
        let next_widest = waves[1];
        assert!(
            widest > next_widest,
            "the table is coarsest first: {widest} then {next_widest}"
        );
        let mut aliased: Vec<u8> = Vec::new();
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            let ratio = m.cells_per_wave(rung);
            let live = m.octaves_at(rung).len();
            assert!(
                (ratio - waves[live - 1] / f64::from(cell_m(rung))).abs() < 1.0e-6,
                "rung {rung}: the body states {ratio} cells per wavelength"
            );
            if m.aliases_at(rung) {
                aliased.push(rung);
                // 4. ONE octave, and it is the widest.
                assert_eq!(live, 1, "rung {rung}: the alias keeps one octave");
                assert_eq!(
                    m.octaves_at(rung)[0],
                    m.octaves[0],
                    "rung {rung}: the alias keeps the widest octave"
                );
            } else {
                // 5. What the rule itself kept covers at least half the table's ask — half, because
                //    that is what a crest is content with.
                assert!(
                    ratio >= cells * 0.5 - 1.0e-6,
                    "rung {rung}: {ratio} cells under the table's {cells}"
                );
            }
            rung += 1;
        }
        // 1. The exact set.
        assert_eq!(aliased, vec![14, 15, 16, 17, 18], "the alias rungs");
        // 2. The ratios, printed for the report and the stamp.
        let mut under_nyquist: Vec<u8> = Vec::new();
        for rung in aliased {
            let ratio = m.cells_per_wave(rung);
            let exact = LONG_WAVE_CAP_M as f64 / f64::from(cell_m(rung));
            assert!(
                (ratio - exact).abs() < 1.0e-6,
                "rung {rung}: {ratio} against {exact}"
            );
            if ratio < NYQUIST_CELLS_PER_WAVE {
                under_nyquist.push(rung);
            }
        }
        // 3. How far past honest it stands, and where.
        assert_eq!(under_nyquist, vec![18], "the rungs under Nyquist");
        assert!((m.cells_per_wave(17) - 3.051_757_812_5).abs() < 1.0e-9);
        assert!((m.cells_per_wave(18) - 1.525_878_906_25).abs() < 1.0e-9);
        // A rung past the table reads the last row, as every per-rung row of the body does.
        let past = RUNGS as u8;
        assert_eq!(m.aliases_at(past), m.aliases_at(past - 1));
        assert!((m.cells_per_wave(past) - m.cells_per_wave(past - 1)).abs() < 1.0e-12);
    }

    /// ★ A BODY WHOSE TABLE OUTLASTS ITS LADDER NEVER ALIASES — the other arm of the alias row, and
    /// the proof that the alias is a property of a SHORT TABLE against a LONG LADDER and never a
    /// rule the code applies to everybody.
    ///
    /// A rock of three kilometres draws its coarsest wavelength at the 20 km floor, so its table
    /// holds ten octaves against a ladder of far fewer rungs: the survival rule answers at every
    /// rung of it, the alias row is false throughout, and every rung covers at least half the
    /// table's own ask.
    ///
    /// ★ RED BEFORE THE STAGE (the control: the alias row left true everywhere): it read
    /// `rung 0: the rock draws no alias`.
    #[test]
    fn a_body_whose_table_outlasts_its_ladder_never_aliases() {
        let rock = BodyDefinition::from_seed(5, 3_000.0, ROCK_3KM_FACTS).expect("a rock");
        let count = usize::from(rock.octave_count);
        // The reason, stated: more octaves than rungs.
        assert!(
            usize::from(rock.ladder.rungs) <= count,
            "{} rungs against {count} octaves",
            rock.ladder.rungs
        );
        let finest = rock.radius_m() / octave_frequency(&rock.octaves[count - 1]);
        let cells = finest / f64::from(cell_m(0));
        let mut rung = 0u8;
        while rung < rock.ladder.rungs {
            assert!(
                !rock.aliases_at(rung),
                "rung {rung}: the rock draws no alias"
            );
            let ratio = rock.cells_per_wave(rung);
            assert!(
                ratio >= cells * 0.5 - 1.0e-3,
                "rung {rung}: {ratio} cells under the table's {cells}"
            );
            assert!(
                ratio >= NYQUIST_CELLS_PER_WAVE,
                "rung {rung}: {ratio} cells, under Nyquist"
            );
            rung += 1;
        }
    }

    /// ★ THE STEP A HANDOVER MAKES STANDS UNDER ONE CELL OF THE RUNG THAT TAKES OVER — the ladder's
    /// own tolerance, and the line `rung_disagreement` (the judge) measures on: one cell of rung
    /// `L + 1` is one pixel at that rung's own switch distance.
    ///
    /// ★ RED BEFORE ruling T7 rule 1: the pair 6 → 7 dropped the 3 125 m crest, 198.84 m against a
    /// 128 m cell — 1.55 cells, and the judge read 1.28 px at the ninety-ninth percentile.
    ///
    /// ★ **IT READS THE BODY'S OWN STEP BOUND SINCE THE CAP-ROCK BENCH LANDED** (slice 8a stage 4).
    /// A handover's step is no longer a plain difference of two amplitude sums: the terrace
    /// amplifies the dropped octaves by its rung's own Lipschitz constant and adds its fade's own
    /// step. The bench's strength is SOLVED against exactly this line, so the worst pair — 3 → 4,
    /// the rung the bench fades over — stands at 0.999 of a cell by construction, and the test is
    /// the measurement that says the solve did what it claims.
    #[test]
    fn the_step_at_every_handover_stands_under_one_cell_of_the_rung_that_takes_over() {
        let m = home();
        let mut rung = 0u8;
        let mut worst = 0.0f64;
        while rung + 1 < m.ladder.rungs {
            let next = rung + 1;
            let step_m = m.step_bound_m(rung);
            let cell = f64::from(cell_m(next));
            assert!(step_m >= 0.0, "rung {rung}: the bounds nest");
            let ratio = step_m / cell;
            if ratio > worst {
                worst = ratio;
            }
            assert!(
                step_m <= cell,
                "rung {rung} -> {next}: {step_m} m over a {cell} m cell"
            );
            rung = next;
        }
        assert!(
            worst > 0.5,
            "the line is not slack: the worst pair reads {worst}"
        );
    }

    #[test]
    fn dropping_octaves_keeps_at_least_one_and_the_bounds_nest() {
        let m = home();
        let count = usize::from(m.octave_count);
        assert_eq!(m.octaves_at(0).len(), count);
        assert_eq!(m.octaves_at(1).len(), count - 1);
        assert_eq!(m.octaves_at(15).len(), 1, "never fewer than one octave");
        assert_eq!(m.octaves_at(200).len(), 1);
        let mut rung = 0u8;
        while rung < 15 {
            assert!(
                m.relief_bound(rung + 1) <= m.relief_bound(rung),
                "the band nests"
            );
            assert!(m.dropped_bound(rung + 1) >= m.dropped_bound(rung));
            rung += 1;
        }
        assert_eq!(m.dropped_bound(0), Gi::ZERO);
        assert_eq!(m.dropped_bound_m(0), 0.0);
        assert!(m.dropped_bound_m(9) > 0.0);
        // A tiny body, SHAPE-limited: `0.077 R` is 231 m on a 3 km rock, so its whole band is
        // the small-body arm of the relief law and a short octave table.
        let rock = BodyDefinition::from_seed(5, 3_000.0, ROCK_3KM_FACTS).expect("a rock");
        assert!(rock.octave_count >= 1);
        assert!(rock.relief_bound_m(0) >= 100.0);
        assert!(
            rock.relief_arms_m().1 < rock.relief_arms_m().0,
            "the rock is shape-limited"
        );
        // A giant, STRENGTH-limited: at 14.83 m/s² and 1 326 kg/m³ the crust yields at 24 346 m,
        // which is what caps a 40 000 km body's band — never a typed ceiling of 12 000 m.
        let giant = BodyDefinition::from_seed(9, 40_000_000.0, GIANT_FACTS).expect("a giant");
        assert!(
            giant.relief_arms_m().0 < giant.relief_arms_m().1,
            "the giant is strength-limited"
        );
        assert!(giant.relief_bound_m(0) <= 26_000.0);
        assert!(
            giant.octave_count >= 12,
            "a 400 km wave halves to 30 m in fourteen steps"
        );
        // The giant's radius is the widest the address can name: its words still fit.
        assert!(giant.radius_steps.raw() < 1 << 33);
        assert_eq!(draw_m(&mut SplitMix64::new(1), 5, 6), 5);
    }
}
