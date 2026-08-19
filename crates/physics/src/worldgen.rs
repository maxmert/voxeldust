//! THE seed universe generator (D-45(a); the placement arc S5) — the single source of truth for
//! realm→region geometry, now living where SL4 puts it: in the motion crate, beside the closed-form
//! celestial math it draws on. Closed-form `f(seed)`: every shard computes the IDENTICAL containment
//! forest at boot from the shared universe seed, so the geometry is REPLICATED BY CONSTRUCTION — no
//! shared mutable state, no inter-shard bytes (HR1).
//!
//! The split rule (SL5's guard, stated once): anything that READS THE SEED STREAM or MINTS A BODY
//! lives here; anything that only reads an already-built `&[RealmRegion]` stays in
//! `vd_core::worldgen` (topology utilities over this generator's output — `level_of`,
//! `coord_of_realm`, `ancestor_realms`, `pin_realm_of`, the neighbourhood scope). ONE generator; the
//! crossing path cannot name it (the `crate_isolation` SL4 law).
//!
//! The P3 forest models the mandate hierarchy **Universe ⊃ Galaxy ⊃ StarSystem ⊃ Planet** (walk scale:
//! Station + Area as first-class hand-placed realms — task #133); the visual/demand presets generate
//! the compressed-real Kepler systems every shipped shard boots (SL5: THE world, one of it).
//!
//! # ★ THE DISCOVERY-PERMANENCE LAW (owner ruling 2026-08-18, standing)
//!
//! **The generator is APPEND-ONLY in its draw stream.** A seed stream is positional exactly like
//! the wire: every draw's MEANING is its position, so inserting or reordering a draw re-rolls every
//! draw after it — and with it every orbit, every star, every albedo a player has already
//! discovered. Therefore:
//!
//! - a NEW derived quantity draws AFTER every existing draw of its stream, never between two
//!   (the practised pattern: the star's photometric draw appended after the planet elements, the
//!   albedo pass appended after the star, the 3-D placement pair appended after the albedos);
//! - every draw's stream POSITION is documented at its site and guarded by a bit-exact pin, so a
//!   reorder fails a named test before it can re-roll a world;
//! - a draw is never deleted while anything downstream of its position survives — retiring one
//!   means retiring the whole suffix behind a stated world-numbers change.
//!
//! What a player has found stays found: the world may gain content forever, and never loses or
//! moves what the seed already said.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use core::f64::consts::TAU;

use crate::celestial::{OrbitalElements, orbital_state};
use crate::motion::Motion;
use crate::taxonomy::{
    FrostThresholds, GalaxyType, SpectralClass, classify_spectral, habitable_zone_radius_au,
    main_sequence_luminosity, orbital_axis_au, sample_imf_mass, sample_rayleigh,
};
use vd_core::frame::FramePlacement;
use vd_core::geometry::{AoiConfig, BandError, Boundary, ContainmentBand, RealmRegion};
use vd_core::pose::{LatticePos, RealmId, Tier, frame_for_realm};
use vd_core::realm_path::RealmLevel;
use vd_core::rng::{SplitMix64, child_seed, realm_stream};
use vd_core::worldgen::{
    AREA_A, GALAXY, GALAXY_SEED, PLANET_A, STATION_A, SYSTEM_A, SYSTEM_A_SEED, SYSTEM_B, UNIVERSE,
    UNIVERSE_SEED, WALK_DEMAND_AOI_GRACE_S, ancestor_realms, grace_ticks_from_seconds, level_of,
    neighbourhood_scope,
};

/// The inner (acquire) edge of the P3 static containment band, metres inside a surface.
const CONTAINMENT_INSET_M: f64 = 1.0;
/// The outer (release) edge of the P3 static containment band, metres outside a surface.
const CONTAINMENT_OUTSET_M: f64 = 2.0;

// --- P3 WALK-SCALE geometry (placeholders; P4/P5 makes every radius/center `f(seed[, tick])`, D-44). ---
/// The ambient-root (Universe) radius — effectively unbounded; an entity beyond it still resolves to the
/// Universe by the container fold IDENTITY. Non-renderable (far above the render-extent threshold).
const UNIVERSE_R_M: f64 = 1.0e9;
/// The galaxy radius — FINITE (it encloses the star systems), and it is the between-systems space an
/// entity occupies after leaving one system SOI and before entering the next. RENDERABLE (below the extent
/// threshold) so the client draws it as the CONTAINING box around the two systems — an entity in the gap is
/// visibly still inside the Galaxy realm, never orphaned. Contains System B's far face (130 + 40 = 170).
const GALAXY_R_M: f64 = 180.0;
/// A star-system SOI radius (walk scale).
const SYSTEM_SOI_R_M: f64 = 40.0;
/// A planet SOI radius (walk scale), nested inside a system.
const PLANET_SOI_R_M: f64 = 10.0;
/// System B's center on +X — a disjoint sibling of System A with a WALKABLE gap of galaxy between them
/// (System A far-face 40, System B near-face 90 ⇒ a ~50 m pure-Galaxy gap: leaving A you are IN the Galaxy
/// realm until you enter B). The round-trip probe points 0/50/100 still resolve System A / Galaxy / System B.
const SYSTEM_B_OFFSET_M: f64 = 130.0;
/// Planet A's center inside System A (offset from the star at the origin).
const PLANET_A_OFFSET_M: f64 = 20.0;
/// Station A's center inside System A, on the -X side (opposite Planet A on +X), clear of the origin
/// crowd + the round-trip legs at x = 0/50/100. A Cartesian box, not an SOI shell.
const STATION_A_OFFSET_M: f64 = -25.0;
/// Station A's box half-extent (a small docked-station volume). `|-25| + 5 = 30 < 40` ⇒ fully inside
/// System A's r=40 SOI.
const STATION_HALF_M: f64 = 5.0;
/// Area A's center inside Planet A, MEASURED FROM PLANET A — every placement is an offset from its own
/// parent, and Area A's parent is the planet, not the system. +5 puts it at +25 in the system's frame
/// (Planet A is at +20, r=10), so its box still spans x∈[22,28] within the planet's sphere and is still
/// OFFSET from the (20,0,0) escape-SOI probe, which must resolve to Planet 7 and not the Area.
///
/// This was written as +25 — the absolute, in the SYSTEM's frame, on a field that means "offset from my
/// parent". Nothing caught it because the login descent carried one unconverted point all the way down
/// and so compared every level's boundary in the system's frame, where the two agree. Once the descent
/// converts as it steps — the parent doing the downward arithmetic, per the ground rule — an area sitting
/// +25 from a planet that is itself only 10 wide is nowhere near it, and the world stops being coherent.
const AREA_OFFSET_M: f64 = 5.0;
/// Area A's box half-extent (a small sub-planet district volume).
const AREA_HALF_M: f64 = 3.0;

// --- FA-5 (D-45(a)) VISUAL-scale single-system generator — the COMPRESSED-REAL game scale (the ONE
// geometry; `visual_scale` is its static-render expression, `visual_demand` its live demand-cluster
// expression). Every visual geometry number is DERIVED (parameterized helpers below), not a literal. ---
/// `child_seed` salt distinguishing PLANET-kind children under a system (a fixed kind discriminant;
/// `child_seed` avalanches `(parent, salt, index)`, so a distinct salt keeps planet ids off other kinds).
const PLANET_SALT: u64 = 0x504c_414e_4554; // "PLANET"
/// The star realm's `child_seed` salt (taxonomy arc T2) — a star realm's seed avalanches off
/// its system exactly as a planet's does, on its own salt so the id spaces never collide.
const STAR_SALT: u64 = 0x0000_5354_4152; // "STAR"
/// The moon realm's `child_seed` salt (taxonomy arc T3) — a MOON IS A PLANET (the owner's
/// ruling, read literally: `RealmId::Planet` whose parent is a `RealmId::Planet`); its seed
/// avalanches off its parent planet on this salt, rung-indexed.
const MOON_SALT: u64 = 0x0000_4d4f_4f4e; // "MOON"
/// Canup & Ward's satellite-mass multiplier bounds (log-uniform, per-moon draw M.1): the
/// realised satellite-system mass spread around the 1e-4 accretion budget (the four solar
/// giants measure 1.21–2.48e-4).
const MOON_MASS_MULTIPLIER_BOUNDS: (f64, f64) = (0.5, 2.0);
/// `child_seed` salt distinguishing SYSTEM-kind children under a galaxy — the sibling-kind discriminant
/// for stars, exactly as [`PLANET_SALT`] is for planets. A system's identity is `f(galaxy, index)`, so two
/// galaxies never mint the same system id and a system's planets never collide with another system's.
const SYSTEM_SALT: u64 = 0x5359_5354_454d; // "SYSTEM"
/// `child_seed` salt distinguishing FIXTURE-PLANTED player-built children (look_horizon.md slice 5
/// G-IDENTICAL — the SL5 fixture-forest doctrine) from every seed-generated kind: a planted
/// station's id is `f(its host system, this salt, index)` and a planted area's is `f(its host
/// planet, this salt, index)`, so plants can never collide with generated ids or with each other.
const FIXTURE_SALT: u64 = 0x0046_4958_5455_5245; // "FIXTURE"
/// SYSTEM_A's RNG lineage root→leaf `[Universe, Galaxy, System]` — MUST equal
/// `realm_path::system_path(SYSTEM_A_SEED).lineage_seeds()` so every shard hosting System A draws the
/// IDENTICAL per-system stream by construction (HR1); consumed once by [`generate_system_forest`].
#[cfg(test)]
const SYSTEM_A_LINEAGE: [u64; 3] = [UNIVERSE_SEED, GALAXY_SEED, SYSTEM_A_SEED];

// ===== THE IN-SYSTEM TRUE-SIZE RE-SOLVE (real-scale design §3.3, landed by the taxonomy arc) ====
//
// The interim compressed in-system world (150 m shells, 3.95 m SOIs, a synthetic Kepler-tuned
// central mass, an AU compression factor) is DELETED. In-system space is TRUE SIZE (χ = 1
// exactly): the ladder is anchored to the star's own luminosity, planet masses are drawn, radii
// and SOIs derived, and each system's shell is SOLVED by the one clearance law.

/// How many star systems THE world's galaxy holds — the existing census parameter (owner ruling
/// Q-B: the placement law went 3-D and seeded; the census DERIVATION is P10's, this count is not).
const WORLD_SYSTEM_COUNT: u32 = 3;

/// THE FROZEN STREAM PREFIX WIDTH (the Stream Law, celestial_taxonomy_design §3.0/§7.1): the
/// per-system draw stream shipped with FIVE planets' element draws before the star/albedo/
/// placement draws. Growing the planet count re-rolls NOTHING because the draws for planets
/// beyond this prefix are APPENDED after the frozen prefix (elements 0..5, star, albedo 0..5,
/// placement ×2 — byte-identical forever), never inserted. A STREAM-shape constant, not a world
/// knob: the world's planet count is [`derived_planet_count`].
const LEGACY_STREAM_PLANETS: u32 = 5;

/// The smallest planet mass the generator draws, Earth masses — Mercury, the smallest confirmed
/// planet. A cited literal (real-scale design §3.3.4), stated plainly as one.
const PLANET_MASS_LO_MEARTH: f64 = 0.0553;
/// Protoplanetary disc-to-star mass fraction (Andrews & Williams 2005, ApJ 631:1134) — the
/// per-planet mass budget is `DISC_MASS_FRACTION · M★ / N_planets`.
const DISC_MASS_FRACTION: f64 = 0.01;
/// Jupiter's mass in Earth masses — the absolute per-planet draw cap (the disc arm binds on
/// every M dwarf; this cap is reachable only around heavier stars).
const M_JUP_MEARTH: f64 = 317.8;
/// The planet-mass draw is LOG-UNIFORM over its bounds: slope 1.0 through the existing
/// [`sample_imf_mass`] log-uniform limit branch (real-scale design §3.3.4).
const PLANET_MASS_SLOPE: f64 = 1.0;
/// Neptune's semi-major axis in AU — with [`crate::taxonomy::FROST_COEFF_AU`] it fixes the disc
/// outer edge as `NEPTUNE_SMA_AU/FROST_COEFF_AU = 11.137×` the frost line, measured on the one
/// planetary system we have (real-scale design §3.3.3).
const NEPTUNE_SMA_AU: f64 = 30.07;

/// The planet count, DERIVED and scale-free (real-scale design §3.3.3): the number of ladder
/// steps `a0·ratio^n` inside the disc edge `(NEPTUNE_SMA_AU/FROST_COEFF_AU) · frost_coeff ·
/// √L`. Both sides scale with √L, so the count is a property of the SPACING LAW, not the seed:
/// **9 for every star of every world** (`0.4·1.7^n ≤ 30.07 ⇔ n ≤ 8.14`). The √L cancellation is
/// why this needs no luminosity argument.
#[must_use]
pub fn derived_planet_count(a0_au: f64, ratio: f64, disc_edge_au: f64) -> u32 {
    (0..u32::MAX)
        .take_while(|&n| orbital_axis_au(n, a0_au, ratio) <= disc_edge_au)
        .count() as u32
}

/// The taxonomy-arc fixture plant sizes — the owner's Q3 build cases (real-scale design §6.3,
/// measured admission margins 8.51× and 4,261×): a 10 km CITY under the home system and a 20 m
/// STRUCTURE on the inner planet. Cited constants of the plant (player-built content the seed
/// never emits), not world knobs.
const FIXTURE_CITY_R_M: f64 = 1.0e4;
const FIXTURE_STRUCTURE_R_M: f64 = 20.0;

/// The minimum angular size (radians) a realm must subtend to enter Area of Interest — a realm is visible
/// (streams in) out to `extent · cot(θ/2) + velocity-lead`. ONE config constant, no kind-branch: this
/// single number sets how far away EVERY realm kind appears, from a planet to a galaxy.
///
/// 1.5° (was 8°). Two independent measurements forced it down, and both are about the same thing — at 8°
/// a body only exists once it is ~14 of its own radii away, which is close enough to be already large:
///
/// 1. A planet was invisible from inside its own star system. Its wake radius was 59.5 m against a system
///    150 m in radius, so you could cross a system's boundary and find it apparently empty — measured, and
///    now pinned by `a_planet_is_visible_from_anywhere_inside_its_own_system`.
/// 2. Bodies POPPED IN at full size instead of growing from a dot, which is the opposite of the owner's
///    stated arrival/departure behaviour (a system shrinks to a dot as you leave, grows from one as you
///    arrive). Angular size IS that behaviour: the smaller this angle, the smaller a body is when it first
///    appears. At 1.5° it enters the scene ~76 radii out — a few pixels — and grows all the way in.
///
/// COST, stated because it is real and pays at every scale: every realm's wake radius grows by the same
/// 5.3×, so more realms are awake at once and the demand loop carries more of them. That is the ONE knob's
/// nature — it cannot be widened for planets alone without branching on kind, which is forbidden. Note the
/// inter-system ring is DERIVED from this same factor ([`UniverseConfig::visual_geometry`]), so a system
/// still sleeps until you approach it no matter what this is set to — that behaviour is invariant here.
const VISIBILITY_THETA_MIN_RAD: f64 = 0.026_180;

// ===== THE OUTER GEOMETRY (real-scale addendum §A2 — owner ruling 2026-08-18, OPTION C) =========
// The four changed numbers, each with its derivation. The chain runs DOWNWARD from the storage
// fence for the root and DOWNWARD from the star for everything inside a system; the two join at
// exactly one place (the placement radius), which is why everything in-system is BIT-IDENTICAL
// across this re-solve (§A1.1) — pinned by the unchanged in-system goldens.

/// ▲ 1. THE UNIVERSE RADIUS: `2⁵¹ m` — solved from the position store, never chosen (§A2.1/§A2.2).
/// The binding budget is F1, the wire-ingress domain: every per-axis cell of a sanitized pose must
/// survive `StampedPose::sanitized` unclamped, `|cell| ≤ CELL_DOMAIN_MAX = i64::MAX/2` (the clamp
/// exists so a cell DIFFERENCE cannot overflow — the hostile-sender panic cure). With one binary
/// octave of headroom against that clamp (`K_SPAN = 2` — the same discipline constant the exactness
/// budget always carried, with a new referent, §A2.2/OQ-3: a LAWFUL position may sit OUTSIDE the
/// root shell, so the clamp must not be reachable from there):
///
/// `K_SPAN · R_uni / FINE_CELL_EDGE_M ≤ CELL_DOMAIN_MAX + 1`
/// `R_uni = 2⁵¹ m = 2 251 799 813 685 248 m ≈ 0.238 ly` — passes with EXACT equality:
/// `2 × 2⁶¹ = 4 611 686 018 427 387 904 = CELL_DOMAIN_MAX + 1` (headroom exactly 2.0000×, domain
/// occupancy exactly 50.0000 %). `2⁵¹` is exactly f64-representable, so the fence and the shipped
/// shell radius agree bit-for-bit. [`guard_root_representable`] is the boot fence on this budget —
/// and its refusal is THE NAMED P10 TRIGGER (R3): a world that outgrows the FINE lattice is the day
/// the galaxy cell lattice is needed.
const REAL_UNIVERSE_R_M: f64 = 2_251_799_813_685_248.0; // 2^51, exact

/// The storage-fence headroom octave (§A2.2, `K_SPAN = 2`) — one binary octave between the root
/// shell and the sanitize clamp, so the clamp is unreachable from any lawful position (OQ-3's
/// recommendation, adopted: a silent clamp is the defect class this coordinate exists to prevent).
const K_SPAN: f64 = 2.0;

/// ▲ 2. THE GALAXY RADIUS: `R_uni − outset` — THE CORRECTED FORMULA (§A2.2, curing H-27: the main
/// design's `R_gal = R_uni/2` was a residue of the REJECTED doubling clearance form and cost a full
/// octave of star gap). The ambient realms carry no look, so their clearance degenerates to their
/// own bound and the shells would TOUCH; the strictness the nesting fence needs is the child's own
/// release band — but a band's governed arm contains τ = T_WAKE, a MEASURED boot latency, and a
/// world radius may never be a function of how fast a shard happens to boot (SL5 / determinism /
/// no-magic-numbers — H-02). The cure is the τ-FREE UPPER BOUND the band-solvability fence
/// supplies: `dt·N ≤ τ/2` ⇒ the governed band is at most `2 · v_cap · dt · N`, so
///
/// `outset = v_cap(R_uni) · GEOMETRY_TICK_DT_S · BAND_TICKS_N · BAND_TAU_HEADROOM`
///         `= (2·R_uni/T_TRAVERSE_S) · 0.02 · 3 · 2 = 3.0023997515803307e12 m`
/// `R_gal  = R_uni − outset = 2.2487974139336678e15 m ≈ 0.237698 ly`
///
/// Cost of the strictness margin: 0.133 % of `R_uni`, against the 50 % an octave would cost.
const REAL_GALAXY_R_M: f64 = REAL_UNIVERSE_R_M
    - (2.0 * REAL_UNIVERSE_R_M / T_TRAVERSE_S)
        * GEOMETRY_TICK_DT_S
        * BAND_TICKS_N
        * BAND_TAU_HEADROOM;

/// The owner's acceptance bar ("journeys in MINUTES") — THE speed law's one policy number, whose
/// single home is [`vd_core::flight::TRAVERSE_S`] (the S3 slice landed the law:
/// `realm_speed_cap_mps`, the geometric throttle, the ramp and the governor all live in
/// `vd_core::flight`, consumed at the sim's one integrator seam). The geometry solve reads the SAME
/// constant through this alias (its τ-free outset above), so the shells and the speed law can never
/// disagree about T — one number, two readers, zero drift.
const T_TRAVERSE_S: f64 = vd_core::flight::TRAVERSE_S;
/// The geometry solve's FIXED reference tick (§A2.2's `tick_dt` input — a constant of the solve,
/// NEVER the cluster's live tick: two clusters at different tick rates must boot the identical
/// world, so no cluster parameter may enter a radius).
const GEOMETRY_TICK_DT_S: f64 = 0.02;
/// `N = max(K_SAFETY, n_entry) = 3` — the in-band tick count the band law owes (§A2.2/§A3.4).
const BAND_TICKS_N: f64 = 3.0;
/// The band-solvability fence's own factor (`dt·N ≤ τ/2` ⇒ the τ term can at most DOUBLE the
/// governed band) — not a new literal, the fence's ×2 (§A2.2, H-02's cure).
const BAND_TAU_HEADROOM: f64 = 2.0;

/// ▲ 3. THE PLACEMENT RADIUS (the star gap): `R_gal − clearance = 2.248490504408914e15 m
/// = 0.2376656 ly = 15 030.23 AU` — what the storage budget leaves after the clearance the solve
/// owes (§A2.2/§A2.3; `VISUAL_RING_SLACK`, the old "THROWAWAY" padding fraction, is DELETED — the
/// radius is never padded by taste again). The clearance is the ONE clearance law (§3.2)
/// evaluated at the FUTURE in-system re-solve's targets, landed NOW so the outer geometry never
/// moves again when the in-system slices (taxonomy: star radius, real SOIs, shells) arrive:
///
/// `clearance = child_clearance(bound, look) = max(bound, look·(1+cot(θ/2))) + look·(1+cot(θ/2))`
/// with `bound = R_sys,max = 2.967026419e11 m` (the largest target system shell, §3.3.5 — solved
/// from the Demircan–Kahraman radius + re-anchored ladder of `System(13979593561158050752)`) and
/// `look = R★,max = 1.318892e8 m` (that star's photosphere radius under the same mass–radius law
/// from its ALREADY-PINNED mass draw 0.16179874709518627 M☉). Both enter as CITED derived targets
/// of the addendum's chain — the in-system machinery that recomputes them lands with the taxonomy
/// slice, and the named pin below flips loudly if that slice lands different numbers.
/// ★ RE-MEASURED AT THE FLAG DAY (the taxonomy arc's in-system re-solve, exactly the flip the
/// paragraph above announced): the shells are now SOLVED by `system_shell_r_m` (the one
/// clearance law at the mass cap, with the honest per-rung worst-look — the composition
/// envelope's super-puff bound, not the bare Chen–Kipping cap), and the solved values supersede
/// the addendum's hand-derived 2.967026419e11 / 1.575690652e11 (a_0-rounding class differences,
/// named by the taxonomy design §12.4 for the flag day to resolve — resolved HERE by
/// measurement). The reservation stays a CONSTANT (the placement radius is config, not
/// seed-derived), pinned EQUAL to the measured solve by the named tests.
pub const TARGET_SYSTEM_BOUND_MAX_M: f64 = 296_703_425_982.042_3;
/// The largest target star's photosphere radius (§3.3.1's `R★` for the most massive pinned
/// draw, `star_radius_m(0.16179874709518627)` — pinned equal by the four-number test).
const TARGET_STAR_LOOK_MAX_M: f64 = 131_889_247.210_144_1;
/// The HOME system's target shell (§3.3.5 row 1, `System(7)`: 1.575690652e11 m = 1.053284 AU) —
/// the flight-table gate's system-leg distance (`2·R_sys` edge-to-edge) and its warp-departure
/// ceiling input. The SAME cited-target discipline as [`TARGET_SYSTEM_BOUND_MAX_M`]: the taxonomy
/// slice's in-system re-solve recomputes it, and the gate that reads it flips loudly if that slice
/// lands a different number. `pub` for exactly that gate.
pub const TARGET_SYSTEM_BOUND_HOME_M: f64 = 158_226_185_287.501_65;
/// The home system's OUTER planet's target SOI at the maximum mass draw (§3.3.4/§3.3.5:
/// 8.567390e9 m) — the flight-table gate's planet-leg distance ("planet surface out to its own
/// shell"). Cited-target discipline as above; the D-REAL-1 equality (realm shell == gravitational
/// SOI) lands it for real with the taxonomy slice.
pub const TARGET_PLANET_SOI_OUTER_HOME_M: f64 = 8_567_390_468.048_405;

/// ▲ 4. THE COMPRESSION χ = 16.378× — stated as the measurement it is (§A2.3): the real mean
/// nearest-neighbour stellar separation over the placement radius. Real separation
/// `0.55396 · n^(−1/3)` at `n = 0.1 pc⁻³` (RECONS 10-parsec census) `= 3.682666e16 m = 3.8926 ly`;
/// `χ = 3.682666e16 / 2.2484905e15 = 16.378` (was 46 463× under the superseded option (a) —
/// 2 837× more real interstellar space). The galaxy cell lattice (P10) exists to drive χ toward 1;
/// [`guard_root_representable`]'s refusal is its named trigger.
const RECONS_STELLAR_DENSITY_PER_PC3: f64 = 0.1;
/// Mean nearest-neighbour coefficient for a Poisson point field (`0.55396·n^(−1/3)`).
const MEAN_NN_COEFF: f64 = 0.55396;
/// One parsec in metres (IAU): the census density's unit.
const PARSEC_M: f64 = 3.085_677_581_491_367e16;

/// The §3.2 clearance a parent owes ONE child: enough to CONTAIN it, plus enough that its picture
/// has already fallen below the minimum angle — the `+ vis` form, whose strictness proof costs
/// 3.3 % of what the rejected doubling form cost (§3.2's sidebar).
fn child_clearance_m(child_bound_m: f64, child_look_m: f64, theta_min_rad: f64) -> f64 {
    let vis = child_look_m * (1.0 + visibility_factor(theta_min_rad));
    child_bound_m.max(vis) + vis
}

/// The placement radius (the star gap) — see the ▲ 3 derivation above.
fn real_placement_r_m() -> f64 {
    REAL_GALAXY_R_M
        - child_clearance_m(
            TARGET_SYSTEM_BOUND_MAX_M,
            TARGET_STAR_LOOK_MAX_M,
            VISIBILITY_THETA_MIN_RAD,
        )
}

/// ▲ 4 as a number: the between-systems compression χ — the real mean nearest-neighbour stellar
/// separation over the placement radius (16.378× on THE world; 1.000000 in-system, exactly,
/// because no in-system compression factor exists to be anything else). Public so the pin and any
/// report read the ONE derivation.
#[must_use]
pub fn real_compression_chi() -> f64 {
    let real_separation_m =
        MEAN_NN_COEFF * RECONS_STELLAR_DENSITY_PER_PC3.powf(-1.0 / 3.0) * PARSEC_M;
    real_separation_m / real_placement_r_m()
}

/// A world whose root outgrows the FINE lattice's representable budget — [`guard_root_representable`]'s
/// loud refusal, carrying every number of the verdict. **THIS REFUSAL IS THE NAMED P10 TRIGGER
/// (R3):** the day a world needs more than the FINE tier can hold exactly is the day the galaxy
/// cell lattice (the COARSE tier's activation) is built — compression → 1 is not a wish, it is
/// this fence's condition.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "the root shell ({root_r_m} m = {root_cells} FINE cells) with K_SPAN {k_span} outgrows the \
     FINE lattice's sanitized domain (CELL_DOMAIN_MAX = {domain_max} cells): occupancy \
     {occupancy_pct}% > 100%/K_SPAN — the world has outgrown the millimetre tier; the cure is \
     the galaxy cell lattice (P10), never a widened clamp"
)]
pub struct RootNotRepresentable {
    pub root_r_m: f64,
    pub root_cells: f64,
    pub k_span: f64,
    pub domain_max: i64,
    pub occupancy_pct: f64,
}

/// The measured storage budget a representable root prints (occupancy + headroom, §A2.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootBudget {
    /// The root radius in FINE cells.
    pub root_cells: f64,
    /// Domain occupancy: root_cells / CELL_DOMAIN_MAX (0.5 exactly on THE world).
    pub occupancy: f64,
    /// Headroom against the clamp: CELL_DOMAIN_MAX / root_cells (2.0 exactly on THE world).
    pub headroom: f64,
}

/// THE STORAGE FENCE (real-scale addendum §A2.1 F1 / §A4.9 R3 — a boot fence beside
/// `guard_visibility_climb_bounded` in every world-deriving process): the root shell, with its
/// `K_SPAN` headroom octave, must fit the FINE lattice's sanitized domain, so the wire-ingress
/// clamp is unreachable from any lawful position. Refusing is THE NAMED P10 TRIGGER. Measured on
/// THE world: occupancy exactly 50.0000 %, headroom exactly 2.0000× (`2⁵¹ m = 2⁶¹ cells;
/// 2 × 2⁶¹ = CELL_DOMAIN_MAX + 1` — the equality is the construction).
///
/// # Errors
/// [`RootNotRepresentable`] with every number of the verdict, naming P10 as the cure.
pub fn guard_root_representable(
    config: &UniverseConfig,
) -> Result<RootBudget, RootNotRepresentable> {
    let edge = vd_core::pose::FINE_CELL_EDGE_M;
    let domain_max = vd_core::pose::CELL_DOMAIN_MAX;
    let root_cells = config.scale.universe_r_m / edge;
    let budget = K_SPAN * root_cells;
    // `+ 1.0` exactly as the ▲ 1 derivation states: 2·2⁶¹ equals CELL_DOMAIN_MAX + 1, so THE
    // world passes with exact equality — the headroom octave is the construction, not slack.
    if budget > domain_max as f64 + 1.0 {
        return Err(RootNotRepresentable {
            root_r_m: config.scale.universe_r_m,
            root_cells,
            k_span: K_SPAN,
            domain_max,
            occupancy_pct: 100.0 * root_cells / domain_max as f64,
        });
    }
    Ok(RootBudget {
        root_cells,
        occupancy: root_cells / domain_max as f64,
        headroom: domain_max as f64 / root_cells,
    })
}

/// THE SEEDED-SYSTEM SEPARATION FENCE as a boot guard (Q-B's re-derived fence over the general
/// point set): build THE world's seeded system placements and judge every pair. Run beside the
/// climb fence at every world-deriving boot.
///
/// # Errors
/// [`SiblingsOverlap`] naming the first overlapping pair.
pub fn guard_seeded_systems_disjoint(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<(), SiblingsOverlap> {
    let centres: Vec<(RealmId, DVec3, f64)> = generate_system_forest(seed_universe, config)
        .iter()
        .filter(|b| b.parent == Some(GALAXY))
        .map(|b| {
            (
                b.realm,
                placement_offset(b.placement),
                b.shape.circumscribed_extent(),
            )
        })
        .collect();
    seeded_systems_disjoint_3d(&centres)
}

/// A body the generator emits before lowering — its realm, parent, shape, and placement. The
/// walk roster uses `StaticOffset` placements (the byte-identity source); THE world's planets use
/// `Orbital`, whose static tick-0 anchor is baked at boot. [`to_regions`]
/// lowers a slice of these to the frozen [`RealmRegion`] forest.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GeneratedBody {
    realm: RealmId,
    parent: Option<RealmId>,
    shape: Boundary,
    placement: Placement,
    /// The body's photometric identity, drawn from the SAME per-realm seed stream that generated
    /// it (the window lane's marker datum — owner-approved 2026-08-15/16,
    /// `docs/design/window_lane.md` §2.2/§2.8: a sleeping child's point of light is authored by
    /// its parent from the child's own generation stream). `Some` on every SYSTEM the seed
    /// generator emits (the star's mass → class → luminosity); `None` on the ambient
    /// Universe/Galaxy shells, on planets (their photometric ladder is an owed later draw), and
    /// on every hand-placed walk body (a player-built station has no seed stream). Stored on the
    /// body row exactly as the sibling derived field (`placement`'s orbital elements) is —
    /// NEVER lowered onto `RealmRegion` and never on the wire: the Slice-A marker emit reads it
    /// off the booted forest and ships TLV scalars, not this struct.
    photometrics: Option<StarPhotometrics>,
    /// The taxonomy row (T1): `Some` on every generated planet — computed, never drawn from
    /// the wire; `None` on ambient/hand-placed/fixture bodies, exactly as `photometrics` is.
    /// Never lowered onto `RealmRegion`, never on the wire: every process derives the whole
    /// forest from the seed at boot (the complete SL6 answer — a planet shard computes its own
    /// mass, gravity, temperature and atmosphere locally, from the seed, with no message).
    taxon: Option<crate::taxonomy::BodyTaxon>,
    /// THE LOOK (real-scale design §3.0 — the BOUND/LOOK split): the outline this body DRAWS.
    /// `None` on the ambient Universe/Galaxy of THE world (never drawn, structurally); `Some`
    /// on every drawable body — a system's look is its STAR's photosphere (`star_radius_m` of
    /// the drawn mass), a planet's its Chen–Kipping radius at its drawn mass, a walk/plant
    /// body's its own bound (bound == look at human scale). Lowered onto `RealmRegion.look`.
    look: Option<Boundary>,
}

/// A star system's drawn photometric identity — pure `f(seed)` through the taxonomy layer
/// (`sample_imf_mass` → `classify_spectral` → `main_sequence_luminosity`), drawn ONCE per system
/// at generation from the system's own [`realm_stream`]. Off-wire DATA (HR1): the window lane's
/// marker (`BodyStmt::Marker`) will carry TLV-framed scalars derived from this at Slice A, and
/// the pinned-values test freezes each system's draw on THE world so any stream drift is loud.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarPhotometrics {
    /// The star's drawn mass (solar masses) — the ONE u01 draw everything below derives from.
    /// On a REFLECTOR's datum (a planet, Slice C1) this is the ILLUMINATING star's mass: a
    /// reflector shines by its star, so its color class and its mass provenance are the star's.
    pub mass_msun: f64,
    /// Morgan-Keenan class from the drawn mass (the marker's color class). A reflector carries
    /// its ILLUMINATOR's class — reflected light keeps the star's color.
    pub class: SpectralClass,
    /// Main-sequence luminosity `L/Lsun` from the drawn mass (the marker's luma scalar). A
    /// reflector carries the star's luminosity geometrically diluted at its own orbit and scaled
    /// by its cross-section × its seed-drawn albedo (see [`reflected_photometrics`]).
    pub luma_lsun: f64,
}

/// The canonical GEOMETRIC-ALBEDO table (lo, hi) a reflector's seed-drawn albedo spans — a
/// physical passable table like [`SpectralClass::MASS_BOUNDS`], not a tuning knob: solar-system
/// geometric albedos run from ~0.1 (dark rock — the Moon, Mercury) to ~0.7 (full cloud decks —
/// Venus). Scale-independent; at near-real scale the same bounds hold unchanged.
pub const GEOMETRIC_ALBEDO_BOUNDS: (f64, f64) = (0.1, 0.7);

/// A sleeping REFLECTOR's marker datum (Slice C1 — `docs/design/window_lane.md` §1.1 item 3b: "a
/// point-of-light datum per DIRECT child"; the planets' half of the per-system draw, owed since
/// Slice 0 and landed with the flag day that made it load-bearing): the star's light reflected.
/// Class = the illuminator's (reflected light keeps the star's color); luma = `L★ · albedo ·
/// r² / (4d²)` — the closed-form geometric dilution of starlight over the orbit radius `d`,
/// intercepted by the body's cross-section `r²`, scaled by ONE seed-drawn albedo over the
/// canonical table. Pure f(seed, config), scale-free, straight-line (HR5).
fn reflected_photometrics(
    star: &StarPhotometrics,
    albedo_u01: f64,
    radius_m: f64,
    orbit_m: f64,
) -> StarPhotometrics {
    let (lo, hi) = GEOMETRIC_ALBEDO_BOUNDS;
    let albedo = lo + albedo_u01 * (hi - lo);
    StarPhotometrics {
        mass_msun: star.mass_msun,
        class: star.class,
        luma_lsun: star.luma_lsun * albedo * (radius_m * radius_m) / (4.0 * orbit_m * orbit_m),
    }
}

/// Where a body sits in its parent inertial frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Placement {
    /// A fixed frame-local offset (the walk roster — the byte-identity source).
    StaticOffset(DVec3),
    /// A Keplerian orbit. PRODUCED IN PRODUCTION: [`generate_system_forest`] emits one per planet of
    /// THE world (`UniverseConfig::world` → `visual_demand` → 5 planets per system), and every shipped
    /// shard boots that forest. This doc used to claim "no production producer at P3" under an
    /// `#[allow(dead_code)]` — both false, and the false claim sat on the exact discriminator SL4
    /// governs (audit finding 28); the producer test
    /// `the_world_produces_an_orbital_planet_in_every_system` makes the claim a measurement now.
    Orbital(OrbitalElements),
}

/// The region CENTER a body's boundary sits at IN ITS OWN FRAME, as a `cell == ZERO` [`LatticePos`]:
/// - a **moving** (`Orbital`) body authors its position LIVE through the placement book its parent's
///   shard writes each tick (`author_placements` over the injected `MotionFn`s), so its boundary is
///   at the frame ORIGIN — center **ZERO**, NEVER the epoch. (This is the moving-realm crossing-flap fix: a
///   nonzero epoch center would be DOUBLE-COUNTED against the live frame placement in
///   [`region_signed_distance`](vd_core::geometry::region_signed_distance) — shifting the SOI ~one orbit off
///   the body, so the parent shard and the body's own shard disagree on containment and a crossing flaps.)
/// - a **`StaticOffset`** body's frame is the identity, so its fixed offset IS the boundary center.
///
/// Walk scale is ALL `StaticOffset` ⇒ unchanged ⇒ byte-identical; only the visual/canonical movers flip to
/// ZERO (whose live pose every other consumer — demand, AoI, feed, render — already reads from the frame).
///
/// ⚠ OWED (the placement arc, DEFERRED): this field's ZERO-for-a-mover is the residual two-meanings
/// leak (finding 27) — the boot fence no longer reads it (it takes the boot's worst-instant
/// [`ChildReach`](vd_core::geometry::ChildReach)), but the FIELD's presence still stores "where I sit"
/// on the region record; its deletion (positions living only in authored placement books) lands with
/// the motion-roster split.
fn region_center_of(placement: Placement) -> LatticePos {
    // THE PRODUCER SWITCH (real-scale addendum §A4.8 row 7): the generator emits NORMALIZED
    // centres — the integer half carries the whole-quantum part, so a static child's placement
    // enters the lattice at boot instead of riding a bare f64 offset. `from_metres` on BOTH arms
    // (the Orbital arm's ZERO normalizes to ZERO — value-identical, one rule).
    match placement {
        Placement::Orbital(_) => LatticePos::from_metres(DVec3::ZERO, Tier::Fine),
        Placement::StaticOffset(_) => {
            LatticePos::from_metres(placement_offset(placement), Tier::Fine)
        }
    }
}

/// The frame-local offset of a placement. `Orbital` evaluates [`orbital_state`] ONCE at tick 0
/// (Tier-2 libm is boot-time here, not a per-tick oracle — the cross-host gate is SPIKE-6a).
fn placement_offset(placement: Placement) -> DVec3 {
    match placement {
        Placement::StaticOffset(v) => v,
        Placement::Orbital(elements) => orbital_state(&elements, 0.0).position,
    }
}

/// Lower generated bodies to the frozen `RealmRegion` forest under `config`. Every region shares
/// the one static containment band; the frame is the realm's canonical authority frame
/// (`frame_for_realm`) with parent-provenance from the body (so the Area `.expect` cannot fire),
/// keeping the input-side containment seam and the output-side `transfer_frame` conversions in
/// agreement.
fn to_regions(bodies: &[GeneratedBody], config: &UniverseConfig) -> Vec<RealmRegion> {
    let band = config
        .band
        .build()
        .expect("containment band edges are valid by construction");
    bodies
        .iter()
        .map(|b| {
            // RLM Step 2: per-realm AoI = factor × the body's OWN finite extent, the dead-zone widened
            // by the occupant speed + THIS child's own closing speed. THE scalar comes from the ONE
            // motion discriminant ([`Motion::closing_speed_mps`]) — the fifth rival has-orbit test
            // (`orbital_of(..).map_or(0.0, v_peri)`) collapsed into that accessor (D-PLACE-1; the
            // batch review caught the accessor dead beside the still-live test).
            let v_child = motion_of(b.placement).closing_speed_mps();
            let aoi = config
                .interest
                .build(b.shape.finite_extent(), v_child)
                .expect("aoi band edges are valid by construction");
            RealmRegion {
                realm: b.realm,
                center: region_center_of(b.placement),
                frame: frame_for_realm(b.realm, b.parent)
                    .expect("roster realms have a canonical frame"),
                shape: b.shape,
                band,
                aoi,
                look: b.look,
                parent: b.parent,
                // Look horizon slice 4 (§3.4.4): the interior band is stamped HERE, from the
                // FULL forest this map runs over — before any scope filter drops the
                // grandchildren it is derived from. That is what settles the §3.4.4 CLAIM: a
                // galaxy shard's boot roster row for a star system carries the reach with no
                // message ever crossing a boundary (Ask D stays deferred).
                interior_band: interior_band(
                    interior_reach_m(bodies, b.realm, config.planet.ecc_cap),
                    &config.interest,
                    v_child,
                ),
            }
        })
        .collect()
}

/// A body's INTERIOR REACH (look_horizon.md §3.4.4): the largest distance from its centre at
/// which something INSIDE it is still visible — the max over its DIRECT children of (that
/// child's worst-instant excursion at the eccentricity cap + that child's visibility reach),
/// the same two terms the climb measurement walks with (§3.3.2's identity: one formula, one
/// worst-case convention). `0.0` for a childless leaf — nothing inside, nothing to reach.
/// On THE world a star system's reach is `142.045826247 + 302.058663384 = 444.104489631` m.
fn interior_reach_m(bodies: &[GeneratedBody], parent: RealmId, ecc_cap: f64) -> f64 {
    bodies
        .iter()
        .filter(|c| c.parent == Some(parent))
        .map(|c| {
            // The visibility term reads the child's LOOK (the picture that can be seen), not
            // its bound (real-scale design §3.0); a look-less child contributes no reach.
            worst_hop_excursion_capped_m(&c.placement, ecc_cap)
                + c.look.map_or(0.0, |look| {
                    vd_core::geometry::visibility_reach_m(
                        look.finite_extent(),
                        VISIBILITY_THETA_MIN_RAD,
                    )
                })
        })
        .fold(0.0, f64::max)
}

/// The interior band for one child region (look_horizon.md §3.4.4, monomorphic — both arms
/// driven by named tests): spin-up AT the interior reach, tear-down widened by the SAME derived
/// velocity lead every AoI band carries (`|v_rel|·dt·(K_SAFETY + extra)` — no new number
/// anywhere). Inert for a leaf (zero reach) and wherever the whole AoI machinery is inert
/// (walk/canonical byte-identity: the live ctor's reject arm is never touched there, the same
/// discipline as [`InterestConfig::build`]).
fn interior_band(reach_m: f64, interest: &InterestConfig, v_child: f64) -> AoiConfig {
    if (reach_m <= 0.0) | !interest.is_live() {
        AoiConfig::inert()
    } else {
        AoiConfig::for_velocity_safe(
            reach_m,
            1.0,
            1.0,
            aoi_v_rel_mps(reach_m, interest.occupant_v_max_mps + v_child),
            interest.tick_dt_s,
            interest.grace_ticks,
            interest.k_safety_extra,
        )
        .expect("a positive reach with a positive closing speed builds a valid band")
    }
}

/// The AoI dead-zone's velocity input, FLOORED at the realm's own traverse speed
/// `2·extent / T_TRAVERSE_S` — the §4.2(a) speed-cap expression used here as the derived scale
/// floor the real-scale geometry needs (addendum H-11's cure, landed at the geometry slice: "the
/// AoI dead-zone is a constant 2.5 ticks at every scale" broke structurally the day the shells
/// went astronomic — at a 2.25e15 m ambient extent a 15 m/s pad is BELOW ONE ULP of the spin-up
/// radius, so `spin + pad == spin` and the band constructor rightly refused the collapsed
/// dead-zone). The floor is the same chain the outer geometry derives from (`T_TRAVERSE_S`), so
/// no new number enters; once the speed-law slice lands, `v_rel` can never be below the realm's
/// own cap anyway — this lands that clearance early. Byte-identical on every interim-scale row,
/// MEASURED: the floor binds only where `extent > v_rel·T/2` (the two ambient shells on THE
/// world; no walk row — walk's distinct tear factor out-binds the pad everywhere).
fn aoi_v_rel_mps(extent_m: f64, v_rel_mps: f64) -> f64 {
    v_rel_mps.max(2.0 * extent_m / T_TRAVERSE_S)
}

/// The DIRECT MOVING children a shard hosting `hosted_realm` AUTHORS (D-45(a) realm-unification FA-2b):
/// each direct child (`parent == hosted_realm`) whose placement is a live `Orbital`. Under LAW-1 a
/// passive orbiting body is the ZERO-SIGNAL case — the parent shard re-authors its live pose each tick
/// from these `OrbitalElements` (the boot wraps them as injected `MotionFn`s for the placement
/// writer), never a static region `center`.
/// Returned `(realm, elements)` so the sim keys it against each region by realm. A branchless-shim (HR5):
/// the `Orbital`/`StaticOffset` match lives in the monomorphic [`orbital_of`] helper, not this closure.
/// The walk roster is ALL `StaticOffset`, so this is EMPTY at walk scale (byte-identity); the canonical
/// seed generation (P4/FA-5) is what populates it.
fn moving_children(
    bodies: &[GeneratedBody],
    hosted_realm: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    bodies
        .iter()
        .filter(|b| b.parent == Some(hosted_realm))
        .filter_map(|b| orbital_of(b.placement).map(|e| (b.realm, e)))
        .collect()
}

/// The `OrbitalElements` of a placement, or `None` for a static one — the MONOMORPHIC discriminator that
/// keeps [`moving_children`]'s closure branchless (HR5: the `match` is covered once, here).
fn orbital_of(placement: Placement) -> Option<OrbitalElements> {
    match placement {
        Placement::Orbital(elements) => Some(elements),
        Placement::StaticOffset(_) => None,
    }
}

/// The [`Motion`] a generated placement lowers to — Kepler for an orbital body, [`Motion::Fixed`] at
/// the stored offset otherwise. THE one lowering from "what the generator drew" onto the motion
/// discriminant, so every scalar motion property the lowering needs (`closing_speed_mps` today) is
/// read off the accessor instead of re-running a has-orbit test beside it (D-PLACE-1's fifth rival).
/// Monomorphic; both arms covered by the `to_regions` AoI tests.
fn motion_of(placement: Placement) -> Motion {
    match placement {
        Placement::Orbital(elements) => Motion::Kepler(elements),
        Placement::StaticOffset(at) => Motion::Fixed(FramePlacement::moving(at, DVec3::ZERO)),
    }
}

/// The direct-child `RealmLevel` roster for `hosted` — from the SAME `(seed, config)` forest
/// [`to_regions`]/[`moving_children`] consume (`b.parent == Some(hosted)`), so the AoI child roster and
/// the containment region roster never diverge (L1). Closed-form `f(seed, config, hosted)`. `filter_map`
/// drops any non-seed-lineage child (a ship — never present in a seed forest). Each level builds the
/// child's `RealmCoord` as `own_coord.child(level)` (§3).
#[must_use]
pub fn direct_child_levels(
    seed_universe: u64,
    config: &UniverseConfig,
    hosted: RealmId,
) -> Vec<RealmLevel> {
    generate_system_forest(seed_universe, config)
        .iter()
        .filter(|b| b.parent == Some(hosted))
        .filter_map(|b| level_of(b.realm))
        .collect()
}

/// [`moving_children`] over the seed forest a shard boots — the AUTHORED moving-child roster for
/// `hosted_realm` (its direct `Orbital` children, each `(realm, elements)`). Closed-form
/// `f(seed, hosted_realm)`; EMPTY at walk scale (byte-identity), populated at canonical scale (P4/FA-5).
#[must_use]
pub fn moving_children_for(
    _seed_universe: u64,
    hosted_realm: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    let config = UniverseConfig::walk_scale();
    moving_children(&generate_walk_forest(&config), hosted_realm)
}

// ===== FA-5 (D-45(a)) the config-driven VISUAL/canonical-scale single-system generator ==========
// The SAME generator serves the VISUAL synthetic-mass preset (window-friendly orbiting boxes) AND the
// canonical real-mass preset (P4 real proportions) with ZERO kind-match — only the config differs.

/// THE ONE visibility formula, re-exported from its home (look_horizon.md §3.3.2 — the formula
/// lives in `vd-core` with three consumers: the world solve, the boot measurement, the runtime
/// tripwire; this module is two of them and BUILDS the third's config). Never a second copy.
use vd_core::geometry::visibility_factor;

// (`two_level_clearance_m` and `galaxy_shell_r_m` are DELETED — real-scale addendum §9.2: the
// upward interim solve collapsed into the ONE clearance law (`child_clearance_m`, §3.2) and the
// downward storage-fence chain (§A2.2). Their pins (`FROZEN_TWO_LEVEL_CLEARANCE_M`,
// `FROZEN_TWO_LEVEL_WORST_MARGIN_M`) retire with them; the §3.2 identity is the successor.)

// (`synthetic_central_mass`, `visual_au_to_render_m`, `visual_planet_soi_r_m`,
// `visual_outer_sma_render_m`, `visual_central_mass_kg` are DELETED — real-scale design §3.3.7:
// the compressed visual in-system geometry died with the re-solve; orbits are true-size around
// the star's real drawn mass.)

/// One planet's five DRAWN orbital quantities, in the FROZEN order (ecc-u, incl-u, Ω, ω, M₀) so
/// the forest is pure `f(seed)`. ALL clamps are BRANCHLESS method calls (HR5): the ecc cap is
/// `.min(ecc_cap)` (an `if ecc > cap` would leave an UNREACHABLE true-arm at `ecc_sigma`≈0.03);
/// inclination is UN-clamped (no convergence domain, and `sample_rayleigh` is `≥ 0`).
///
/// The DERIVED halves — `sma` (the √L-anchored ladder) and `central_mass` (the star's drawn
/// mass) — are NOT here, because they consume NO draw and depend on the star's photometrics,
/// which the frozen stream order draws AFTER the legacy planets' elements: draws first,
/// derivations second ([`planet_orbital_elements`] assembles them).
#[derive(Clone, Copy, Debug)]
struct PlanetElementDraws {
    ecc: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly_epoch: f64,
}

fn planet_element_draws(config: &UniverseConfig, stream: &mut SplitMix64) -> PlanetElementDraws {
    PlanetElementDraws {
        ecc: sample_rayleigh(stream.next_f64(), config.planet.ecc_sigma).min(config.planet.ecc_cap),
        inclination: sample_rayleigh(stream.next_f64(), config.planet.incl_sigma),
        raan: stream.next_f64() * TAU,
        arg_periapsis: stream.next_f64() * TAU,
        mean_anomaly_epoch: stream.next_f64() * TAU,
    }
}

/// Assemble one planet's [`OrbitalElements`] from its frozen draws + the two DERIVED halves:
/// `sma = a0 · √L · ratio^n` in TRUE metres (THE LADDER LAW, real-scale design §3.3.2 — the
/// dead [`habitable_zone_radius_au`] made live: `√L` IS the habitable-zone radius at flux 1, so
/// the whole ladder is measured in habitable-zone radii and rung 2 sits at 0.748 S⊕ for every
/// star at every seed), and `central_mass` = the star's real drawn mass in kg.
fn planet_orbital_elements(
    config: &UniverseConfig,
    draws: PlanetElementDraws,
    n: u32,
    star: &StarPhotometrics,
) -> OrbitalElements {
    let a0_au = config.planet.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    OrbitalElements {
        sma: orbital_axis_au(n, a0_au, config.planet.orbital_ratio) * crate::taxonomy::AU_M,
        ecc: draws.ecc,
        inclination: draws.inclination,
        raan: draws.raan,
        arg_periapsis: draws.arg_periapsis,
        mean_anomaly_epoch: draws.mean_anomaly_epoch,
        central_mass: star.mass_msun * crate::taxonomy::M_SUN_KG,
    }
}

/// Draw one star system's photometric identity from ITS OWN per-system `stream` (the window
/// lane's per-system draw — owner-approved 2026-08-15/16, `docs/design/window_lane.md` §2.2:
/// "luma drawn from the same seed stream the parent generated the child from"). ONE u01 draw:
/// mass through the bounded-IMF inverse-CDF, then class + luminosity as closed-form derivations
/// of that mass — the taxonomy discipline (no rejection loop, bit-reproducible). Straight-line,
/// monomorphic (HR5). Parameters are DATA off [`StellarConfig`] (no magic numbers); the MK mass
/// bounds are the taxonomy's canonical passable table, the MLR segments the config's.
fn draw_star_photometrics(st: &StellarConfig, stream: &mut SplitMix64) -> StarPhotometrics {
    let mass_msun = sample_imf_mass(
        stream.next_f64(),
        st.imf_slope,
        st.mass_lo_msun,
        st.mass_hi_msun,
    );
    StarPhotometrics {
        mass_msun,
        class: classify_spectral(mass_msun, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(mass_msun, &st.mlr_segments),
    }
}

/// The seed of the `n`-th star system in a galaxy. System 0 keeps [`SYSTEM_A_SEED`] — it is the identity
/// every existing fixture, label and gate already names — and the rest avalanche off the galaxy through
/// the same [`child_seed`] every other sibling set uses, so a system's identity is a pure function of
/// (galaxy, index) and no two galaxies ever collide.
#[must_use]
fn system_seed_at(n: u32) -> u64 {
    // Branchless in the HR5 sense: ONE covered comparison, no nested control flow.
    if n == 0 {
        SYSTEM_A_SEED
    } else {
        child_seed(GALAXY_SEED, SYSTEM_SALT, u64::from(n))
    }
}

/// How many star systems a galaxy holds — its CENSUS, drawn from the galaxy's own stream inside
/// `[system_count_lo, system_count_hi]`. Pure `f(universe_seed)` against the galaxy lineage, so every
/// shard agrees on the population without exchanging a byte, and two galaxies from one universe differ
/// without anyone authoring either. `lo == hi` pins the count exactly (the walk roster's shape).
///
/// One draw, no rejection loop — the taxonomy discipline: a sampler is a closed-form map from one uniform.
#[must_use]
fn galaxy_system_count(seed_universe: u64, config: &UniverseConfig) -> u32 {
    let lo = config.galaxy.system_count_lo;
    let hi = config.galaxy.system_count_hi.max(lo);
    let span = u64::from(hi - lo) + 1;
    let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED]);
    // `next_f64` is [0,1); scaling by the inclusive span and truncating lands in [lo, hi] with the last
    // bucket reachable only at exactly 1.0, which the generator never produces — hence the `min`.
    let draw = (stream.next_f64() * span as f64) as u64;
    lo + u32::try_from(draw.min(span - 1)).unwrap_or(0)
}

/// Where the `n`-th system sits in its galaxy — THE 3-D SEEDED PLACEMENT LAW (owner ruling Q-B,
/// 2026-08-18: *"don't do 3 star systems in the line — get rid of this code; the whole world
/// generated from the seeds"*). The collinear ring (evenly-spaced angles on one circle in the XZ
/// plane — three stars on a line at N = 3, zero transverse parallax, §A7.1's named cost) is
/// DELETED; each non-home system takes a seeded uniform direction on the sphere at the derived
/// placement radius:
///
/// - `cos_polar = 2·u − 1`, `azimuth = TAU·v` — the standard closed-form uniform-on-the-sphere
///   construction from the system's own TWO appended stream draws (branchless, bit-reproducible);
/// - the RADIUS is the derived placement radius ([`real_placement_r_m`]) for every system — the
///   placement radius CLASS: the direction is the seed's, the magnitude is the storage budget's;
/// - system 0 (the HOME system) stays at the GALACTIC ORIGIN — the anchor the landed J1 fence
///   asserts ("a home↔galaxy crossing is numerically an identity in the drawn space"), the lineage
///   identity every fixture and gate names. The zero MULTIPLIER is the whole decision, no branch —
///   and the home draws its two placement u01s exactly like every sibling, so the stream SHAPE is
///   uniform across systems and a future re-rule of the home anchor shifts no draw.
///
/// The separation guarantee moved WITH the law: the ring's closed-form neighbour separation
/// (`2·r·sin(π/(n−1))`) is replaced by the pairwise 3-D fence over the seeded point set
/// ([`seeded_systems_disjoint_3d`] — closed-form per pair, refusal loud), measured on THE world by
/// the named pins beside it.
#[must_use]
fn system_center_at(config: &UniverseConfig, n: u32, dir_u01: f64, azim_u01: f64) -> DVec3 {
    let radius = config.stellar.system_ring_r_m;
    let cos_polar = 2.0 * dir_u01 - 1.0;
    let sin_polar = (1.0 - cos_polar * cos_polar).max(0.0).sqrt();
    let azimuth = TAU * azim_u01;
    let anchored = f64::from(u32::from(n != 0));
    DVec3::new(
        sin_polar * azimuth.cos(),
        cos_polar,
        sin_polar * azimuth.sin(),
    ) * (radius * anchored)
}

/// THE 3-D SEPARATION FENCE — the ring's closed-form fence RE-DERIVED for the seeded point set
/// (Q-B: a proof rewrite, never a weakening). For every pair of seeded systems the pairwise
/// centre distance must exceed the sum of their circumscribed extents (containment stays
/// single-answer: no position may be inside two sibling authorities), and — the wake law's half —
/// every pair must be separated by more than one system's AoI spin-up reach, so a system is ASLEEP
/// at departure from any sibling. Closed form per pair (an exact subtraction and two sums), loud
/// on refusal with both names and the measured gap.
fn seeded_systems_disjoint_3d(centres: &[(RealmId, DVec3, f64)]) -> Result<(), SiblingsOverlap> {
    for (i, (a, a_at, a_ext)) in centres.iter().enumerate() {
        for (b, b_at, b_ext) in centres.iter().skip(i + 1) {
            if (*b_at - *a_at).length() < a_ext + b_ext {
                return Err(SiblingsOverlap {
                    a: *a,
                    b: *b,
                    parent: GALAXY,
                });
            }
        }
    }
    Ok(())
}

/// Two sibling realms whose boundaries INTERSECT — the authoring mistake that makes "which realm contains
/// this position" have two answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiblingsOverlap {
    /// The two siblings, ordered as authored.
    pub a: RealmId,
    pub b: RealmId,
    /// Their shared parent.
    pub parent: RealmId,
}

/// Refuse a forest in which two STATICALLY-placed siblings intersect.
///
/// WHY THIS IS A CORRECTNESS FENCE, not tidiness. Authority is decided by descending into the deepest
/// child whose boundary contains you. If two siblings overlap, a position inside both has two equally
/// valid answers, and which shard owns you depends on iteration order — a coin flip that decides where
/// your input is applied and who simulates your collisions. It cannot be repaired downstream, because by
/// then the ambiguity is already a routing decision.
///
/// STATIC SIBLINGS ONLY, and that limit is real rather than convenient: an orbiting body's lowered region
/// sits at its frame ORIGIN (center zero — its position is authored live through its frame each tick), so
/// two planets are indistinguishable from co-located by any static comparison. Judging orbits needs their
/// SHELLS compared — two orbits are disjoint iff their radii differ by more than the sum of their
/// boundaries, at every eccentricity — which is a separate check over the moving roster. Recorded rather
/// than silently skipped: an unchecked orbital overlap is the same defect one level down.
///
/// Conservative on both sides: it compares FARTHEST-surface-point distances, so a doubtful placement is
/// refused rather than waved through. Cross-frame siblings are not judged (their numbers are not
/// comparable) — the same honest decline the parent-fit check makes.
/// RUN AT TEST TIME, not at boot, and that is a decision rather than an omission: the generator is
/// deterministic, so proving it over the shipped presets across a sweep of seeds proves every world that can
/// actually be booted, while a per-boot pass would be quadratic in a galaxy's population for an answer that
/// cannot change between runs. If worlds ever stop being purely seed-derived — the moment players place
/// structures the generator did not — this moves to the placement path, where the new body is the only thing
/// that needs judging.
#[cfg(test)]
fn siblings_disjoint(bodies: &[GeneratedBody]) -> Result<(), SiblingsOverlap> {
    for (i, a) in bodies.iter().enumerate() {
        let Placement::StaticOffset(a_at) = a.placement else {
            continue; // an orbit is judged on its shell, not its epoch — see above
        };
        let Some(parent) = a.parent else {
            continue; // the ambient root has no siblings
        };
        for b in bodies.iter().skip(i + 1).filter(|b| b.parent == a.parent) {
            let Placement::StaticOffset(b_at) = b.placement else {
                continue;
            };
            let reach = a.shape.circumscribed_extent() + b.shape.circumscribed_extent();
            if (b_at - a_at).length() < reach {
                return Err(SiblingsOverlap {
                    a: a.realm,
                    b: b.realm,
                    parent,
                });
            }
        }
    }
    Ok(())
}

/// A grandchild-or-deeper body that would be VISIBLE from just outside one of its ancestors — the
/// two-level bound broken by geometry. Carries every number of the verdict so the failure names
/// itself: the worst-instant distance of the body's centre from the ancestor's centre, the body's
/// extent, the resulting minimum viewer distance, and the distance the visibility rule requires.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "{body:?} subtends >= the visibility threshold from just outside its ancestor {ancestor:?}: \
     worst-instant centre distance {worst_dist_m} m, extent {extent_m} m, minimum viewer distance \
     {d_min_m} m, but the interest band keeps it visible out to {required_m} m — a not-running \
     realm two levels down would owe pixels its parent's placement marker cannot author"
)]
pub struct GrandchildVisibleOutside {
    pub body: RealmId,
    pub ancestor: RealmId,
    /// Worst-instant distance of the body's centre from the ancestor's centre (triangle bound —
    /// each hop contributes its own worst-instant offset magnitude).
    pub worst_dist_m: f64,
    /// The body's finite extent (the same extent the interest band judges visibility on).
    pub extent_m: f64,
    /// `R_ancestor − worst_dist − extent`: how close a viewer just outside the ancestor can get.
    pub d_min_m: f64,
    /// `extent · cot(θ_min/2)`: the distance out to which the interest band keeps the body visible.
    pub required_m: f64,
}

/// One hop's WORST-INSTANT offset magnitude — the same machinery the boot's `ChildReach` roster
/// states: a static child's authored offset, a mover's closed-form worst-instant excursion
/// ([`Motion::max_excursion_m`], the apoapsis — never a re-derived `a·(1+e)` beside it).
/// TEST-ONLY since look_horizon slice 2: the boot fence measures with the CAPPED excursion
/// ([`worst_hop_excursion_capped_m`] — the solve's own worst case); the drawn-eccentricity walk
/// below stays as the pinned drawn-margin history.
#[cfg(test)]
fn worst_hop_excursion_m(placement: &Placement) -> f64 {
    match placement {
        Placement::StaticOffset(at) => at.length(),
        Placement::Orbital(elements) => Motion::Kepler(*elements).max_excursion_m(Tier::Fine),
    }
}

/// The two-level VERDICT NUMBERS for EVERY `(body, ancestor)` pair — ancestor two or more levels
/// up — with the body at its worst-instant position and the viewer just outside the ancestor's
/// boundary at closest approach. PURE GEOMETRY over the roster: no realm kinds, no motion kinds (a
/// hop's excursion is a magnitude whichever way it is produced). The threshold enters as the SAME
/// `cot(θ/2)` the interest band uses ([`visibility_factor`]) — the condition
/// `angular_size(extent, d_min) < θ_min` is exactly `d_min > extent · cot(θ_min/2)`. A pair is an
/// OFFENCE iff `d_min ≤ required` ([`grandchild_visibility_offences`] filters); a green pair's
/// margin `d_min − required` is the measured headroom the re-solve pins.
#[cfg(test)]
fn grandchild_visibility_pairs(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    let by_id: std::collections::BTreeMap<RealmId, &GeneratedBody> =
        bodies.iter().map(|b| (b.realm, b)).collect();
    let factor = visibility_factor(theta_min_rad);
    let mut pairs = Vec::new();
    for body in bodies {
        // The subject's PICTURE is its LOOK (real-scale design §3.0); a look-less body draws
        // nothing and has no two-level visibility question.
        let Some(look) = body.look else { continue };
        let extent_m = look.finite_extent();
        // Walk the ancestor chain, accumulating the worst-instant centre distance hop by hop.
        let mut worst_dist_m = worst_hop_excursion_m(&body.placement);
        let mut hops = 1_usize;
        let mut cursor = body.parent;
        while let Some(ancestor_id) = cursor {
            let ancestor = by_id
                .get(&ancestor_id)
                .expect("the generated forests resolve every parent (guarded at boot)");
            if hops >= 2 {
                let d_min_m = ancestor.shape.finite_extent() - worst_dist_m - extent_m;
                let required_m = extent_m * factor;
                pairs.push(GrandchildVisibleOutside {
                    body: body.realm,
                    ancestor: ancestor_id,
                    worst_dist_m,
                    extent_m,
                    d_min_m,
                    required_m,
                });
            }
            worst_dist_m += worst_hop_excursion_m(&ancestor.placement);
            hops += 1;
            cursor = ancestor.parent;
        }
    }
    pairs
}

/// Every pair of [`grandchild_visibility_pairs`] that IS an offence — the body would still be
/// VISIBLE (subtend ≥ `theta_min_rad`) from just outside its ancestor, equality included (the
/// margin the shell solve reserves is what keeps the worst lawful seed strictly clear).
#[cfg(test)]
fn grandchild_visibility_offences(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    grandchild_visibility_pairs(bodies, theta_min_rad)
        .into_iter()
        .filter(|p| p.d_min_m <= p.required_m)
        .collect()
}

// ===== THE LOOK HORIZON's boot MEASUREMENT (look_horizon.md §3.3.2 — slice 2) ==================
// The boolean guard this replaces (`guard_grandchildren_invisible_outside`) gave a yes-or-no
// answer over the seed forest ONLY — §3.3.1 proves that cannot serve as a termination proof (the
// generator emits universe/galaxy/systems/planets and nothing else, so player-built content never
// entered it, and its predicate would refuse the first city). The MEASUREMENT below reports a
// NUMBER per body — how many levels its picture must travel — and the fence refuses a world whose
// number exceeds what the look carrier can carry (`vd_wire::session_flow::LOOK_CARRIER_ARITY`).

/// One hop's worst-instant offset at the ECCENTRICITY CAP — the SAME worst-case convention the
/// shell solve bounds against (`galaxy_shell_r_m`'s `outer_sma · (1 + ecc_cap)`), which is what
/// makes the measured stopping slack and the solve's reserved margin ONE equation written twice
/// (§3.3.2's identity; the drawn-eccentricity margin is the LOOSER `FROZEN_TWO_LEVEL_WORST_MARGIN_M`
/// — the engineering-relevant number is the reserved one). Straight-line per arm (HR5).
fn worst_hop_excursion_capped_m(placement: &Placement, ecc_cap: f64) -> f64 {
    match placement {
        Placement::StaticOffset(at) => at.length(),
        Placement::Orbital(elements) => elements.sma * (1.0 + ecc_cap),
    }
}

/// One body's measured VISIBILITY CLIMB (look_horizon.md §3.3.2): how far its own picture must
/// travel for every lawful observer to draw it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityClimb {
    pub body: RealmId,
    /// The HIGHEST ancestor from just outside which the body is STILL visible (the body itself
    /// when not even its parent's outside can see it — the degenerate one-level climb).
    pub top: RealmId,
    /// How many levels the body's own picture must travel: `1` (its own batch reaching its
    /// parent — every child's baseline) plus one per consecutive ancestor, parent upward, from
    /// outside which it is still visible.
    pub levels: usize,
    /// The slack at the level where visibility STOPPED: `d_min − required` at the first ancestor
    /// that does NOT see the body (POSITIVE — the measured headroom §3.3.2 pins at the world's
    /// own containment margin), or the ROOT's non-positive figure when the climb never stopped
    /// inside the forest (a pathological world the arity fence then refuses).
    pub slack_m: f64,
}

/// The climb walk over a generated forest — the SAME ancestor walk as
/// [`grandchild_visibility_pairs`] with the `hops >= 2` filter dropped (§3.3.2's construction)
/// and the excursions taken at the eccentricity CAP (the solve's own worst case). Bodies without
/// a parent (the root) have no climb and report nothing.
fn visibility_climbs(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
    ecc_cap: f64,
) -> Vec<VisibilityClimb> {
    let by_id: std::collections::BTreeMap<RealmId, &GeneratedBody> =
        bodies.iter().map(|b| (b.realm, b)).collect();
    let mut climbs = Vec::new();
    for body in bodies.iter().filter(|b| b.parent.is_some()) {
        // The climb carries the body's PICTURE — its LOOK (real-scale design §3.0). A body
        // with no look draws nothing, so there is no picture to carry and no climb (the
        // ambient galaxy's levels-2 root artifact of the outer re-solve dissolves here).
        let Some(look) = body.look else { continue };
        let extent_m = look.finite_extent();
        let required_m = vd_core::geometry::visibility_reach_m(extent_m, theta_min_rad);
        let mut worst_dist_m = worst_hop_excursion_capped_m(&body.placement, ecc_cap);
        let mut levels = 1_usize;
        let mut top = body.realm;
        let mut slack_m = f64::INFINITY;
        let mut cursor = body.parent;
        while let Some(ancestor_id) = cursor {
            let ancestor = by_id
                .get(&ancestor_id)
                .expect("the generated forests resolve every parent (guarded at boot)");
            let d_min_m = ancestor.shape.finite_extent() - worst_dist_m - extent_m;
            slack_m = d_min_m - required_m;
            if slack_m > 0.0 {
                break; // NOT visible from outside this ancestor: the climb stops HERE.
            }
            // Still visible (equality included — the same convention as the offence filter):
            // the picture must travel one level further.
            top = ancestor_id;
            levels += 1;
            worst_dist_m += worst_hop_excursion_capped_m(&ancestor.placement, ecc_cap);
            cursor = ancestor.parent;
        }
        climbs.push(VisibilityClimb {
            body: body.realm,
            top,
            levels,
            slack_m,
        });
    }
    climbs
}

/// THE BOOT MEASUREMENT (look_horizon.md §3.3.2, replacing the boolean guard): for every body of
/// the generated world, how many levels its picture must travel — the highest still-visible
/// ancestor, and the slack at the level where visibility stopped. On THE world today: max climb
/// 2, planet stopping slack exactly the containment margin (the solve identity, G-CLIMB's pin).
#[must_use]
pub fn measure_visibility_climb(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<VisibilityClimb> {
    visibility_climbs(
        &generate_system_forest(seed_universe, config),
        VISIBILITY_THETA_MIN_RAD,
        config.planet.ecc_cap,
    )
}

/// A world (or a candidate placement) whose measured visibility climb EXCEEDS what the look
/// carrier can carry — the fail-loud shape the fences print: the body and its numbers.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "{body:?} needs its picture carried {levels} levels (visible from outside every ancestor up \
     to {top:?}; slack at the stop {slack_m} m), but the look carrier serves {arity} — the owner's \
     Q3 ruling (look_horizon.md RULINGS 2026-08-17): the arity STAYS 2 and this REFUSES at interim \
     scale; the near-real-scale re-solve is the scheduled cure, and its first gate run must \
     include measure_visibility_climb"
)]
pub struct VisibilityClimbExceeded {
    pub body: RealmId,
    pub top: RealmId,
    pub levels: usize,
    pub slack_m: f64,
    pub arity: usize,
}

/// The one comparison both fences share (monomorphic, both arms driven by named tests — HR5).
fn first_climb_over(
    climbs: &[VisibilityClimb],
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    match climbs.iter().find(|c| c.levels > arity) {
        Some(c) => Err(VisibilityClimbExceeded {
            body: c.body,
            top: c.top,
            levels: c.levels,
            slack_m: c.slack_m,
            arity,
        }),
        None => Ok(()),
    }
}

/// THE BOOT FENCE (look_horizon.md §3.3.4 instrument 1, wired into EVERY world-deriving
/// process's boot — the shard AND the gateway): the generated world's required climb must not
/// exceed the look carrier's arity. A refusal is a measurement; a wrong pixel is not.
///
/// # Errors
/// [`VisibilityClimbExceeded`] naming the first offending body with its numbers.
pub fn guard_visibility_climb_bounded(
    seed_universe: u64,
    config: &UniverseConfig,
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    first_climb_over(&measure_visibility_climb(seed_universe, config), arity)
}

/// A candidate player-built region put to the build-admission fence: a STATIC body (SL4 — a
/// built structure does not orbit) of `shape` at `offset_m` in `parent`'s frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateRegion {
    pub realm: RealmId,
    pub parent: RealmId,
    pub shape: Boundary,
    pub offset_m: DVec3,
}

/// THE BUILD-ADMISSION FENCE (look_horizon.md §3.3.4 instrument 2 — D-LOOK-1): a candidate
/// placement whose required climb exceeds the carrier's arity is REFUSED — the PLACEMENT, never
/// the boot. The candidate joins the world's own generated forest (SL5: THE world, no variant)
/// and is measured by the same walk, the same formula, the same worst-case convention. At
/// today's interim scale a ~20 m surface structure measures a climb of 3 and refuses — the Q3
/// evidence, produced by a test rather than an argument; the near-real-scale re-solve is the
/// scheduled cure (owner ruling 2026-08-17).
///
/// # Errors
/// [`VisibilityClimbExceeded`] naming the candidate with its numbers.
pub fn guard_candidate_climb_bounded(
    candidate: &CandidateRegion,
    seed_universe: u64,
    config: &UniverseConfig,
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    let mut bodies = generate_system_forest(seed_universe, config);
    bodies.push(GeneratedBody {
        realm: candidate.realm,
        parent: Some(candidate.parent),
        shape: candidate.shape,
        placement: Placement::StaticOffset(candidate.offset_m),
        photometrics: None,
        // A built structure draws itself at its own bound (bound == look at build scale).
        taxon: None,
        look: Some(candidate.shape),
    });
    let climbs = visibility_climbs(&bodies, VISIBILITY_THETA_MIN_RAD, config.planet.ecc_cap);
    first_climb_over(
        &climbs
            .into_iter()
            .filter(|c| c.body == candidate.realm)
            .collect::<Vec<_>>(),
        arity,
    )
}

/// The config-driven star-system forest: Universe → Galaxy → `stellar.n_systems` star systems, each with
/// `planet.n_planets` `Orbital` planets (D-45(a) FA-5). A System shell IS its star's frame — the planets
/// orbit its center and the star is DATA (`stellar.central_mass_kg`), never a `RealmId::Star` (HR3).
///
/// EVERY system is built by the SAME loop from its own seed — there is no "system A" special case beyond
/// which seed index 0 carries. That is what makes a galaxy expressible at all: the previous shape named
/// one system in code and hung the planets off a constant, so a second star could not exist at any scale,
/// which is why the login side and the shard side ended up with two different worlds.
///
/// Pure `f(seed_universe, config)`: each system draws its planets from its OWN [`realm_stream`], keyed on
/// its own lineage, in a fixed order — so a shard hosting system 4 generates byte-identical planets to
/// every other shard's view of system 4, without any shared state (HR1). `n_planets == 0` emits no
/// planet; `n_systems == 0` emits the ambient forest alone.
fn generate_system_forest(seed_universe: u64, config: &UniverseConfig) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let st = &config.stellar;
    let pl = &config.planet;
    let shell = |r: f64| Boundary::Shell { r };
    let origin = Placement::StaticOffset(DVec3::ZERO);
    let mut bodies = vec![
        // Universe: the ambient ROOT (parent None) — the container-fold identity. NEVER drawn:
        // `look = None` (the two-body law — a containment boundary is structurally undrawable).
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe. NEVER drawn.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
    ];
    // HOW MANY STARS THIS GALAXY HOLDS — drawn from the galaxy's OWN stream against its census, so two
    // galaxies in one universe differ without anyone choosing. `lo == hi` pins it exactly.
    // (The star COUNT stays this existing config parameter — owner ruling Q-B: the placement law
    // went 3-D and seeded; the census census-derivation is P10's.)
    let n_systems = galaxy_system_count(seed_universe, config);
    for s in 0..n_systems {
        let seed = system_seed_at(s);
        let system = RealmId::System(seed);
        let system_ix = bodies.len();
        bodies.push(GeneratedBody {
            realm: system,
            parent: Some(GALAXY),
            // Filled below once the star is drawn: the shell is SOLVED (the clearance law) and
            // the look is the star's photosphere.
            shape: shell(0.0),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
            taxon: None,
            look: None,
        });
        // ============ THE PER-SYSTEM DRAW STREAM — the Stream Law (append-only) ============
        // FROZEN PREFIX (byte-identical since the interim world; the pins measure it):
        //   draws 1–25   five element draws for each of the first LEGACY_STREAM_PLANETS planets
        //   draw  26     the star's photometric mass (the pinned triples)
        //   draws 27–31  one geometric albedo per legacy planet
        //   draws 32–33  the two 3-D placement draws (owner ruling Q-B)
        // APPENDED by the taxonomy arc's in-system re-solve (new consumptions AFTER the frozen
        // prefix — planets 5–8 and every mass are new draws, never insertions):
        //   draws 34–53  five element draws for each planet beyond the legacy prefix
        //   draws 54–57  one geometric albedo per planet beyond the legacy prefix
        //   draws 58–66  one log-uniform mass per planet (ALL planets, index order)
        // Every `sma` and every `central_mass` is a DERIVATION (zero draws), so anchoring the
        // ladder to the star's √L consumes nothing and moves nothing.
        let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
        let legacy = pl.n_planets.min(LEGACY_STREAM_PLANETS);
        let mut element_draws: Vec<PlanetElementDraws> = (0..legacy)
            .map(|_| planet_element_draws(config, &mut stream))
            .collect();
        let star = draw_star_photometrics(st, &mut stream);
        bodies[system_ix].photometrics = Some(star);
        let mut albedo_draws: Vec<f64> = (0..legacy).map(|_| stream.next_f64()).collect();
        // THE 3-D PLACEMENT DRAWS (owner ruling Q-B, 2026-08-18) — the HOME system (index 0)
        // consumes its two draws exactly like every sibling (the anchor multiplier, not the
        // stream shape, pins it to the galactic origin).
        bodies[system_ix].placement = Placement::StaticOffset(system_center_at(
            config,
            s,
            stream.next_f64(),
            stream.next_f64(),
        ));
        // ---- the APPENDED draws (planets beyond the frozen prefix, then every mass) ----
        for _ in legacy..pl.n_planets {
            element_draws.push(planet_element_draws(config, &mut stream));
        }
        for _ in legacy..pl.n_planets {
            albedo_draws.push(stream.next_f64());
        }
        let mass_hi_mearth = planet_mass_hi_mearth(pl, &star);
        let masses_mearth: Vec<f64> = (0..pl.n_planets)
            .map(|_| {
                sample_imf_mass(
                    stream.next_f64(),
                    PLANET_MASS_SLOPE,
                    pl.mass_lo_mearth,
                    mass_hi_mearth,
                )
            })
            .collect();
        // ============ THE DERIVATIONS (consume nothing; true-size in-system space) ============
        let star_look_r_m = crate::taxonomy::star_radius_m(star.mass_msun);
        let frost_line_au =
            crate::taxonomy::frost_line_radius_au(star.luma_lsun, pl.frost_coeff_au);
        let th = pl.frost_thresholds();
        let mut planet_bodies = Vec::with_capacity(element_draws.len());
        let mut moon_inputs: Vec<(RealmId, u64, f64, f64, f64)> = Vec::new();
        for (n, (draws, mass_mearth)) in element_draws.iter().zip(&masses_mearth).enumerate() {
            let planet_seed = child_seed(seed, PLANET_SALT, n as u64);
            let elements = planet_orbital_elements(config, *draws, n as u32, &star);
            // THE PER-PLANET STREAM (T1 — a brand-new lineage nobody else reads; the Stream
            // Law's standing doctrine): P.1 = `f_env`, log-uniform over `F_ENV_BOUNDS` through
            // the existing sample_imf_mass log-uniform limit. Drawn UNCONDITIONALLY (the Stream
            // Law corollary: a draw is never conditional on a derived value); whether the
            // classifier READS it is downstream.
            let mut planet_stream = realm_stream(
                seed_universe,
                &[UNIVERSE_SEED, GALAXY_SEED, seed, planet_seed],
            );
            let f_env = sample_imf_mass(
                planet_stream.next_f64(),
                PLANET_MASS_SLOPE,
                crate::taxonomy::F_ENV_BOUNDS.0,
                crate::taxonomy::F_ENV_BOUNDS.1,
            );
            // THE ONE DERIVATION PATH (par 4.4): identical code for a planet under a star and
            // (T3) a moon under a planet.
            let taxon = crate::taxonomy::body_params(
                mass_mearth * crate::taxonomy::M_EARTH_KG,
                star.luma_lsun,
                star.class,
                elements.sma,
                frost_line_au,
                th,
                f_env,
            );
            let soi_m = planet_bound_m(pl, &star, n as u32, *mass_mearth);
            moon_inputs.push((
                RealmId::Planet(planet_seed),
                planet_seed,
                elements.sma,
                taxon.mass_kg,
                albedo_draws[n],
            ));
            planet_bodies.push(GeneratedBody {
                realm: RealmId::Planet(planet_seed),
                parent: Some(system),
                // D-REAL-1 (owner ruling 2026-08-18): the realm shell IS the gravitational
                // sphere of influence at the DRAWN mass (clamped by half the worst-instant
                // inter-orbit gap — an inert-but-live arm, fenced).
                shape: shell(soi_m),
                placement: Placement::Orbital(elements),
                // A planet's marker is REFLECTED LIGHT: the star's luma diluted over its orbit,
                // intercepted by its own DERIVED cross-section (its composition radius — its
                // LOOK, not its authority bound).
                photometrics: Some(reflected_photometrics(
                    &star,
                    albedo_draws[n],
                    taxon.radius_m,
                    elements.sma,
                )),
                taxon: Some(taxon),
                // SL3: the planet draws ITSELF at its own derived radius.
                look: Some(shell(taxon.radius_m)),
            });
        }
        // THE SYSTEM SHELL — the one clearance solve (real-scale design §3.2), bounded at the
        // MASS CAP (the worst lawful draw), so the shell is `f(star)` alone and no mass draw
        // can move it: every lawful planet fits by construction.
        bodies[system_ix].shape = shell(system_shell_r_m(config, &star));
        // THE SYSTEM'S LOOK IS ITS STAR: `star_radius_m` of the drawn mass — ONE function, TWO
        // call sites (celestial_taxonomy_design §5.2): the system's marker-side look here and
        // the Star realm's own look below carry the SAME value, so the marker→body handover is
        // the same radius at every depth. No double-draw: "a realm containing the eye is not a
        // subject" makes the two mutually exclusive.
        bodies[system_ix].look = Some(shell(star_look_r_m));
        bodies.extend(planet_bodies);
        // ★ THE STAR REALM (taxonomy arc T2, owner ruling 2026-08-19): the star becomes a
        // BODY-BEARING CHILD of its system — a separate authority volume whose crossing turns
        // on near-star physics. ZERO draws (its photometrics ARE the system's pinned draw);
        // pushed AFTER the system's planets (the §7.1 append-only forest order). Its BOUND is
        // the dust-sublimation radius (ruling E: the honest "solid matter is destroyed by heat
        // here" surface — `flux_radius_m` at 1500 K, the same equilibrium law as every planet
        // temperature, inverted); its LOOK is its own photosphere. The owner's Dyson-standoff
        // sub-item did NOT close honestly and is STOPPED per the ruling's own instruction —
        // see DEFERRED.md D-TAX-3 for the measured option set.
        bodies.push(GeneratedBody {
            realm: RealmId::Star(child_seed(seed, STAR_SALT, 0)),
            parent: Some(system),
            shape: shell(star_bound_m(&star)),
            placement: Placement::StaticOffset(DVec3::ZERO),
            // The star's marker IS its photometrics — the same kind-blind row the system
            // carries (a star's point of light is its own).
            photometrics: Some(star),
            taxon: None,
            look: Some(shell(star_look_r_m)),
        });
        // ★ THE MOON PASS (taxonomy arc T3, owner ruling: "for moons — it's the same as
        // Planet — just another realm"): moons are `RealmId::Planet` bodies whose parent is a
        // planet — zero new realm kinds, zero wire, the identical crossing/demand/draw code.
        // Pushed AFTER the star, planet order then rung order (the §7.1 append-only forest
        // order); every moon draws from its OWN per-moon lineage (the Stream Law one level
        // down: a count correction adds or removes moons without shifting a sibling's draws).
        for (planet_realm, planet_seed, planet_sma_m, planet_mass_kg, planet_albedo_u01) in
            moon_inputs
        {
            append_moons(
                &mut bodies,
                config,
                seed_universe,
                seed,
                &star,
                planet_realm,
                planet_seed,
                planet_sma_m,
                planet_mass_kg,
                planet_albedo_u01,
            );
        }
    }
    // The fixture plant (look_horizon slice 5 — G-IDENTICAL), appended LAST: with `None` (every
    // shipped constructor) this is a no-op and the forest is byte-identical to the pre-plant world.
    append_fixture_plant(&mut bodies, config);
    bodies
}

/// The planet mass draw's upper bound, Earth masses: `min(mass_cap, disc_fraction·M★/N)` —
/// the disc arm binds on every M dwarf (34.4/40.0/59.9 M⊕ for THE world's three stars), which
/// is observationally correct (M dwarfs do not host Jupiters). Monomorphic straight-line.
fn planet_mass_hi_mearth(pl: &PlanetConfig, star: &StarPhotometrics) -> f64 {
    let disc_budget_mearth = pl.disc_mass_fraction * star.mass_msun * crate::taxonomy::M_SUN_KG
        / crate::taxonomy::M_EARTH_KG
        / f64::from(pl.n_planets.max(1));
    pl.mass_cap_mearth.min(disc_budget_mearth)
}

/// One planet's BOUND: its gravitational sphere of influence at its drawn mass (D-REAL-1),
/// clamped by half the worst-instant gap to its neighbouring rungs — `.min` is a branchless
/// clamp; MEASURED never to bind on any lawful draw (`soi/a = (m/M★)^0.4 ≤ 0.0659` against a
/// worst-instant half-gap of `0.188·a`), kept as an inert-but-live arm with a fence rather than
/// deleted (real-scale design §3.3.4).
fn planet_bound_m(pl: &PlanetConfig, star: &StarPhotometrics, n: u32, mass_mearth: f64) -> f64 {
    let a0_au = pl.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    let a_n = orbital_axis_au(n, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
    let soi = crate::celestial::planet_soi(
        a_n,
        mass_mearth * crate::taxonomy::M_EARTH_KG,
        star.mass_msun * crate::taxonomy::M_SUN_KG,
    );
    // Half the WORST-INSTANT gap to each REAL neighbouring rung (both rungs at the eccentricity
    // cap) — provable non-overlap at every instant for every seed (§3.3.4; the interim model
    // measured its own overlap at the cap, invisible to the static sibling fence). The first
    // rung has no inner neighbour and the last no outer one — no phantom clamp (both arms
    // driven by the rung sweep).
    let gap_out = if n + 1 < pl.n_planets {
        let a_next = orbital_axis_au(n + 1, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
        0.5 * (a_next * (1.0 - pl.ecc_cap) - a_n * (1.0 + pl.ecc_cap))
    } else {
        f64::INFINITY
    };
    let gap_in = if n > 0 {
        let a_prev = orbital_axis_au(n - 1, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
        0.5 * (a_n * (1.0 - pl.ecc_cap) - a_prev * (1.0 + pl.ecc_cap))
    } else {
        f64::INFINITY
    };
    soi.min(gap_out.min(gap_in))
}

/// THE STAR REALM'S BOUND (taxonomy arc §5.3, owner ruling E): the DUST-SUBLIMATION radius —
/// the distance at which silicate dust reaches [`crate::taxonomy::DUST_SUBLIMATION_K`]
/// (Dullemond & Monnier 2010: the inner rim of every observed protoplanetary disc). ONE law,
/// two consumers, two depths: this is [`crate::taxonomy::flux_radius_m`] — the exact inverse of
/// every planet's equilibrium temperature — at `A = 0`, `T = T_sub`. Scale-free identity:
/// `bound / a₀ = 0.0861` for every star at every seed (both scale with √L) — pinned by the T2
/// gates. NO sibling clamp ships (the identity makes one unable to ever bind — an arm HR5
/// would need a synthetic the world cannot produce); only the photosphere fence below.
fn star_bound_m(star: &StarPhotometrics) -> f64 {
    crate::taxonomy::flux_radius_m(
        star.luma_lsun * crate::taxonomy::L_SUN_W,
        crate::taxonomy::DUST_SUBLIMATION_K,
        0.0,
    )
}

/// A star whose dust-sublimation bound fails to clear its own photosphere — the T2 boot fence's
/// loud refusal (`bound/R★ = 0.5·(T_eff/T_sub)²` → 1.993 at the hydrogen-burning limit, so the
/// fence holds across the whole IMF domain but is a FENCE, not an assumption — the swept gate
/// prints the minimum).
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "the star of {system:?} has a dust-sublimation bound ({bound_m} m) at or inside its own \
     photosphere ({photosphere_m} m) — the Star realm would be drawn wider than its authority; \
     the mass-radius or luminosity law changed under the extent derivation"
)]
pub struct StarBoundInsidePhotosphere {
    pub system: RealmId,
    pub bound_m: f64,
    pub photosphere_m: f64,
}

/// THE T2 BOOT FENCE (`guard_star_bound_exceeds_photosphere`): every generated star's bound
/// strictly exceeds its photosphere. Wired beside the climb fence in every world-deriving boot.
///
/// # Errors
/// [`StarBoundInsidePhotosphere`] naming the first offending system with both radii.
pub fn guard_star_bound_exceeds_photosphere(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<(), StarBoundInsidePhotosphere> {
    guard_star_bounds(&generate_system_forest(seed_universe, config))
}

/// The fence's ARITHMETIC over a forest already in hand — split from the generate-and-check shell
/// so the REFUSAL arm is reachable from a unit test (HR5: the branching lives in a monomorphic
/// helper a test can hand a hostile forest; THE world only ever produces the green arm).
fn guard_star_bounds(bodies: &[GeneratedBody]) -> Result<(), StarBoundInsidePhotosphere> {
    for b in bodies {
        if let (RealmId::Star(_), Some(p)) = (b.realm, b.photometrics.as_ref()) {
            let bound_m = b.shape.finite_extent();
            let photosphere_m = crate::taxonomy::star_radius_m(p.mass_msun);
            if bound_m <= photosphere_m {
                return Err(StarBoundInsidePhotosphere {
                    system: b.parent.expect("a star nests in its system"),
                    bound_m,
                    photosphere_m,
                });
            }
        }
    }
    Ok(())
}

/// ★ T3 — ONE planet's MOON PASS (celestial_taxonomy_design §4): the count law (density-
/// corrected Roche inner edge, the `ratio^(k+1)` offset ladder, the Machida `R_Hill/48` disc
/// edge), the Σ ∝ 1/a gas-starved mass partition with the per-moon log-uniform multiplier,
/// the potato-radius emission floor, and the climb-1 clearance clamp as the disc edge's
/// THIRD `child_clearance_m` consumer (live, measured never to bind). A moon IS a
/// `RealmId::Planet` under a planet: its bound is its OWN gravitational SOI (Hill-class,
/// the same `celestial::planet_soi`, different arguments), its look its own derived radius,
/// its taxon the ONE `body_params` path with the illuminator at its PARENT's orbit.
#[allow(clippy::too_many_arguments)] // one private pass; every argument is the parent's own datum
fn append_moons(
    bodies: &mut Vec<GeneratedBody>,
    config: &UniverseConfig,
    seed_universe: u64,
    system_seed: u64,
    star: &StarPhotometrics,
    planet_realm: RealmId,
    planet_seed: u64,
    planet_sma_m: f64,
    planet_mass_kg: f64,
    planet_albedo_u01: f64,
) -> u32 {
    use crate::taxonomy::{
        MOON_ECC_SIGMA, MOON_INCL_SIGMA, MOON_MIN_RADIUS_M, RHO_ICE_KGM3, RHO_ROCK_KGM3,
        SATELLITE_DISC_HILL_FRACTION, SATELLITE_MASS_FRACTION, hill_radius_m, roche_radius_m,
        satellite_disc_edge_m, satellite_ladder_a_m,
    };
    let pl = &config.planet;
    let frost_line_au = crate::taxonomy::frost_line_radius_au(star.luma_lsun, pl.frost_coeff_au);
    let inside_frost = planet_sma_m / crate::taxonomy::AU_M < frost_line_au;
    // Moon composition density: ice beyond the frost line, rock inside (the satellite census's
    // own calibration: Ganymede/Callisto/Titan ~1900 vs the Moon 3344).
    let rho_moon = if inside_frost {
        RHO_ROCK_KGM3
    } else {
        RHO_ICE_KGM3
    };
    let roche_m = roche_radius_m(planet_mass_kg, rho_moon);
    let star_mass_kg = star.mass_msun * crate::taxonomy::M_SUN_KG;
    let disc_edge_m = satellite_disc_edge_m(
        hill_radius_m(planet_sma_m, planet_mass_kg, star_mass_kg),
        SATELLITE_DISC_HILL_FRACTION,
    );
    let planet_soi_m = bodies
        .iter()
        .find(|b| b.realm == planet_realm)
        .expect("the moon pass runs after its planet is pushed")
        .shape
        .finite_extent();
    // THE COUNT: ladder rungs inside the disc edge — and inside the climb-1 clearance clamp
    // (the §4.5.1 third `child_clearance_m` arm: a rung whose worst instant + clearance would
    // breach the planet's SOI is not minted; measured 25×-slack inert on THE world, fenced by
    // the census pin printing the counts).
    let budget_kg = SATELLITE_MASS_FRACTION * planet_mass_kg;
    let worst_moon_mass_kg = MOON_MASS_MULTIPLIER_BOUNDS.1 * budget_kg;
    let mut rungs_m: Vec<f64> = Vec::new();
    for k in 0..64u32 {
        let a_k = satellite_ladder_a_m(k, roche_m, pl.orbital_ratio);
        if a_k > disc_edge_m {
            break;
        }
        let worst_bound_m = crate::celestial::planet_soi(a_k, worst_moon_mass_kg, planet_mass_kg);
        let worst_look_m = crate::taxonomy::composition_radius_m(
            crate::taxonomy::PlanetType::Rocky,
            inside_frost,
            worst_moon_mass_kg / crate::taxonomy::M_EARTH_KG,
            0.0,
            star.luma_lsun / (planet_sma_m / crate::taxonomy::AU_M).powi(2),
        );
        let ecc_cap = crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS;
        if a_k * (1.0 + ecc_cap)
            + child_clearance_m(worst_bound_m, worst_look_m, VISIBILITY_THETA_MIN_RAD)
            > planet_soi_m
        {
            break;
        }
        rungs_m.push(a_k);
    }
    let share_denominator: f64 = rungs_m.iter().sum();
    let th = pl.frost_thresholds();
    let mut emitted = 0u32;
    for (k, a_k) in rungs_m.iter().enumerate() {
        let moon_seed = child_seed(planet_seed, MOON_SALT, k as u64);
        // THE PER-MOON STREAM (M.1..M.6, frozen order): mass multiplier, ecc, incl, Ω, ω, M₀
        // — keyed on the MOON's own lineage, so a later count correction never shifts a
        // surviving sibling's draws (the Stream Law's own reason).
        let mut moon_stream = realm_stream(
            seed_universe,
            &[
                UNIVERSE_SEED,
                GALAXY_SEED,
                system_seed,
                planet_seed,
                moon_seed,
            ],
        );
        let multiplier = sample_imf_mass(
            moon_stream.next_f64(),
            PLANET_MASS_SLOPE,
            MOON_MASS_MULTIPLIER_BOUNDS.0,
            MOON_MASS_MULTIPLIER_BOUNDS.1,
        );
        let ecc = sample_rayleigh(moon_stream.next_f64(), MOON_ECC_SIGMA)
            .min(MOON_ECC_SIGMA * ECC_CAP_SIGMAS);
        let inclination = sample_rayleigh(moon_stream.next_f64(), MOON_INCL_SIGMA);
        let raan = moon_stream.next_f64() * TAU;
        let arg_periapsis = moon_stream.next_f64() * TAU;
        let mean_anomaly_epoch = moon_stream.next_f64() * TAU;
        // Σ ∝ 1/a partition: annulus mass ∝ a on a geometric ladder — masses increase
        // outward and the outermost dominates, the two observed facts the law reproduces.
        let mass_kg = budget_kg * (a_k / share_denominator) * multiplier;
        // THE ONE DERIVATION PATH (§4.4): the identical `body_params` a planet takes, with
        // the illuminator distance at the PARENT's orbit. `f_env = 0`: the per-moon stream
        // draws none (M.1..M.6), and no lawful moon mass can classify envelope-retaining
        // (the whole domain sits far under every rung's stripped ceiling).
        let taxon = crate::taxonomy::body_params(
            mass_kg,
            star.luma_lsun,
            star.class,
            planet_sma_m,
            frost_line_au,
            th,
            0.0,
        );
        // THE POTATO FLOOR (Lineweaver & Norman 2010): below hydrostatic equilibrium no
        // realm is emitted — rubble is a later debris slice. The DRAWS above happened
        // unconditionally (the Stream Law corollary); only the emission is gated.
        if taxon.radius_m < MOON_MIN_RADIUS_M {
            continue;
        }
        emitted += 1;
        bodies.push(GeneratedBody {
            // A MOON IS A PLANET (the ruling, literally): same kind, parent = a planet.
            realm: RealmId::Planet(moon_seed),
            parent: Some(planet_realm),
            // Its bound is its OWN gravitational SOI (Hill-class — the same one function).
            shape: Boundary::Shell {
                r: crate::celestial::planet_soi(*a_k, mass_kg, planet_mass_kg),
            },
            placement: Placement::Orbital(OrbitalElements {
                sma: *a_k,
                ecc,
                inclination,
                raan,
                arg_periapsis,
                mean_anomaly_epoch,
                central_mass: planet_mass_kg,
            }),
            // A moon's marker IS reflected light — the one existing function, the star as
            // illuminator, the PLANET's orbit as the dilution distance. Its albedo reuses its
            // parent planet's drawn u01 (per-moon albedo variety is the design's own named
            // deferral — a future M.7 append, never a re-roll).
            photometrics: Some(reflected_photometrics(
                star,
                planet_albedo_u01,
                taxon.radius_m,
                planet_sma_m,
            )),
            taxon: Some(taxon),
            look: Some(Boundary::Shell { r: taxon.radius_m }),
        });
    }
    emitted
}

/// THE SYSTEM SHELL SOLVE (real-scale design §3.2): `max` over the ladder rungs of
/// `worst_excursion + child_clearance_m(bound_at_cap, look_at_cap)` — each rung bounded at the
/// system's own MASS CAP, the worst lawful draw, so the shell is a pure function of the star
/// and no planet draw can move it. The §3.2 identity then gives every child a stopping slack
/// equal to exactly the clearance the solve reserved — the structural climb-1 proof.
fn system_shell_r_m(config: &UniverseConfig, star: &StarPhotometrics) -> f64 {
    let pl = &config.planet;
    let cap_mearth = planet_mass_hi_mearth(pl, star);
    let a0_au = pl.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    // The STAR child's own clearance arm (T2): zero excursion + the clearance its bound/look
    // owe. LIVE, and MEASURED never to bind (§5.3.2 Prediction A: the outer planet's term wins
    // by ~13× — `g_star_shell_unmoved` proves the shells byte-identical with the arm in).
    let star_term = child_clearance_m(
        star_bound_m(star),
        crate::taxonomy::star_radius_m(star.mass_msun),
        VISIBILITY_THETA_MIN_RAD,
    );
    (0..pl.n_planets)
        .map(|n| {
            let a_n = orbital_axis_au(n, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
            let insolation_rel = star.luma_lsun
                / (orbital_axis_au(n, a0_au, pl.orbital_ratio)
                    * orbital_axis_au(n, a0_au, pl.orbital_ratio));
            let bound_cap_m = planet_bound_m(pl, star, n, cap_mearth);
            let look_cap_m = worst_rung_look_m(pl, insolation_rel, cap_mearth);
            a_n * (1.0 + pl.ecc_cap)
                + child_clearance_m(bound_cap_m, look_cap_m, VISIBILITY_THETA_MIN_RAD)
        })
        .fold(star_term, f64::max)
}

/// The LARGEST look any lawful draw can put on a rung (the shell solve's per-rung bound): the
/// max over (a) the Chen–Kipping radius at the mass cap (the giant/population envelope), (b)
/// the ice core at the cap, and (c) the retained composition radius at both ENDPOINTS of the
/// retained-mass interval — endpoints suffice because `core + envelope` is `a·m^0.27 +
/// b·m^-0.21`, whose single interior critical point is a MINIMUM (the derivative goes negative
/// → positive), so the maximum sits at an end. Ice coefficient throughout (≥ rock). Monomorphic.
fn worst_rung_look_m(pl: &PlanetConfig, insolation_rel: f64, cap_mearth: f64) -> f64 {
    use crate::taxonomy::{
        F_ENV_BOUNDS, ICE_MR_SEGMENTS, PMR_SEGMENTS, R_EARTH_M, RADIUS_VALLEY_1SEARTH_REARTH,
        RADIUS_VALLEY_INSOLATION_EXP, SYSTEM_AGE_GYR, envelope_radius_rearth, radius_valley_rearth,
        segmented_power_law,
    };
    let ice_core = |m: f64| segmented_power_law(m, &ICE_MR_SEGMENTS);
    let retained = |m: f64| {
        ice_core(m) + envelope_radius_rearth(m, F_ENV_BOUNDS.1, insolation_rel, SYSTEM_AGE_GYR)
    };
    // The smallest mass the classifier can call retained at this insolation: the rock core
    // crosses the valley (`m^0.27 = valley` — the par 3.4.5 stripped ceiling), floored at the
    // draw's own lower bound.
    let valley = radius_valley_rearth(
        insolation_rel,
        RADIUS_VALLEY_1SEARTH_REARTH,
        RADIUS_VALLEY_INSOLATION_EXP,
    );
    let m_retained_lo = valley
        .powf(1.0 / ICE_MR_SEGMENTS[0].2)
        .clamp(pl.mass_lo_mearth, cap_mearth);
    let candidates = [
        segmented_power_law(cap_mearth, &PMR_SEGMENTS),
        ice_core(cap_mearth),
        retained(m_retained_lo),
        retained(cap_mearth),
    ];
    R_EARTH_M * candidates.iter().fold(0.0_f64, |a, &b| a.max(b))
}

/// Every system's drawn [`StarPhotometrics`] over the config-driven forest — `(realm, draw)`
/// pairs, forest order; the config twin of [`moving_children_for_config`] (the SAME
/// `(seed, config)` builds the SAME forest, so the marker roster and the regions can never
/// disagree). The Slice-A marker emit reads THIS; Slice 0 lands it consumer-less (nothing moves),
/// pinned by the THE-world goldens below. The closure is a branchless shim (HR5): the
/// `Some`/`None` split lives in `Option::map`'s monomorphic body over `photometrics`.
#[must_use]
pub fn system_photometrics_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<(RealmId, StarPhotometrics)> {
    generate_system_forest(seed_universe, config)
        .iter()
        .filter_map(|b| b.photometrics.map(|p| (b.realm, p)))
        .collect()
}

/// One sleeping child's marker DATUM for the window lane (Slice A → look_horizon.md slice 1):
/// the spectral class as its stable code (color) plus the main-sequence luminosity (brightness)
/// — exactly the two scalars the owner-ruled R4 datum names. The class→code cast lives HERE and
/// nowhere else. The boot plumbs these datums onto the shard's roster (`vd-sim` `ChildLuma`);
/// the BAG is framed per child by the sim through the one `vd_core::look::marker_bag` codec,
/// which appends the child's circumscribed extent (the presence floor's one radius).
#[must_use]
pub fn marker_datum(p: &StarPhotometrics) -> (u8, f64) {
    (p.class as u8, p.luma_lsun)
}

/// [`to_regions`] over the config-driven system forest — the config-parameterised twin of
/// [`realm_regions_for`] (S2 wraps it with [`UniverseConfig::visual_scale`]). Reuses `to_regions` verbatim.
#[must_use]
pub fn realm_regions_for_config(seed_universe: u64, config: &UniverseConfig) -> Vec<RealmRegion> {
    to_regions(&generate_system_forest(seed_universe, config), config)
}

/// [`moving_children`] over the config-driven system forest — the config twin of [`moving_children_for`].
/// The SAME `(seed, config)` builds the SAME forest as [`realm_regions_for_config`], so the authored
/// moving roster and the regions can NEVER disagree (all-shard-seams-same-config). Reuses `moving_children`.
#[must_use]
pub fn moving_children_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
    hosted: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    moving_children(&generate_system_forest(seed_universe, config), hosted)
}

// ===== THE FIXTURE PLANT (look_horizon.md slice 5 — G-IDENTICAL / SL5 fixture-forest doctrine) ==
// Stations and areas are built by PLAYERS; no seed emits one. The pixel gate that proves the look
// horizon is kind-blind (HR4) therefore needs player-built content, and SL5's landed ruling is
// that fixtures may plant player-built regions on THE world. There was no process-tier plant path
// before this (the walk fixture forest is a SEPARATE hand-placed world, retired from the boots),
// so this is the MINIMAL LAWFUL one: a named pair appended to the generated forest through the ONE
// generator/lowering, selected by `UniverseConfig::fixture_plant`, measured by the SAME boot
// fences (`guard_visibility_climb_bounded` at every process boot) as everything else. It is NOT
// build admission (the live build feature does not exist yet — D-LOOK-1): a fixture states the
// content, and the fences judge it exactly as they will judge a player's candidate.

/// WHICH player-built content a config plants beside the generated bodies. `None` (default) is
/// byte-identical to the pre-plant world; the ONE named plant today is the slice-5 G-IDENTICAL
/// pair. A NAMED enumeration, deliberately not a geometry parameter: a free-form plant input would
/// be a second world generator wearing a config field (SL5 forbids it), while a named fixture is
/// content with one derivation, shared by every process that boots it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixturePlant {
    /// Nothing built — THE world exactly as the seed generates it.
    #[default]
    None,
    /// The G-IDENTICAL pair (look_horizon.md slice 5): one player-built STATION under the home
    /// star system and one player-built AREA on that system's inner planet — see
    /// [`station_area_plant`] for every derived number.
    StationArea,
}

/// The G-IDENTICAL plant's derived spec — public so the pixel gate's ORACLE derives its parks and
/// expectations from the SAME numbers the boots plant, out-of-band (never by reading the drawn
/// scene back).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationAreaPlant {
    /// The planted station's realm id: `Station(child_seed(home system, FIXTURE_SALT, 0))`.
    pub station: RealmId,
    /// The station's parent — the HOME star system (the root's first grandchild, the same lineage
    /// rule `default_home_realm` applies to the lowered forest).
    pub station_parent: RealmId,
    /// The station's static offset in its parent's frame.
    pub station_offset_m: DVec3,
    /// The station's shell radius.
    pub station_extent_m: f64,
    /// The planted area's realm id: `Area(child_seed(inner planet, FIXTURE_SALT, 0))`.
    pub area: RealmId,
    /// The area's parent — the home system's INNER planet (smallest semi-major axis, the same
    /// rule the world roster's `inner` uses). The frame law (`frame_for_realm`) requires an Area's
    /// parent to be a PLANET, which is WHY the pair is planted in the walk-forest shape (station
    /// under the system, area under a planet) and not as parent/child of each other.
    pub area_parent: RealmId,
    /// The area's static offset in the PLANET's frame.
    pub area_offset_m: DVec3,
    /// The area's shell radius.
    pub area_extent_m: f64,
}

/// The G-IDENTICAL plant, derived from a generated forest + its config. Every number is an
/// expression over THE world's own values, with its constraint stated (and pinned by this crate's
/// units — a plant that broke one would fail the boot fence loudly, not draw wrongly):
///
/// - **station extent** = the planet SOI radius (`planet.planet_soi_r_m`): planet-extent CLASS, so
///   every visibility bound the world already proves for a planet (reach 302.06 m, climb stops at
///   the galaxy) holds for the station verbatim, and the home system's interior band stays
///   PLANET-dominated (444.104489631 m — the station's `75 + 302.06 = 377.06 m` term is smaller).
/// - **station offset** = half the system shell radius up the `(1, 0, 2)/√5` tilted-polar
///   direction: `|offset| = 75 m` nests with a whole planet-orbit annulus of margin
///   (`75 + 3.954 < 150`); the `z = 67.1 m` component stands clear of the orbital plane (worst
///   planet `|z|` is apoapsis · sin(inclination), measured tiny against it in the units); the
///   `x = 33.5 m` component stands clear of BOTH ±Z polar flight axes (the licensed exit corridor
///   and the gate's own park legs) by far more than its extent.
/// - **area parent** = the INNER planet, forced by the physics, not chosen: under the OUTER planet
///   the area's worst-instant excursion (142.05 m at the eccentricity cap) leaves less system
///   slack than its own visibility reach, so its climb would be 3 and every boot would refuse
///   (the Q3 posture). Under the inner planet (excursion 16.07 m) the climb stops at the system:
///   levels 2, exactly what the carrier serves.
/// - **area extent** = a quarter of the planet SOI (`0.9885 m`): visibility reach
///   `76.39 × 0.9885 = 75.51 m` — the planet's interior band it induces brackets the planet's own
///   3.954 m shell (the park band exists), while the system-level slack
///   `150 − (16.07 + 1.977) − 0.99 = 130.97 m` stays far above that reach (the climb stops).
/// - **area offset** = half the planet SOI up +Z in the planet's frame: nests at `3/4` of the
///   planet's inscribed extent, a quarter-extent of margin.
#[must_use]
pub fn station_area_plant(seed_universe: u64, config: &UniverseConfig) -> StationAreaPlant {
    let mut base = *config;
    base.fixture_plant = FixturePlant::None;
    station_area_plant_spec(&generate_system_forest(seed_universe, &base))
}

/// The spec over an already-generated (plant-free) forest — the one derivation both
/// [`station_area_plant`] and the generator's own append share.
fn station_area_plant_spec(bodies: &[GeneratedBody]) -> StationAreaPlant {
    let home = bodies
        .iter()
        .find(|b| b.parent == Some(GALAXY))
        .expect("THE world generates at least one star system")
        .realm;
    let inner = bodies
        .iter()
        .filter(|b| b.parent == Some(home))
        .filter_map(|b| orbital_of(b.placement).map(|e| (b.realm, e.sma)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("THE home system generates orbiting planets")
        .0;
    let tilt = DVec3::new(1.0, 0.0, 2.0).normalize();
    let home_seed = plant_seed_of(home).expect("the home system is seed-keyed");
    let inner_seed = plant_seed_of(inner).expect("a generated planet is seed-keyed");
    // The plant's OFFSETS are proportions of the bodies it hangs under, read from the forest
    // itself (the config carried no in-system radii after the re-solve — the shells are
    // solved); its EXTENTS are the owner's Q3 build cases (the 10 km city, the 20 m structure).
    let home_shell_m = bodies
        .iter()
        .find(|b| b.realm == home)
        .expect("the home system is in the forest it was found in")
        .shape
        .finite_extent();
    let inner_shell_m = bodies
        .iter()
        .find(|b| b.realm == inner)
        .expect("the inner planet is in the forest it was found in")
        .shape
        .finite_extent();
    StationAreaPlant {
        station: RealmId::Station(child_seed(home_seed, FIXTURE_SALT, 0)),
        station_parent: home,
        station_offset_m: tilt * (home_shell_m * 0.5),
        station_extent_m: FIXTURE_CITY_R_M,
        area: RealmId::Area(child_seed(inner_seed, FIXTURE_SALT, 0)),
        area_parent: inner,
        area_offset_m: DVec3::new(0.0, 0.0, inner_shell_m * 0.5),
        area_extent_m: FIXTURE_STRUCTURE_R_M,
    }
}

/// The u64 seed of a SEED-LINEAGE realm (a system or a planet — the only parents a plant hangs
/// under), `None` for the entity/plant-keyed kinds. Monomorphic; both arms driven by named units.
fn plant_seed_of(realm: RealmId) -> Option<u64> {
    match realm {
        RealmId::System(s) | RealmId::Planet(s) => Some(s),
        // No fixture plants on a star (taxonomy arc §6.2 site 4): nothing is built inside the
        // dust-sublimation radius.
        RealmId::Ship(_) | RealmId::Station(_) | RealmId::Area(_) | RealmId::Star(_) => None,
    }
}

/// Append the named plant's bodies to a generated forest — called at the END of
/// [`generate_system_forest`], AFTER every generated body and every stream draw, so the additive
/// discipline holds: with a plant present every generated body, id, orbit and photometric draw is
/// byte-identical to the plant-free world.
fn append_fixture_plant(bodies: &mut Vec<GeneratedBody>, config: &UniverseConfig) {
    match config.fixture_plant {
        FixturePlant::None => {}
        FixturePlant::StationArea => {
            let plant = station_area_plant_spec(bodies);
            bodies.push(GeneratedBody {
                realm: plant.station,
                parent: Some(plant.station_parent),
                shape: Boundary::Shell {
                    r: plant.station_extent_m,
                },
                placement: Placement::StaticOffset(plant.station_offset_m),
                // A player-built structure has no seed stream and no photometric draw — the
                // presence floor (look_horizon slice 1) states its point of light from its
                // extent alone.
                photometrics: None,
                taxon: None,
                look: Some(Boundary::Shell {
                    r: plant.station_extent_m,
                }),
            });
            bodies.push(GeneratedBody {
                realm: plant.area,
                parent: Some(plant.area_parent),
                shape: Boundary::Shell {
                    r: plant.area_extent_m,
                },
                placement: Placement::StaticOffset(plant.area_offset_m),
                photometrics: None,
                taxon: None,
                look: Some(Boundary::Shell {
                    r: plant.area_extent_m,
                }),
            });
        }
    }
}

// ===== T4 — THE EARTH-LIKE PREDICATE + THE SEED SEARCH (celestial_taxonomy_design §8) =======

/// One Earth-like candidate of a swept seed — everything the report prints, derived entirely
/// from `BodyTaxon` + `StarPhotometrics`, both of which the ONE generator emits (SL5: the tool
/// never builds a world; it READS THE world at a candidate seed and scores it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarthLikeCandidate {
    pub system: RealmId,
    pub body: RealmId,
    pub star_mass_msun: f64,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub insolation_rel: f64,
    pub t_eq_k: f64,
}

/// The owner's Earth-radius band (Earth radii) — the search's stated size criterion.
pub const EARTH_LIKE_RADIUS_BAND_REARTH: (f64, f64) = (0.8, 1.25);

/// THE EARTH-LIKE PREDICATE (§8.1, under the owner's rulings): a YELLOW SUN (G class), a
/// ROCKY verdict (now a derived composition + envelope conclusion), the owner's radius band,
/// the Kopparapu CONSERVATIVE flux band (ruling B), the temperate `T_eq` band derived from
/// those same flux limits at the Rocky-with-atmosphere Bond albedo (no second literal), and
/// AIR — the shoreline verdict survived.
///
/// TWO OF THE SIX CLAUSES ARE STRUCTURAL NO-OPS ON ANY WORLD (§8.2, stated so the tool cannot
/// report false precision): insolation is QUANTISED AND SEED-FREE (rung 2 = 0.748 S⊕ for
/// every star at every seed — the ladder law), and a rocky rung-2 `T_eq` is one of two
/// class-albedo values, both inside the band. The search's real discriminants are the star's
/// class and the planet's drawn mass. `earth_like_no_op_clauses…` pins this as a measurement.
#[must_use]
pub fn earth_like(star: &StarPhotometrics, taxon: &crate::taxonomy::BodyTaxon) -> bool {
    use crate::taxonomy::{KOPPARAPU_FLUX_CONSERVATIVE, PlanetType, R_EARTH_M, SpectralClass};
    let (s_lo, s_hi) = KOPPARAPU_FLUX_CONSERVATIVE;
    let (r_lo, r_hi) = EARTH_LIKE_RADIUS_BAND_REARTH;
    let yellow_sun = star.class == SpectralClass::G;
    let rocky = taxon.class == PlanetType::Rocky;
    let earth_sized = (r_lo..=r_hi).contains(&(taxon.radius_m / R_EARTH_M));
    let temperate_flux = (s_lo..=s_hi).contains(&taxon.insolation_rel);
    // T grows with flux: the band's LOW temperature is the OUTER flux limit's.
    let temperate_k =
        (earth_like_t_bound_k(s_lo)..=earth_like_t_bound_k(s_hi)).contains(&taxon.t_eq_k);
    let air = taxon.atmosphere.is_some();
    yellow_sun & rocky & earth_sized & temperate_flux & temperate_k & air
}

/// The temperate band's temperature at flux `s` — the SAME Kopparapu limits converted once
/// through the one equilibrium law at the Rocky-with-atmosphere Bond albedo (the class
/// table's own value; no second literal): `T(1.10) ≈ 260.2 K`, `T(0.53) ≈ 216.7 K`.
#[must_use]
pub fn earth_like_t_bound_k(s_rel: f64) -> f64 {
    use crate::taxonomy::{AU_M, L_SUN_W, PlanetType, bond_albedo_of, equilibrium_temperature_k};
    equilibrium_temperature_k(
        L_SUN_W * s_rel,
        AU_M,
        bond_albedo_of(PlanetType::Rocky, true),
    )
}

/// SWEEP one seed: every Earth-like body of THE world at that seed (SL5: the one generator,
/// read — never a variant). The tool's whole read path, shared by the bin and the tests.
#[must_use]
pub fn earth_like_candidates(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<EarthLikeCandidate> {
    earth_like_in_forest(&generate_system_forest(seed_universe, config))
}

/// The sweep's ARITHMETIC over a forest already in hand — split from the generate-and-score shell
/// so its two defensive arms (a body whose parent is not in the forest; a chain that names no
/// photometric star) are reachable from a unit test. THE world never produces either: every
/// generated body's parent is present and every system carries its star.
fn earth_like_in_forest(bodies: &[GeneratedBody]) -> Vec<EarthLikeCandidate> {
    let mut out = Vec::new();
    for b in bodies.iter().filter(|b| b.taxon.is_some()) {
        let taxon = b.taxon.expect("filtered on presence");
        let Some(parent) = b.parent else { continue };
        // The illuminating STAR: the body's parent system (a planet) or grandparent (a moon).
        let system = match parent {
            RealmId::System(_) => parent,
            _ => match bodies
                .iter()
                .find(|p| p.realm == parent)
                .and_then(|p| p.parent)
            {
                Some(gp) => gp,
                None => continue,
            },
        };
        let Some(star) = bodies
            .iter()
            .find(|p| p.realm == system)
            .and_then(|p| p.photometrics)
        else {
            continue;
        };
        if earth_like(&star, &taxon) {
            out.push(EarthLikeCandidate {
                system,
                body: b.realm,
                star_mass_msun: star.mass_msun,
                mass_kg: taxon.mass_kg,
                radius_m: taxon.radius_m,
                insolation_rel: taxon.insolation_rel,
                t_eq_k: taxon.t_eq_k,
            });
        }
    }
    out
}

/// A WORLD, materialised once: the bodies that exist, and the containment regions they lower to.
///
/// WHY THIS TYPE EXISTS. The world used to be re-derived from `(seed, config)` at every question — where a
/// login lands, which regions a shard evaluates, how a realm's origin folds. Three problems followed from
/// that, and this type is the answer to all three at once:
///
/// 1. **Different questions could get different worlds.** The gateway resolved a player's home against the
///    GENERATED world and then validated that answer against the HAND-PLACED one. With a single star the two
///    agreed by accident; with several stars a login beside any other star resolves to a realm the check has
///    never heard of, and the gateway panics on its own defence. Holding ONE world makes that class of bug
///    unstateable rather than fixed.
/// 2. **The world could not contain anything a seed does not produce.** Stations are built by players and
///    areas mostly are; no seed ever emits one, so anything needing them had to reach for a second world
///    builder — which is precisely the seam that let (1) happen. A world is now something you HOLD, so it
///    can be generated content, generated content plus what players have built, or (in a test) content
///    placed by hand. Same code path, different contents.
/// 3. **It regenerated the entire forest per call.** A login rebuilt every star and planet to answer one
///    question about one position.
///
/// HONEST SCOPE: this materialises the whole forest once. A universe too large to hold in memory needs the
/// per-subtree lazy generator that is the P4 owe; this type is where that laziness will live, and moving it
/// here first means no caller has to change when it lands.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldView {
    bodies: Vec<GeneratedBody>,
    regions: Vec<RealmRegion>,
}

impl WorldView {
    /// The world a SEED produces: stars and their planets, and nothing else. This is the production world —
    /// what a player logs into today, before anyone has built anything.
    #[must_use]
    pub fn generated(seed_universe: u64, config: &UniverseConfig) -> WorldView {
        WorldView::of(generate_system_forest(seed_universe, config), config)
    }

    /// A world with structures PLACED BY HAND — a station, an area, a planet at a known spot.
    ///
    /// TEST WORLD. The generator emits none of these on purpose: stations are built by players, and so are
    /// areas in all but a few cases, so a seed-derived world contains no station to stand in and no area to
    /// spin up. A test that needs one places it. This is the same shape player-built content will take when
    /// it lands — a world is bodies, and where they came from is not something anything downstream asks.
    ///
    /// Everything below the placement is IDENTICAL to the generated path: the same lowering, the same bands,
    /// the same descend. Nothing here is a second implementation of anything.
    #[must_use]
    pub fn hand_placed(config: &UniverseConfig) -> WorldView {
        WorldView::of(generate_walk_forest(config), config)
    }

    /// Lower a body list to its regions once, and keep both — the regions answer containment questions, the
    /// bodies answer placement ones (an orbit's elements do not survive lowering).
    fn of(bodies: Vec<GeneratedBody>, config: &UniverseConfig) -> WorldView {
        let regions = to_regions(&bodies, config);
        WorldView { bodies, regions }
    }

    /// The containment forest — what a shard evaluates membership against.
    #[must_use]
    pub fn regions(&self) -> &[RealmRegion] {
        &self.regions
    }

    /// The regions a shard holding `held` evaluates: ancestors ∪ direct children, never siblings.
    #[must_use]
    pub fn neighbourhood(&self, held: &std::collections::BTreeSet<RealmId>) -> Vec<RealmRegion> {
        neighbourhood_scope(&self.regions, held)
    }

    /// Is this realm part of this world? The honest form of the check that used to consult a DIFFERENT world
    /// than the one that produced the answer being checked.
    #[must_use]
    pub fn contains_realm(&self, realm: RealmId) -> bool {
        self.regions.iter().any(|r| r.realm == realm)
    }

    /// THE DEFAULT HOME STANDOFF (T2's forced re-derivation of the login pose): the home
    /// realm's own centre stopped being empty space the day the STAR became a body-bearing
    /// child there — a zero-offset spawn would put every new account inside the Star realm.
    /// The pose pushes out along +Z (the I-AXIS-clear polar axis) by TWICE the largest STATIC
    /// child region that CONTAINS the centre (the star: 2× its dust bound). Orbital children
    /// never contain the centre (periapsis − shell > 0, the sibling fences) and static
    /// children clear of the centre contribute nothing — so on a world whose origin IS empty
    /// (the walk fixture) this is ZERO, byte-identical to the pre-T2 spawn. Derived from the
    /// forest this view already holds, never a picked number.
    #[must_use]
    pub fn default_home_offset_m(&self) -> DVec3 {
        let Some(home) = vd_core::worldgen::default_home_realm(self.regions()) else {
            return DVec3::ZERO;
        };
        let clearing_z = self
            .bodies
            .iter()
            .filter(|b| b.parent == Some(home))
            .filter_map(|b| match b.placement {
                Placement::StaticOffset(at) => {
                    let bound = b.shape.finite_extent();
                    (at.length() < bound).then_some(2.0 * bound)
                }
                Placement::Orbital(_) => None,
            })
            .fold(0.0, f64::max);
        DVec3::new(0.0, 0.0, clearing_z)
    }

    /// This world LOWERED for the connection plane: the region forest alone, held as
    /// [`vd_core::worldgen::WorldRealms`]. The bodies — and every orbit element — deliberately do
    /// NOT survive the lowering, so the gateway can hold a world it queries without a dependency
    /// edge to this crate (SL4: the batch review found the crate fence carved open for
    /// vd-connection-plane precisely so it could hold a `WorldView`; it now receives this instead).
    #[must_use]
    pub fn lowered(&self) -> vd_core::worldgen::WorldRealms {
        vd_core::worldgen::WorldRealms::new(self.regions.clone())
    }
}

/// The walk-scale mandate forest as config-driven bodies, in forest order (Universe → Galaxy →
/// System A → Planet A → System B → Station A → Area A). All placements are `StaticOffset`, so the
/// lowering is byte-identical to the pre-generator forest. The GENERIC seed-driven child
/// enumeration (canonical scale) is deferred to P4 — its bodies are not live containment regions
/// until the D-41 non-zero-cell re-quantization.
fn generate_walk_forest(config: &UniverseConfig) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let st = &config.stellar;
    let pl = &config.planet;
    let sa = &config.satellite;
    let shell = |r: f64| Boundary::Shell { r };
    let boxed = |half: f64| Boundary::Aabb {
        half: DVec3::splat(half),
    };
    let at_x = |off: f64| Placement::StaticOffset(DVec3::new(off, 0.0, 0.0));
    let origin = Placement::StaticOffset(DVec3::ZERO);
    vec![
        // Universe: the ambient ROOT (parent None) — contains all reachable space (fold identity).
        // Never drawn (1e9 m was beyond the render cut since P3): `look = None`, the two-body
        // law stated structurally at the walk root too.
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe. The WALK galaxy IS
        // drawn (the 180 m containing box every walk gate has always seen) — bound == look at
        // human scale, so the split changes no walk pixel.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: Some(shell(sc.galaxy_r_m)),
        },
        // Star system A: nested in the Galaxy at the origin.
        GeneratedBody {
            realm: SYSTEM_A,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: Some(shell(st.system_soi_r_m)),
        },
        // Planet A: nested in system A, offset from the star.
        GeneratedBody {
            realm: PLANET_A,
            parent: Some(SYSTEM_A),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(sa.planet_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(pl.planet_soi_r_m)),
        },
        // Star system B: a DISJOINT sibling of system A under the Galaxy (a walkable galaxy gap between).
        GeneratedBody {
            realm: SYSTEM_B,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: at_x(sa.system_b_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(st.system_soi_r_m)),
        },
        // Station A: a first-class Station BOX under System A (depth 3), on the -X side opposite Planet A.
        GeneratedBody {
            realm: STATION_A,
            parent: Some(SYSTEM_A),
            shape: boxed(sa.station_half_m),
            placement: at_x(sa.station_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(boxed(sa.station_half_m)),
        },
        // Area A: a first-class sub-planet Area BOX under Planet A (depth 4) — the DEEPEST region.
        GeneratedBody {
            realm: AREA_A,
            parent: Some(PLANET_A),
            shape: boxed(sa.area_half_m),
            placement: at_x(sa.area_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(boxed(sa.area_half_m)),
        },
    ]
}

/// A FIXTURE forest containing PLAYER-BUILT structures — a station and an area — hand-placed beside the
/// generated bodies.
///
/// NOT A SECOND WORLD, and the distinction is the whole point. The seed generates what NATURE puts
/// there: stars and their planets. Stations and areas are built by PLAYERS, so no seeded world contains
/// one, at any scale. Tests that exercise crossing into a station therefore have to place it themselves —
/// exactly as a player would — and that is what this is for. It is never called by a running game.
///
/// It previously masqueraded as world generation, which is how the login side ended up descending a
/// roster with hand-placed bodies while the shards built the real thing: the two disagreed about what
/// exists because one of them was a test fixture.
///
/// The single source of truth for realm→region geometry (see the module docs). At P3 returns the
/// static WALK-scale mandate forest, now GENERATED from [`UniverseConfig::walk_scale`] via
/// [`generate_walk_forest`] + [`to_regions`] (byte-identical to the pre-generator forest);
/// `_seed_universe` is threaded for the frozen P4/P5 `f(seed)` signature (unused while static —
/// the seed-driven canonical generation lands at P4).
#[must_use]
pub fn realm_regions_for(seed_universe: u64) -> Vec<RealmRegion> {
    realm_regions_for_walk_config(seed_universe, &UniverseConfig::walk_scale())
}

/// The walk mandate forest for an EXPLICIT `config` (RLM 5f-4) — the walk topology of
/// [`generate_walk_forest`] lowered by [`to_regions`], carrying `config.interest` into each region's AoI
/// band. `realm_regions_for` is this with [`walk_scale`](UniverseConfig::walk_scale) (AoI inert,
/// byte-identical); a demand cluster passes [`walk_demand`](UniverseConfig::walk_demand) for the LIVE band
/// over the SAME geometry. Uses `generate_walk_forest` (NOT `generate_system_forest`, whose `n_planets = 0`
/// at walk yields only Universe+Galaxy+System A) so the full mandate chain (Planet/Station/Area/System B) is
/// present — the same forest [`container_coord_at`] descends. `_seed_universe` threads the frozen P4/P5
/// `f(seed)` signature (unused while static).
#[must_use]
pub fn realm_regions_for_walk_config(
    _seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    to_regions(&generate_walk_forest(config), config)
}

/// The regions a shard hosting `hosted_realm` evaluates CONTAINMENT against: its own realm + its ancestor
/// chain to the ambient root + the children it hosts authority INTO — **never siblings**. A sibling
/// crossing routes through the shared PARENT (leaving a system lands you in the galaxy ANCESTOR, and the
/// GALAXY shard — which hosts the systems as its children — detects the entry into the next one). This
/// keeps the per-shard region set O(depth + owned-children), bounded by `MAX_REGIONS`; a galaxy of
/// THOUSANDS of sibling systems never loads them all (an ambient shard scanning many children is the P6
/// spatial index, DEFERRED D-45). Closed-form `f(seed, hosted_realm)`, replicated by construction (HR1).
#[must_use]
pub fn realm_neighbourhood_for(seed_universe: u64, hosted_realm: RealmId) -> Vec<RealmRegion> {
    let all = realm_regions_for(seed_universe);
    let ancestry = ancestor_realms(&all, hosted_realm);
    all.iter()
        .copied()
        .filter(|r| ancestry.contains(&r.realm) || r.parent == Some(hosted_realm))
        .collect()
}

/// The regions a CO-HOSTING shard evaluates containment against: the UNION of the per-realm
/// neighbourhoods over every realm the shard HOLDS (the un-hosted-child cure). A shard that hosts its
/// system AND that system's Planet/Station/Area children must evaluate the deeper regions (a Planet's
/// child Area is a GRANDCHILD of the system, absent from the system's own neighbourhood), so the shard
/// scans the union — deduped by realm, order-stable (region forest order), so the boot depth-key/guard
/// results are deterministic. For a SINGLE-realm shard (`held == {hosted}`) this equals
/// [`realm_neighbourhood_for`] exactly (byte-identical). Closed-form `f(seed, held)`, replicated by
/// construction (HR1) — the held-set is itself seed-derivable topology, not shared mutable state.
#[must_use]
// FIXTURE path — scopes the hand-placed roster (with its player-built station and area), NOT a generated
// world. The `_config` twin is the real one.
pub fn realm_neighbourhood_for_held(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
) -> Vec<RealmRegion> {
    // Scoped over the FIXTURE roster, matching its single-realm twin `realm_neighbourhood_for`. Both
    // exist so a test can hold a station or an area — realms a player builds and no seed ever produces.
    // Routing this at the generated world instead would silently disagree with its own twin.
    neighbourhood_scope(&realm_regions_for(seed_universe), held)
}

/// The co-hosting neighbourhood for an EXPLICIT `config` (RLM 5f-4) — the config twin of
/// [`realm_neighbourhood_for_held`], preserving the ancestors-union-direct-children ("never siblings")
/// scope. A demand shard passes [`walk_demand`](UniverseConfig::walk_demand) so its evaluated regions carry
/// the LIVE AoI band; `realm_neighbourhood_for_held` is this with `walk_scale` (inert, byte-identical). One
/// implementation (HR3).
#[must_use]
pub fn realm_neighbourhood_for_held_config(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    // THE SEAM, CLOSED. This read the WALK roster while the shards built the SYSTEM forest, so the login
    // side and the simulating side described different worlds from the same seed — the client was told a
    // star system was 40 m across while the shards flew planets to 152. Two identical functions differing
    // only in which world they consult is the same defect as branching on shard kind (HR3), wearing
    // different clothes. There is now ONE forest, and this is a thin alias kept only so the call sites
    // read naturally; it will collapse into its twin when the walk world goes.
    realm_neighbourhood_for_config(seed_universe, held, config)
}

/// The config twin of [`realm_neighbourhood_for_held_config`] over the SYSTEM forest (orbiting planets) — the
/// scope a `Visual`/`VisualDemand` shard boots. A planet shard thus evaluates containment against its OWN
/// realm + its ancestor chain + the children IT authors, and NEVER its sibling planets — the direct cure for
/// the origin-stacking re-home flap (a shard cannot place a realm it does not author, so it must not fold it).
/// Same ancestors-union-direct-children scope, over [`realm_regions_for_config`] instead of the walk forest.
/// One filter (HR3). Closed-form `f(seed, held, config)`, replicated by construction (HR1).
#[must_use]
pub fn realm_neighbourhood_for_config(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    neighbourhood_scope(&realm_regions_for_config(seed_universe, config), held)
}

// --- Stellar/orbital PHYSICS (scale-independent; walk + canonical share these) ---
/// Salpeter IMF slope α (Salpeter 1955): `dN/dM ∝ M^-2.35`.
const IMF_SLOPE: f64 = 2.35;
/// Stellar mass sampling bounds (solar masses): the hydrogen-burning limit to a massive-O cap.
const IMF_MASS_LO_MSUN: f64 = 0.08;
const IMF_MASS_HI_MSUN: f64 = 120.0;
/// Titius-Bode orbital spacing seed (AU) + geometric ratio (Chambers 1996).
const ORBITAL_A0_AU: f64 = 0.4;
const ORBITAL_RATIO: f64 = 1.7;
/// Rayleigh scale for orbital eccentricity / inclination (Fabrycky 2014) — small so sampled
/// values stay well inside `KEPLER_ECC_MAX` (the generator also hard-caps at `ecc_cap`).
const ECC_SIGMA: f64 = 0.03;
/// The eccentricity cap in RAYLEIGH SIGMAS (the placement arc S4, owner-gated lever 1): `ecc_cap =
/// ECC_SIGMA · ECC_CAP_SIGMAS = 0.12`, and the AU compression is solved against the APOAPSIS at that
/// cap — so every planet's worst instant lands inside its system shell BY CONSTRUCTION, for every
/// seed, exactly. The cap replaces `KEPLER_ECC_MAX` doing geometry duty (a solver-convergence bound
/// has no business sizing a world); the clamp truncates the physical Rayleigh distribution with
/// probability `e^-(4²/2) = 3.35e-4` per planet — named, derived, never a magic number. MEASURED
/// before this lever: 2 of 3 systems' outer planets crossed their own shell at apoapsis (152.57 m and
/// 155.05 m against 150 m) — the S0 tripwire this lever turns green.
const ECC_CAP_SIGMAS: f64 = 4.0;
const INCL_SIGMA: f64 = 0.02;
/// Per-system occurrence probability of a station / a sub-planet area district.
const STATION_OCCURRENCE_PROB: f64 = 0.3;
const AREA_OCCURRENCE_PROB: f64 = 0.3;

// (The `canonical()`/`seed_derived()` real-scale presets and their CANONICAL_* geometry constants
// are DELETED — SL5, Stage-C audit :866: two complete world variants with zero production callers,
// whose own doc conceded they were not even the starting point for the true-scale work ("its
// astronomical-unit conversion is 2.5x the real value and its bodies do not nest correctly"). The
// true-astronomical-scale numbers are an owed change to THE ONE world, never a parallel preset.
// `CANONICAL_STAR_MASS_KG` and the taxonomy's `CANONICAL_*` census/threshold data survive — they are
// physical constants THE world reads, not preset geometry.)

/// The walk forest is exactly two systems (A + its disjoint sibling B).
const WALK_SYSTEM_COUNT: u32 = 2;

/// Ambient-root + galaxy + client-render scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleConfig {
    pub universe_r_m: f64,
    pub galaxy_r_m: f64,
}
// (`au_to_render_m` is DELETED — real-scale design §3.3.7: in-system compression is 1.000000
// EXACTLY, and the field does not exist to be anything else. Orbits are `a0·√L·ratio^n` in TRUE
// metres through the one IAU AU constant.)

/// Galaxy population + morphology census.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GalaxyConfig {
    /// Cumulative galaxy-type thresholds for `taxonomy::sample_galaxy_type`.
    pub type_cumulative: [f64; 2],
    pub system_count_lo: u32,
    pub system_count_hi: u32,
}

/// Star physics: SOI scale + the IMF + the mass-luminosity fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StellarConfig {
    /// The WALK fixture forest's system shell (read only by [`generate_walk_forest`]). The
    /// generated world's system shells are SOLVED per system by the clearance law
    /// (real-scale design §3.2) — no single number could state them.
    pub system_soi_r_m: f64,
    pub imf_slope: f64,
    pub mass_lo_msun: f64,
    pub mass_hi_msun: f64,
    pub mlr_segments: [(f64, f64, f64); 3],
    // (`central_mass_kg` is DELETED — real-scale design §3.3.7: every planet's
    // `OrbitalElements::central_mass` is its OWN star's drawn mass in kg,
    // `mass_msun · taxonomy::M_SUN_KG`; the synthetic Kepler-tuned mass died with the
    // compression.)
    /// The radius of the ring the non-origin systems are spaced around (system 0 sits at the galactic
    /// origin). Must leave every system's boundary disjoint from every other's AND inside the galaxy —
    /// overlapping systems would make "which realm contains this position" ambiguous, which is the one
    /// question the whole authority model rests on.
    ///
    /// HOW MANY systems is NOT here: it is drawn from the galaxy's own census
    /// ([`GalaxyConfig::system_count_lo`]..=[`GalaxyConfig::system_count_hi`]) against the galaxy's seed,
    /// so two galaxies from one universe differ. A second count field here would be a second source of
    /// truth for the same fact.
    pub system_ring_r_m: f64,
}

/// Planet physics: SOI scale, orbital spacing, eccentricity/inclination, and the frost/mass
/// thresholds (kept as raw f64 so the config stays serde-clean; `frost_thresholds()` builds the
/// `taxonomy::FrostThresholds` view).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanetConfig {
    /// The WALK fixture forest's planet shell (read only by [`generate_walk_forest`]). A
    /// generated planet's shell is its GRAVITATIONAL SOI at its drawn mass (D-REAL-1, the
    /// owner's 2026-08-18 ruling), clamped by half the worst-instant inter-orbit gap.
    pub planet_soi_r_m: f64,
    pub orbital_a0_au: f64,
    pub orbital_ratio: f64,
    pub ecc_sigma: f64,
    pub incl_sigma: f64,
    /// Hard eccentricity cap the generator clamps to — a fail-loud cross-slice invariant: it
    /// MUST stay `<= KEPLER_ECC_MAX` (the fixed Kepler solver's convergence domain).
    pub ecc_cap: f64,
    pub frost_coeff_au: f64,
    pub m_gas_mearth: f64,
    pub m_core_crit_mearth: f64,
    /// The Kepler radius-valley normalisation (`taxonomy::RADIUS_VALLEY_1SEARTH_REARTH`).
    pub valley_r1_rearth: f64,
    /// The valley's insolation exponent (`taxonomy::RADIUS_VALLEY_INSOLATION_EXP`).
    pub valley_insolation_exp: f64,
    /// The number of `Orbital` planets [`generate_system_forest`] emits for THIS system. `0` on
    /// walk_scale() (no planet body ⇒ ambient-only forest, byte-identity); the world preset sets
    /// the DERIVED count ([`derived_planet_count`] — 9, scale-free). The `0..n_planets` range is
    /// the generator's only control flow.
    pub n_planets: u32,
    /// The planet mass draw's lower bound, Earth masses ([`PLANET_MASS_LO_MEARTH`] — Mercury).
    pub mass_lo_mearth: f64,
    /// Protoplanetary disc-to-star mass fraction ([`DISC_MASS_FRACTION`]) — the per-planet
    /// budget arm of the draw's upper bound `min(mass_cap, disc_fraction·M★/N)`.
    pub disc_mass_fraction: f64,
    /// The absolute per-planet mass cap, Earth masses ([`M_JUP_MEARTH`] — Jupiter).
    pub mass_cap_mearth: f64,
}

impl PlanetConfig {
    /// The `taxonomy::FrostThresholds` view over the raw config fields (fed to `classify_planet`).
    #[must_use]
    pub fn frost_thresholds(&self) -> FrostThresholds {
        FrostThresholds {
            m_gas_mearth: self.m_gas_mearth,
            m_core_crit_mearth: self.m_core_crit_mearth,
            valley_r1_rearth: self.valley_r1_rearth,
            valley_insolation_exp: self.valley_insolation_exp,
        }
    }
}

/// Satellite bodies: station/area occurrence + the walk-scale placement geometry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SatelliteConfig {
    pub station_prob: f64,
    pub area_prob: f64,
    pub station_half_m: f64,
    pub area_half_m: f64,
    pub station_offset_m: f64,
    pub area_offset_m: f64,
    pub planet_offset_m: f64,
    pub system_b_offset_m: f64,
}

/// Containment-band edges (the acquire/release hysteresis).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandConfig {
    pub inset_m: f64,
    pub outset_m: f64,
    pub k_safety_extra: f64,
}

impl BandConfig {
    /// Build the P3 static containment band (v_rel = 0, dt = 1 — the widening is inert). Fallible
    /// (the ctor validates edges); walk-scale edges are valid by construction.
    pub fn build(&self) -> Result<ContainmentBand, BandError> {
        ContainmentBand::for_containment_velocity_safe(
            self.inset_m,
            self.outset_m,
            0.0,
            1.0,
            self.k_safety_extra,
        )
    }
}

// ---- RLM Step 2: per-realm AoI radii (all seed-relative factors, no magic numbers) ----------------
/// The universe seconds-per-tick the AoI widening is measured against — MUST equal `StubConfig.tick_dt_s`
/// (a boot `debug_assert!` cross-checks it, M-2). Named, not inline. `pub` so THE-world pins in
/// sibling crates state the shipped posture from THIS one place instead of copying the literal.
pub const AOI_TICK_DT_S: f64 = 0.05;
/// Grace ticks a would-be release is held (1 s at the visual 20 Hz).
const VISUAL_AOI_GRACE_TICKS: u32 = 20;
/// Extra velocity-safety margin folded into the dead-zone widening (beyond `K_SAFETY`).
const VISUAL_AOI_K_SAFETY_EXTRA: f64 = 0.5;
/// The visual occupant's max speed (m/s) — MUST equal `StubConfig.move_speed_mps · time_multiplier`
/// (boot `debug_assert!`, M-2), so the anti-thrash pad is measured against the speed the sim integrates.
/// Under the single visibility factor (`spin_up_factor == tear_down_factor`) the geometric dead-zone
/// collapses, so THIS non-zero occupant speed is what keeps `tear_down > spin_up` (band validity requires
/// `occupant_v_max + v_child > 0`; see [`UniverseConfig::visual_demand`]). `pub` for the same
/// one-place reason as [`AOI_TICK_DT_S`].
pub const VISUAL_OCCUPANT_V_MAX_MPS: f64 = 2.0;

/// Walk-demand-scale AoI (RLM 5f-4): the LIVE band for the WALK forest, so a WALKING occupant's AoI
/// crosses each separated child's band. Tighter than visual (a walking player over metres, not AU): spin a
/// child up within this multiple of its extent, release past the larger tear-down multiple — HR3
/// proportional, no kind-match. The two DYNAMICS inputs (occupant speed, tick dt) are NOT consts — they are
/// supplied at the composer boot from the LIVE cluster values, which is what closes the M-2 two-home owe (a
/// hardcoded `AOI_TICK_DT_S = 0.05` is wrong at the dev cluster's 50 Hz = 0.02).
const WALK_DEMAND_AOI_SPIN_UP_FACTOR: f64 = 1.2;
const WALK_DEMAND_AOI_TEAR_DOWN_FACTOR: f64 = 1.8;
const WALK_DEMAND_AOI_K_SAFETY_EXTRA: f64 = 0.5;

/// Per-realm AoI radii as UNIFORM FACTORS of the realm's own finite extent (HR3: no match-on-kind — a
/// bigger realm reaches proportionally farther), plus the widening inputs (`occupant_v_max_mps`,
/// `tick_dt_s`) so [`to_regions`] stays a pure `f(UniverseConfig)` (a boot `debug_assert!` cross-checks
/// them against the live `StubConfig`, M-2 — no two-home drift). `walk_scale`/`canonical` set
/// `spin_up_factor = 0` ⇒ AoI inert ⇒ behaviour byte-identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterestConfig {
    pub spin_up_factor: f64,
    pub tear_down_factor: f64,
    pub grace_ticks: u32,
    pub k_safety_extra: f64,
    pub occupant_v_max_mps: f64,
    pub tick_dt_s: f64,
}

impl InterestConfig {
    /// Build the per-realm AoI band from the realm's own `finite_extent` + the child's own orbital
    /// closing speed `v_child` (its `v_peri`; 0 for a static child). At walk-scale (`spin_up_factor <=
    /// 0`) returns the inert band BRANCHLESSLY — NEVER through the fallible ctor (whose reject arm the
    /// byte-identity path must not touch). Fallible only for the LIVE case.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] from [`AoiConfig::for_velocity_safe`] if the live factors are degenerate.
    pub fn build(&self, finite_extent: f64, v_child: f64) -> Result<AoiConfig, BandError> {
        if self.spin_up_factor <= 0.0 {
            Ok(AoiConfig::inert())
        } else {
            AoiConfig::for_velocity_safe(
                finite_extent,
                self.spin_up_factor,
                self.tear_down_factor,
                aoi_v_rel_mps(finite_extent, self.occupant_v_max_mps + v_child),
                self.tick_dt_s,
                self.grace_ticks,
                self.k_safety_extra,
            )
        }
    }

    /// The inert preset (walk/canonical): zero factor ⇒ AoI never fires ⇒ byte-identity.
    #[must_use]
    pub fn inert() -> InterestConfig {
        InterestConfig {
            spin_up_factor: 0.0,
            tear_down_factor: 0.0,
            grace_ticks: 0,
            k_safety_extra: 0.0,
            occupant_v_max_mps: 0.0,
            tick_dt_s: AOI_TICK_DT_S,
        }
    }

    /// Whether AoI is LIVE (a positive spin-up factor) — the M-2 boot cross-check applies only here.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.spin_up_factor > 0.0
    }
}

/// THE one config home for the seed universe generator — named sub-structs (no god-struct),
/// every field seed-derivable and doc-cited (no magic numbers).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UniverseConfig {
    pub scale: ScaleConfig,
    pub galaxy: GalaxyConfig,
    pub stellar: StellarConfig,
    pub planet: PlanetConfig,
    pub satellite: SatelliteConfig,
    pub band: BandConfig,
    /// Per-realm AoI radii (RLM Step 2). Inert at walk/canonical (byte-identity); live at visual.
    pub interest: InterestConfig,
    /// PLAYER-BUILT content planted beside the generated bodies (look_horizon.md slice 5
    /// `G-IDENTICAL` — the SL5 fixture-forest doctrine: "fixtures planting player-built regions on
    /// THE world" is APPROVED-BY-EXISTING-RULING). NOT a world variant and NOT a scale: the seed
    /// generates what nature puts there, and this field adds what a player would have built — the
    /// same one generator, the same lowering, the same fences, plus content. `None` (the default,
    /// and every shipped constructor's value) is byte-identical to the pre-field world.
    #[serde(default)]
    pub fixture_plant: FixturePlant,
}

impl UniverseConfig {
    /// The walk-scale preset — its fields ARE today's named geometry consts, so the generator
    /// reproduces the EXACT current forest (the byte-identity source). The galaxy is renderable
    /// (`galaxy_r_m < render_extent_m`), so the client draws it as the containing box.
    #[must_use]
    pub fn walk_scale() -> UniverseConfig {
        UniverseConfig {
            scale: ScaleConfig {
                universe_r_m: UNIVERSE_R_M,
                galaxy_r_m: GALAXY_R_M,
            },
            galaxy: GalaxyConfig {
                type_cumulative: GalaxyType::CANONICAL_CUMULATIVE,
                system_count_lo: WALK_SYSTEM_COUNT,
                system_count_hi: WALK_SYSTEM_COUNT,
            },
            stellar: StellarConfig {
                system_soi_r_m: SYSTEM_SOI_R_M,
                imf_slope: IMF_SLOPE,
                mass_lo_msun: IMF_MASS_LO_MSUN,
                mass_hi_msun: IMF_MASS_HI_MSUN,
                mlr_segments: SpectralClass::MLR_SEGMENTS,
                // Where the walk roster's second star already sat. It is no longer inert: with ONE
                // world, this preset drives the same generator as everything else, and a zero ring
                // would stack both stars on the origin — two authorities over one point.
                system_ring_r_m: SYSTEM_B_OFFSET_M,
            },
            planet: PlanetConfig {
                planet_soi_r_m: PLANET_SOI_R_M,
                orbital_a0_au: ORBITAL_A0_AU,
                orbital_ratio: ORBITAL_RATIO,
                ecc_sigma: ECC_SIGMA,
                incl_sigma: INCL_SIGMA,
                // The GEOMETRY cap (4σ of the Rayleigh draw), NOT the solver bound: the compression
                // below solves apoapsis-at-this-cap exactly onto the system shell (see ECC_CAP_SIGMAS).
                ecc_cap: ECC_SIGMA * ECC_CAP_SIGMAS,
                frost_coeff_au: crate::taxonomy::FROST_COEFF_AU,
                m_gas_mearth: FrostThresholds::CANONICAL.m_gas_mearth,
                m_core_crit_mearth: FrostThresholds::CANONICAL.m_core_crit_mearth,
                valley_r1_rearth: FrostThresholds::CANONICAL.valley_r1_rearth,
                valley_insolation_exp: FrostThresholds::CANONICAL.valley_insolation_exp,
                n_planets: 0, // ambient-only forest (no Orbital body) — the world derives N.
                mass_lo_mearth: PLANET_MASS_LO_MEARTH,
                disc_mass_fraction: DISC_MASS_FRACTION,
                mass_cap_mearth: M_JUP_MEARTH,
            },
            satellite: SatelliteConfig {
                station_prob: STATION_OCCURRENCE_PROB,
                area_prob: AREA_OCCURRENCE_PROB,
                station_half_m: STATION_HALF_M,
                area_half_m: AREA_HALF_M,
                station_offset_m: STATION_A_OFFSET_M,
                area_offset_m: AREA_OFFSET_M,
                planet_offset_m: PLANET_A_OFFSET_M,
                system_b_offset_m: SYSTEM_B_OFFSET_M,
            },
            band: BandConfig {
                inset_m: CONTAINMENT_INSET_M,
                outset_m: CONTAINMENT_OUTSET_M,
                k_safety_extra: 0.0,
            },
            interest: InterestConfig::inert(), // walk: AoI OFF ⇒ byte-identity.
            fixture_plant: FixturePlant::None, // nothing built ⇒ the seed world exactly.
        }
    }

    /// This config with the [`FixturePlant::StationArea`] pair planted (look_horizon.md slice 5
    /// `G-IDENTICAL`). A BUILDER, not a preset: the geometry, the census, the bands — everything —
    /// stays exactly this config's; only player-built content is added. The process boots opt in
    /// through `VD_FIXTURE_PLANT` (vd-bins), so a cluster is planted whole or not at all.
    #[must_use]
    pub fn with_station_area_plant(mut self) -> UniverseConfig {
        self.fixture_plant = FixturePlant::StationArea;
        self
    }

    /// The COMPRESSED-REAL visual geometry on a `walk_scale()` clone (system SOI 150, 5 Kepler planets,
    /// outer period 300 s) WITHOUT an interest band — the ONE game geometry that `visual_scale` (static
    /// render) and `visual_demand` (live demand cluster) both drive, so the two can NEVER disagree on
    /// geometry (they call the SAME parameterized derive helpers with the SAME compressed-real numbers).
    /// Mutates a `walk_scale()` CLONE (byte-identity: walk never calls these helpers), overriding
    /// only the System SOI + the four moving-planet fields (AU→render + planet SOI both DERIVED so the outer
    /// orbit + SOI + margin land exactly at the System surface — provable non-overlap + strict containment,
    /// the SYNTHETIC central mass Kepler-3-tuned to a seconds-scale period, and `n_planets`).
    fn visual_geometry() -> UniverseConfig {
        let mut cfg = UniverseConfig::walk_scale();
        // ▲ THE IN-SYSTEM TRUE-SIZE RE-SOLVE (real-scale design §3.3, the taxonomy arc's flag
        // day): the compressed interim in-system numbers (150 m shells, 3.95 m SOIs, AU
        // compression, synthetic Kepler mass, 5 hand-counted planets) are GONE. Orbits are the
        // √L-anchored ladder in TRUE metres; planet masses are drawn (log-uniform between
        // Mercury and the disc budget); radii are Chen–Kipping; every planet's shell is its
        // gravitational SOI at its drawn mass (D-REAL-1); each system's shell is SOLVED by the
        // one clearance law at the mass cap. The planet COUNT is derived and scale-free: the
        // disc edge over the ladder ratio — 9 for every star at every seed.
        cfg.planet.n_planets = derived_planet_count(
            cfg.planet.orbital_a0_au,
            cfg.planet.orbital_ratio,
            (NEPTUNE_SMA_AU / crate::taxonomy::FROST_COEFF_AU) * cfg.planet.frost_coeff_au,
        );
        cfg.galaxy.system_count_lo = WORLD_SYSTEM_COUNT;
        cfg.galaxy.system_count_hi = WORLD_SYSTEM_COUNT;
        // ▲ THE OUTER GEOMETRY (real-scale addendum §A2 — the four changed numbers, derivations
        // at their consts): the universe from the storage fence (2⁵¹ m), the galaxy from the
        // τ-free outset (`R_uni − outset`), the placement radius from the reserved clearance
        // (`R_gal − clearance` = 0.2376656 ly), the compression χ = 16.378× stated at the
        // census consts.
        cfg.scale.universe_r_m = REAL_UNIVERSE_R_M;
        cfg.scale.galaxy_r_m = REAL_GALAXY_R_M;
        cfg.stellar.system_ring_r_m = real_placement_r_m();
        cfg
    }

    /// The VISUAL-scale preset (D-45(a) FA-5): the STATIC-render expression of the one compressed-real game
    /// geometry (via [`visual_geometry`](UniverseConfig::visual_geometry)) — a 5-planet Kepler system whose
    /// planets ORBIT visibly. Its interest band is the ONE visibility rule: `spin_up_factor ==
    /// tear_down_factor == cot(θ/2)` (a realm streams in once it subtends ≥ θ_min), authored against the
    /// static occupant speed [`VISUAL_OCCUPANT_V_MAX_MPS`]. Under a single visibility factor the geometric
    /// dead-zone collapses, so band validity rests on `occupant_v_max + v_child > 0` (that non-zero speed);
    /// [`visual_demand`](UniverseConfig::visual_demand) is the live demand-cluster expression of the SAME
    /// geometry.
    #[must_use]
    pub fn visual_scale() -> UniverseConfig {
        let vis_factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let mut cfg = UniverseConfig::visual_geometry();
        cfg.interest = InterestConfig {
            spin_up_factor: vis_factor,
            tear_down_factor: vis_factor,
            grace_ticks: VISUAL_AOI_GRACE_TICKS,
            k_safety_extra: VISUAL_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps: VISUAL_OCCUPANT_V_MAX_MPS,
            tick_dt_s: AOI_TICK_DT_S,
        };
        cfg
    }

    /// The VISUAL-DEMAND preset (RLM realistic-demo Slice 0): the LIVE demand-cluster expression of the
    /// SAME compressed-real geometry as [`visual_scale`](UniverseConfig::visual_scale) (via
    /// [`visual_geometry`](UniverseConfig::visual_geometry)), with the interest band built the `walk_demand`
    /// way — the two DYNAMICS inputs are ARGUMENTS from the live cluster (`occupant_v_max_mps` = the
    /// occupant's max speed `move_speed · time_multiplier`, `tick_dt_s` = the cluster's seconds-per-tick),
    /// the loiter grace converted from seconds ([`grace_ticks_from_seconds`]), and the SAME `cot(θ/2)`
    /// visibility factor for both edges. It reuses [`generate_system_forest`]/[`planet_elements`]/
    /// [`moving_children_for_config`] verbatim (zero new generator control flow).
    ///
    /// PRECONDITION (the equal-factor tripwire): under a single visibility factor
    /// (`spin_up_factor == tear_down_factor`) the geometric dead-zone collapses, so band validity requires
    /// `occupant_v_max_mps + v_child > 0` — a LIVE occupant. A zero-relative-velocity band is
    /// [`BandError::InvalidEdges`], NEVER a valid inert band (and `to_regions`'s `.expect` would panic at
    /// boot on one). The shipped callers thread a positive speed (the demand cluster's 15 m/s ship); an
    /// idle-occupant (`v_rel = 0`) preset under one factor has no valid band (a deferred design choice, not
    /// papered over with an epsilon — see `equal_visibility_factor_with_zero_v_rel_is_err_not_panic`).
    #[must_use]
    pub fn visual_demand(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        let vis_factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let mut cfg = UniverseConfig::visual_geometry();
        cfg.interest = InterestConfig {
            spin_up_factor: vis_factor,
            tear_down_factor: vis_factor,
            grace_ticks: grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, tick_dt_s),
            k_safety_extra: WALK_DEMAND_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps,
            tick_dt_s,
        };
        cfg
    }

    /// **THE WORLD.** Not a preset, not a scale, not a variant — the only universe this program has.
    ///
    /// WHY THIS EXISTS AND WHY EVERYTHING ELSE HERE IS GOING. There used to be a knob, and on a live
    /// cluster it was MEASURED holding two different values at once: the orchestrator booted one world
    /// and the gateway another, in the same launch, from the same script. Logins were placed by one
    /// universe's rules and simulated by another's. That is not a bug in either half — it is what a knob
    /// IS, and no amount of care downstream removes it. The owner's ruling, twice now: one world, all the
    /// time, with nothing to select.
    ///
    /// The two arguments are NOT a choice of world. They are facts about the cluster running it — how
    /// fast an occupant may travel and how long a tick lasts — which the interest band has to be measured
    /// against or it is measured against a number nobody uses.
    ///
    /// ⚠ ITS SIZES ARE NOT FINAL, and that is a separate, sequenced piece of work rather than a variant
    /// hiding here: the geometry below still carries the tiny-world numbers, stations and areas are still
    /// hand-placed rather than generated, and the galaxy is not yet a lattice of cells. Those land as
    /// changes to THIS world's numbers. Never as a second one.
    #[must_use]
    pub fn world(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        UniverseConfig::visual_demand(occupant_v_max_mps, tick_dt_s)
    }

    /// The WALK-demand preset (RLM 5f-4): the EXACT walk-scale geometry with the AoI band turned LIVE, so a
    /// WALKING occupant drives demand-driven spin-up/down over the static walk forest. Differs from
    /// [`walk_scale`](UniverseConfig::walk_scale) in the `interest` field ONLY (geometry byte-identical). The
    /// two DYNAMICS inputs are ARGUMENTS from the live cluster: `occupant_v_max_mps` (the occupant's max
    /// speed `move_speed · time_multiplier`) and `tick_dt_s` (the cluster's seconds-per-tick) — so the
    /// anti-thrash pad + the loiter grace are measured against the speed the sim integrates and the rate it
    /// ticks at (closing the M-2 two-home owe; no hardcoded `AOI_TICK_DT_S`).
    #[must_use]
    pub fn walk_demand(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        let mut cfg = UniverseConfig::walk_scale();
        cfg.interest = InterestConfig {
            spin_up_factor: WALK_DEMAND_AOI_SPIN_UP_FACTOR,
            tear_down_factor: WALK_DEMAND_AOI_TEAR_DOWN_FACTOR,
            grace_ticks: grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, tick_dt_s),
            k_safety_extra: WALK_DEMAND_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps,
            tick_dt_s,
        };
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::KEPLER_ECC_MAX;
    use vd_core::geometry::{region_depth, region_signed_distance};
    use vd_core::realm_coord::RealmCoord;
    use vd_core::realm_path::{RealmKindTag, RealmPath};
    use vd_core::worldgen::{coord_of_realm, default_home_realm};

    /// The world's DERIVED planet count (9 — scale-free; the disc edge over the ladder).
    fn world_n_planets() -> u32 {
        UniverseConfig::visual_scale().planet.n_planets
    }

    fn regions() -> Vec<RealmRegion> {
        realm_regions_for(0)
    }

    /// THE world at a nominal occupant speed — the config-driven twin the bins boot uses.
    fn boot_world_for_tests() -> WorldView {
        WorldView::generated(0, &UniverseConfig::world(15.0, 0.05))
    }

    #[test]
    fn the_world_preset_is_the_visual_demand_geometry_and_the_alias_is_its_twin() {
        // `UniverseConfig::world` IS `visual_demand` (SL5: one world, the name states the law), and
        // the held-config alias builds the SAME neighbourhood as the fn it delegates to (HR3: the
        // seam stays closed — two identical functions consulting different worlds is how the login
        // side and the simulating side once described different universes from one seed).
        let world = UniverseConfig::world(15.0, 0.05);
        let demand = UniverseConfig::visual_demand(15.0, 0.05);
        assert_eq!(
            generate_system_forest(0, &world).len(),
            generate_system_forest(0, &demand).len(),
            "one geometry"
        );
        let held = std::collections::BTreeSet::from([RealmId::System(7)]);
        assert_eq!(
            realm_neighbourhood_for_held_config(0, &held, &world),
            realm_neighbourhood_for_config(0, &held, &world),
            "the alias is byte-equal to its twin"
        );
    }

    #[test]
    fn the_sibling_fence_judges_an_orbit_on_its_shell_never_its_epoch() {
        // The overlap fence skips a KEPLER sibling on BOTH sides of the pair loop: an orbit is
        // judged on its shell (the SOI nesting fence), not on where its epoch anchor happens to sit
        // — an epoch-position overlap between an orbiting body and a static one is not a defect.
        let orbital = Placement::Orbital(OrbitalElements {
            sma: 1.0e11,
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: 1.989e30,
        });
        let body = |realm: RealmId, placement: Placement| GeneratedBody {
            realm,
            parent: Some(RealmId::System(1)),
            shape: Boundary::Shell { r: 10.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 10.0 }),
            placement,
            photometrics: None,
        };
        // A static + an ORBITING sibling at the "same place": no overlap verdict — the orbiter is
        // skipped by the inner arm.
        let mixed = [
            body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
            body(RealmId::Planet(2), orbital),
        ];
        assert!(siblings_disjoint(&mixed).is_ok());
        // Two STATIC siblings genuinely overlapping: the fence still fires (non-vacuity).
        let clash = [
            body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
            body(
                RealmId::Planet(2),
                Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
            ),
        ];
        assert!(siblings_disjoint(&clash).is_err());
    }

    #[test]
    fn coord_of_realm_resolves_the_full_root_rooted_lineage_and_is_none_for_a_ship() {
        // The un-lossy RealmId→RealmCoord a source shard uses to KeepAlive-demand a crossing DEST's whole
        // ancestor chain (the Symptom-B freeze fix). A System resolves to [Universe, Galaxy, System]; a
        // Planet one level deeper; a ship (entity-backed, no seed level) resolves to None.
        let regions = realm_regions_for(0);
        let want_sys = RealmCoord::from_path(RealmPath::from_levels(vec![
            level_of(UNIVERSE).expect("Universe is a seed realm"),
            level_of(GALAXY).expect("Galaxy is a seed realm"),
            level_of(SYSTEM_A).expect("System A is a seed realm"),
        ]))
        .expect("a 3-level path has a leaf");
        let want_planet = want_sys.child(level_of(PLANET_A).expect("Planet A is a seed realm"));
        assert_eq!(coord_of_realm(&regions, SYSTEM_A), Some(want_sys));
        assert_eq!(want_planet.path().levels().len(), 4); // full lineage, never lowered()
        assert_eq!(coord_of_realm(&regions, PLANET_A), Some(want_planet));
        // A ship is entity-backed (no seed RealmLevel) ⇒ None — the `?` early-return arm.
        let ship = RealmId::Ship(vd_core::ids::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            1,
        ));
        assert_eq!(coord_of_realm(&regions, ship), None);
    }

    #[test]
    fn region_signed_distance_is_frame_aware_and_identity_at_p3() {
        let rs = regions();
        let system_a = rs
            .iter()
            .find(|r| r.realm == SYSTEM_A)
            .expect("system A is in the forest");
        // At the system center: signed distance = -r_soi (fully inside). Identity frame ⇒ pos unchanged.
        let at_centre = vd_core::pose::StampedPose::at_rest(
            vd_core::pose::FrameRef::SystemSpace { system_seed: 0 },
            DVec3::ZERO,
            vd_core::ids::UniverseTick(0),
        );
        // The book: anchored on the pose's own frame, with the system's frame at the identity (the
        // P3 shipping shape — every placement is the identity, so the reframe moves nothing).
        let book = vd_core::placement::PlacementBook::new(
            at_centre.frame,
            at_centre.universe_tick,
            vec![(system_a.frame, vd_core::frame::FramePlacement::identity())],
        );
        let sd = region_signed_distance(&at_centre, system_a, &book).expect("ok");
        assert!(
            (sd - (-SYSTEM_SOI_R_M)).abs() < 1e-9,
            "center is r_soi inside: {sd}"
        );
    }

    #[test]
    fn realm_neighbourhood_scopes_to_own_ancestors_and_children_never_siblings() {
        // System 7's shard: own + ancestors (Galaxy, Universe) + children Planet 7 AND Station 7 (a
        // first-class child under System 7) — NOT sibling System 8, NOT the grandchild Area 7.
        let n7: Vec<RealmId> = realm_neighbourhood_for(0, SYSTEM_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(n7.contains(&SYSTEM_A));
        assert!(n7.contains(&GALAXY));
        assert!(n7.contains(&UNIVERSE));
        assert!(n7.contains(&PLANET_A));
        assert!(
            n7.contains(&STATION_A),
            "the Station is an OWNED child of System 7 — the shard scans it",
        );
        assert!(
            !n7.contains(&SYSTEM_B),
            "a shard NEVER loads a sibling — the scale-bounded rule",
        );
        assert!(
            !n7.contains(&AREA_A),
            "Area 7 is a grandchild (under Planet 7), not a direct child of System 7",
        );
        assert_eq!(n7.len(), 5);
        // The GALAXY shard: own + ancestor Universe + children System 7 & 8 (the between-space owner that
        // routes a sibling crossing) — NOT Planet 7 (a grandchild, not a direct child).
        let ng: Vec<RealmId> = realm_neighbourhood_for(0, GALAXY)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(ng.contains(&GALAXY));
        assert!(ng.contains(&UNIVERSE));
        assert!(ng.contains(&SYSTEM_A));
        assert!(ng.contains(&SYSTEM_B));
        assert!(
            !ng.contains(&PLANET_A),
            "a grandchild is not a direct child"
        );
        assert_eq!(ng.len(), 4);
        // The STATION 7 shard: its own realm + its ANCESTOR CHAIN (System 7, Galaxy, Universe) and NO
        // children (a leaf) — it never pulls its sibling Planet 7 (they share the System 7 parent).
        let nst: Vec<RealmId> = realm_neighbourhood_for(0, STATION_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(nst.contains(&STATION_A));
        assert!(nst.contains(&SYSTEM_A));
        assert!(nst.contains(&GALAXY));
        assert!(nst.contains(&UNIVERSE));
        assert!(
            !nst.contains(&PLANET_A),
            "the Station never loads its sibling Planet 7",
        );
        assert_eq!(nst.len(), 4);
        // The AREA 7 shard: its own realm + its ANCESTOR CHAIN (Planet 7, System 7, Galaxy, Universe) and
        // NO children — it never pulls its sibling Station 7 (they share the System 7 ancestor, not a parent).
        let nar: Vec<RealmId> = realm_neighbourhood_for(0, AREA_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(nar.contains(&AREA_A));
        assert!(nar.contains(&PLANET_A));
        assert!(nar.contains(&SYSTEM_A));
        assert!(nar.contains(&GALAXY));
        assert!(nar.contains(&UNIVERSE));
        assert!(
            !nar.contains(&STATION_A),
            "the Area never loads the Station (they are not parent/child)",
        );
        assert_eq!(nar.len(), 5);
        // A shard hosting an unknown realm ⇒ empty neighbourhood ⇒ the detector is inert (safe degrade).
        // Both appended kinds (Station/Area) at an ABSENT seed (99) degrade to empty — the seed-7 plant
        // above does NOT make every Station/Area live.
        assert!(realm_neighbourhood_for(0, RealmId::Station(99)).is_empty());
        assert!(realm_neighbourhood_for(0, RealmId::Area(99)).is_empty());
    }

    #[test]
    fn a_single_held_realm_neighbourhood_union_equals_the_single_neighbourhood() {
        // Co-hosting DEGENERATE case: a held-set of exactly one realm is byte-identical to
        // `realm_neighbourhood_for` — the single-realm shard path is untouched.
        for r in [SYSTEM_A, GALAXY, PLANET_A, STATION_A] {
            let single = realm_neighbourhood_for(0, r);
            let held = realm_neighbourhood_for_held(0, &std::collections::BTreeSet::from([r]));
            assert_eq!(
                single, held,
                "the held-set union for {{{r}}} equals its single neighbourhood",
            );
        }
    }

    #[test]
    fn a_cohosted_system_plus_children_union_reaches_the_deepest_grandchild_area() {
        // The un-hosted-child cure: a shard co-hosting System 7 + its children (Planet/Station/Area) must
        // evaluate the DEEPEST region (Area 7, a GRANDCHILD of System 7 absent from System 7's OWN
        // neighbourhood). The union reaches it via Planet 7 being held.
        let held = std::collections::BTreeSet::from([SYSTEM_A, PLANET_A, STATION_A, AREA_A]);
        let realms: Vec<RealmId> = realm_neighbourhood_for_held(0, &held)
            .iter()
            .map(|r| r.realm)
            .collect();
        for expected in [UNIVERSE, GALAXY, SYSTEM_A, PLANET_A, STATION_A, AREA_A] {
            assert!(realms.contains(&expected), "the union includes {expected}");
        }
        // System B (a SIBLING of System 7 — never a child/ancestor of any held realm) is EXCLUDED.
        assert!(
            !realms.contains(&SYSTEM_B),
            "a sibling system is never in the co-hosting union",
        );
        // The union deduplicates (Universe/Galaxy/System 7 appear once even though several held realms
        // share them as ancestors) — the region set is a valid single-root forest.
        assert_eq!(
            realms.len(),
            6,
            "6 distinct regions (7-forest minus the sibling System B)"
        );
    }

    #[test]
    fn realm_neighbourhood_for_config_excludes_sibling_planets_over_the_visual_forest() {
        // The flap cure at VISUAL scale: a planet shard's neighbourhood is its own realm + ancestors + the
        // children it authors — NEVER its sibling planets. A shard cannot place a realm it does not author, so
        // folding a sibling collapses it to the origin and a hosted occupant reads as inside all of them at
        // once (the production hot-potato). Unlike the walk forest, the visual system forest has MULTIPLE
        // orbiting planets, so this is where the exclusion actually bites.
        let cfg = UniverseConfig::visual_scale();
        let forest = realm_regions_for_config(0, &cfg);
        let planets: Vec<RealmId> = forest
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Planet(_)))
            .map(|r| r.realm)
            .collect();
        assert!(
            planets.len() >= 2,
            "the visual forest must have sibling planets to distinguish (got {planets:?})",
        );
        let target = planets[0];
        let sibling = planets[1];
        let parent = forest
            .iter()
            .find(|r| r.realm == target)
            .and_then(|r| r.parent)
            .expect("a visual-forest planet has a parent system");
        let scope: Vec<RealmId> =
            realm_neighbourhood_for_config(0, &std::collections::BTreeSet::from([target]), &cfg)
                .iter()
                .map(|r| r.realm)
                .collect();
        assert!(scope.contains(&target), "own realm is in scope");
        assert!(
            scope.contains(&parent),
            "the parent system (an ancestor) is in scope",
        );
        assert!(
            !scope.contains(&sibling),
            "a SIBLING planet is NEVER in scope — the origin-stacking flap cure",
        );
    }

    // ---- D-45(a) Slice 3b: UniverseConfig -------------------------------------------

    #[test]
    fn walk_scale_equals_the_named_geometry_consts() {
        let c = UniverseConfig::walk_scale();
        assert_eq!(c.scale.universe_r_m, UNIVERSE_R_M);
        // The galaxy is DERIVED to contain the ring of stars, no longer the walk constant: it must hold
        // every system with its reach, or a star sits outside its own galaxy.
        assert!(c.scale.galaxy_r_m > c.stellar.system_ring_r_m + c.stellar.system_soi_r_m);
        assert_eq!(c.stellar.system_soi_r_m, SYSTEM_SOI_R_M);
        assert_eq!(c.planet.planet_soi_r_m, PLANET_SOI_R_M);
        assert_eq!(c.satellite.planet_offset_m, PLANET_A_OFFSET_M);
        assert_eq!(c.satellite.system_b_offset_m, SYSTEM_B_OFFSET_M);
        assert_eq!(c.satellite.station_offset_m, STATION_A_OFFSET_M);
        assert_eq!(c.satellite.station_half_m, STATION_HALF_M);
        assert_eq!(c.satellite.area_offset_m, AREA_OFFSET_M);
        assert_eq!(c.satellite.area_half_m, AREA_HALF_M);
        assert_eq!(c.band.inset_m, CONTAINMENT_INSET_M);
        assert_eq!(c.band.outset_m, CONTAINMENT_OUTSET_M);
        assert_eq!(c.planet.n_planets, 0);
        // The re-solve's draw bounds ride every preset (walk never reads them — no Orbital body).
        assert_eq!(c.planet.mass_lo_mearth, PLANET_MASS_LO_MEARTH);
        assert_eq!(c.planet.disc_mass_fraction, DISC_MASS_FRACTION);
        assert_eq!(c.planet.mass_cap_mearth, M_JUP_MEARTH);
    }

    #[test]
    fn universe_config_presets_serde_round_trip() {
        for c in [
            UniverseConfig::walk_scale(),
            UniverseConfig::visual_scale(),
            // The PLANTED config too (look_horizon slice 5): the plant field is DATA and must
            // survive the codec like every other field — both enum arms round-trip.
            UniverseConfig::world(15.0, 0.05).with_station_area_plant(),
        ] {
            let bytes = postcard::to_allocvec(&c).expect("encode");
            let back: UniverseConfig = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(c, back);
        }
    }

    #[test]
    fn every_preset_ecc_cap_is_within_the_kepler_domain() {
        // Fail-loud cross-slice invariant: no preset may cap eccentricity above the fixed Kepler
        // solver's convergence domain (KEPLER_ECC_MAX).
        assert!(UniverseConfig::walk_scale().planet.ecc_cap <= KEPLER_ECC_MAX);
        assert!(UniverseConfig::visual_scale().planet.ecc_cap <= KEPLER_ECC_MAX);
    }

    #[test]
    fn walk_band_builds_the_static_band() {
        UniverseConfig::walk_scale()
            .band
            .build()
            .expect("walk band is valid by construction");
    }

    #[test]
    fn frost_thresholds_reads_the_planet_config() {
        let ft = UniverseConfig::walk_scale().planet.frost_thresholds();
        assert_eq!(ft, FrostThresholds::CANONICAL);
    }

    // ---- D-45(a) Slice 3c: generate -> to_regions lowering (byte-identity) -----------

    #[test]
    fn realm_regions_for_matches_the_frozen_pre_generator_golden() {
        // A HAND-AUTHORED frozen golden (literal values, independent of the generator path): if
        // to_regions drifts any coordinate / shape / parent, this fails against the literals — NOT
        // a self-referential capture of the (rewritten) realm_regions_for output.
        let rs = realm_regions_for(0);
        let expected: [(RealmId, DVec3, Boundary, Option<RealmId>); 7] = [
            (UNIVERSE, DVec3::ZERO, Boundary::Shell { r: 1.0e9 }, None),
            (
                GALAXY,
                DVec3::ZERO,
                Boundary::Shell { r: 180.0 },
                Some(UNIVERSE),
            ),
            (
                SYSTEM_A,
                DVec3::ZERO,
                Boundary::Shell { r: 40.0 },
                Some(GALAXY),
            ),
            (
                PLANET_A,
                DVec3::new(20.0, 0.0, 0.0),
                Boundary::Shell { r: 10.0 },
                Some(SYSTEM_A),
            ),
            (
                SYSTEM_B,
                DVec3::new(130.0, 0.0, 0.0),
                Boundary::Shell { r: 40.0 },
                Some(GALAXY),
            ),
            (
                STATION_A,
                DVec3::new(-25.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::splat(5.0),
                },
                Some(SYSTEM_A),
            ),
            (
                AREA_A,
                // +5 FROM ITS PARENT, the planet at +20 — so still +25 in the system's frame, the same
                // place it has always occupied. The golden used to read 25 here, an absolute on a field
                // that means "offset from my parent"; it went unnoticed while nothing converted between
                // levels. The number changed; the world did not.
                DVec3::new(5.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::splat(3.0),
                },
                Some(PLANET_A),
            ),
        ];
        assert_eq!(rs.len(), 7);
        for (r, (realm, offset, shape, parent)) in rs.iter().zip(expected) {
            assert_eq!(r.realm, realm);
            // The centre is NORMALIZED since the cell activation (the generator is a producer):
            // the VALUE is the golden's literal, exactly — every walk offset is dyadic, so both the
            // normalized form and the flatten are bit-exact.
            assert_eq!(r.center, LatticePos::from_metres(offset, Tier::Fine));
            assert_eq!(r.center.delta_m(LatticePos::ORIGIN, Tier::Fine), offset);
            assert_eq!(r.shape, shape);
            assert_eq!(r.parent, parent);
            assert_eq!(
                r.frame,
                frame_for_realm(realm, parent).expect("canonical frame")
            );
        }
        // The whole forest shares the one static walk band.
        let band = UniverseConfig::walk_scale()
            .band
            .build()
            .expect("walk band");
        for r in &rs {
            assert_eq!(r.band, band);
        }
    }

    #[test]
    fn to_regions_gives_an_orbital_body_a_zero_center_position_authored_by_the_frame() {
        // The Orbital placement arm: a moving body carries NO baked position — its boundary sits at the
        // ZERO origin of its own frame; its live pose is authored per tick into the placement book.
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        let body = GeneratedBody {
            realm: RealmId::Planet(42),
            parent: Some(RealmId::System(42)),
            shape: Boundary::Shell { r: 9.0e8 },
            taxon: None,
            look: Some(Boundary::Shell { r: 9.0e8 }),
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let regions = to_regions(&[body], &UniverseConfig::visual_scale());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].center.cell(), glam::I64Vec3::ZERO);
        assert_eq!(regions[0].center.offset(), DVec3::ZERO);
    }

    #[test]
    fn placement_offset_reads_static_verbatim_and_the_orbital_tick_zero_position() {
        // The frame-local offset utility. `StaticOffset` returns its fixed vector verbatim; `Orbital`
        // returns the tick-0 ephemeris position. `region_center_of` only ever feeds it `StaticOffset`
        // (movers are frame-AUTHORED ⇒ ZERO center, the moving-frame fix), so its `Orbital` arm — the
        // orbital-position fold the S2 own-absolute path reuses — is exercised directly here.
        let v = DVec3::new(3.0, -4.0, 5.0);
        assert_eq!(placement_offset(Placement::StaticOffset(v)), v);
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        assert_eq!(
            placement_offset(Placement::Orbital(elements)),
            orbital_state(&elements, 0.0).position
        );
    }

    #[test]
    fn moving_children_selects_only_direct_orbital_children() {
        // FA-2b: the AUTHORED moving-child roster keeps ONLY a hosted realm's DIRECT children whose
        // placement is `Orbital` — a STATIC child, an orbital NON-child (someone else's), and an
        // orbital GRANDchild are all excluded. Exercises both `orbital_of` arms + the parent filter.
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        let orbital_child = GeneratedBody {
            realm: RealmId::Planet(1),
            parent: Some(RealmId::System(7)),
            shape: Boundary::Shell { r: 9.0e8 },
            taxon: None,
            look: Some(Boundary::Shell { r: 9.0e8 }),
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let static_child = GeneratedBody {
            realm: RealmId::Station(2),
            parent: Some(RealmId::System(7)),
            shape: Boundary::Shell { r: 1.0e6 },
            taxon: None,
            look: Some(Boundary::Shell { r: 1.0e6 }),
            placement: Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
            photometrics: None,
        };
        let orbital_non_child = GeneratedBody {
            realm: RealmId::Planet(3),
            parent: Some(RealmId::System(99)),
            shape: Boundary::Shell { r: 9.0e8 },
            taxon: None,
            look: Some(Boundary::Shell { r: 9.0e8 }),
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let bodies = [orbital_child, static_child, orbital_non_child];
        assert_eq!(
            moving_children(&bodies, RealmId::System(7)),
            vec![(RealmId::Planet(1), elements)],
        );
    }

    #[test]
    fn moving_children_for_is_empty_at_walk_scale() {
        // The byte-identity guarantee: the walk roster is ALL `StaticOffset`, so a walk-scale shard
        // authors NO moving child — `frame_context` registers every region at identity, unchanged.
        assert!(moving_children_for(0, RealmId::System(7)).is_empty());
        assert!(moving_children_for(0, RealmId::System(8)).is_empty());
    }

    // ===== RLM Step 2: per-realm AoI config + generator accessors ========================

    #[test]
    fn interest_config_build_inert_at_zero_factor() {
        let inert = InterestConfig::inert();
        assert!(!inert.is_live());
        assert_eq!(
            inert.build(100.0, 5.0).expect("inert always ok"),
            AoiConfig::inert()
        );
    }

    #[test]
    fn one_world_answers_and_checks_with_the_same_contents() {
        // THE DEFECT THIS TYPE RETIRES, measured. A home used to be RESOLVED against the generated world and
        // then VALIDATED against the hand-placed one. With one star the two happened to agree; with several,
        // the stars a seed draws are simply absent from the hand-placed world, so a perfectly valid home
        // beside any other star fails its own defence — and the gateway panics on the spot.
        let cfg = UniverseConfig::visual_scale();
        let generated = WorldView::generated(0, &cfg);
        let placed = WorldView::hand_placed(&cfg);
        // The two worlds really do hold different things — otherwise the rest of this proves nothing.
        let only_generated: Vec<RealmId> = generated
            .regions()
            .iter()
            .map(|r| r.realm)
            .filter(|realm| !placed.contains_realm(*realm))
            .collect();
        assert!(
            !only_generated.is_empty(),
            "the generated world holds stars the hand-placed one does not"
        );
        // …and EVERY one of them passes the check when the check reads the world that produced it.
        for realm in only_generated {
            assert!(generated.contains_realm(realm));
        }
    }

    #[test]
    fn the_hand_placed_world_is_the_one_with_structures_to_stand_in() {
        // Stations are built by players and areas mostly are, so the generator emits neither — which is why
        // a test that needs one places it. Pinned both ways: the placed world HAS them, the generated world
        // has NONE, and that is the whole reason both exist.
        let cfg = UniverseConfig::walk_scale();
        let placed = WorldView::hand_placed(&cfg);
        let generated = WorldView::generated(0, &cfg);
        let structures = |w: &WorldView| -> usize {
            w.regions()
                .iter()
                .filter(|r| matches!(r.realm, RealmId::Station(_) | RealmId::Area(_)))
                .count()
        };
        assert_eq!(structures(&generated), 0);
        assert_eq!(structures(&placed), 2);
    }

    #[test]
    fn the_world_answers_neighbourhood_from_its_own_contents() {
        // The remaining questions a world is asked, each equal to the free function it delegates to — so
        // holding a world can never mean a different answer than deriving one, only a cheaper one. It was
        // also asked for a realm's ORIGIN CHAIN; that question no longer exists, because no realm is
        // entitled to know where it sits.
        let cfg = UniverseConfig::visual_scale();
        let world = WorldView::generated(0, &cfg);
        let held = std::collections::BTreeSet::from([SYSTEM_A]);
        assert_eq!(
            world.neighbourhood(&held),
            realm_neighbourhood_for_config(0, &held, &cfg)
        );
        assert_eq!(world.regions(), realm_regions_for_config(0, &cfg));
    }

    #[test]
    fn the_lowered_world_is_the_regions_and_nothing_else() {
        // `WorldView::lowered` hands the connection plane the region forest VERBATIM — the same
        // slice `regions()` answers with — and (by its type) nothing a body knows: the lowered
        // value is `vd_core::worldgen::WorldRealms`, a crate with no path back to an orbit (SL4).
        let cfg = UniverseConfig::visual_scale();
        let world = WorldView::generated(0, &cfg);
        assert_eq!(world.lowered().regions(), world.regions());
    }

    #[test]
    fn no_two_static_siblings_ever_overlap_in_any_shipped_world() {
        // THE FENCE, RUN. A position inside two overlapping siblings has two equally valid owners, and
        // which shard gets you falls out of iteration order — see `siblings_disjoint` for why that is
        // unrepairable downstream. The generator is deterministic, so proving it over the shipped presets
        // and a spread of seeds proves the worlds that can actually be booted.
        //
        // Seeds swept rather than one sampled: the star ring is drawn from the galaxy's own stream, so a
        // seed that happened to draw a crowded galaxy is exactly the case a single-seed test would miss.
        for seed in 0..64_u64 {
            for config in [
                UniverseConfig::visual_scale(),
                UniverseConfig::visual_demand(15.0, 0.02),
            ] {
                let bodies = generate_system_forest(seed, &config);
                assert_eq!(siblings_disjoint(&bodies), Ok(()));
            }
            assert_eq!(
                siblings_disjoint(&generate_walk_forest(&UniverseConfig::walk_scale())),
                Ok(())
            );
        }
    }

    #[test]
    fn the_fence_refuses_two_siblings_that_reach_each_other() {
        // The fence's OWN failing case, so passing above is a fact and not a fence that never says no.
        // Two shells of radius 1 whose centres are 1.5 apart: their surfaces interpenetrate.
        let shell = Boundary::Shell { r: 1.0 };
        let at = |x: f64| Placement::StaticOffset(DVec3::new(x, 0.0, 0.0));
        let body = |realm, placement| GeneratedBody {
            realm,
            parent: Some(GALAXY),
            shape: shell,
            taxon: None,
            look: Some(shell),
            placement,
            photometrics: None,
        };
        let a = RealmId::System(1);
        let b = RealmId::System(2);
        assert_eq!(
            siblings_disjoint(&[body(a, at(0.0)), body(b, at(1.5))]),
            Err(SiblingsOverlap {
                a,
                b,
                parent: GALAXY
            })
        );
        // …and clears once they are pushed apart past the sum of their radii.
        assert_eq!(
            siblings_disjoint(&[body(a, at(0.0)), body(b, at(2.5))]),
            Ok(())
        );
    }

    #[test]
    fn the_first_system_draws_the_stream_its_realm_path_names() {
        // HR1 restated as a measurement: every shard hosting a system must draw the IDENTICAL per-system
        // stream, which holds only if the stream's lineage IS the realm's path from the universe down.
        // Pinned on system zero because that is the one with a named seed to compare against.
        assert_eq!(
            [UNIVERSE_SEED, GALAXY_SEED, system_seed_at(0)],
            SYSTEM_A_LINEAGE
        );
    }

    #[test]
    fn guard_wake_covers_visibility_every_body_wakes_before_it_is_visible() {
        // THE SUCCESSOR of the interim "a planet is visible from anywhere inside its own system"
        // pin (a tiny-world artifact: at true scale a 3,400 km planet across a 1.6e11 m system is
        // genuinely below the visibility angle — that IS the real sky). The lawful invariant at
        // any scale is the taxonomy design's §4.5.6 gauge: the AoI wake trigger reads the BOUND
        // (`spin_up_factor · bound`), the picture needs `cot(θ/2) · look` — so the wake is early
        // (safe) exactly when `spin_up ≥ look·cot(θ/2)` for every drawable body. The minimum
        // ratio is printed; the gauge fails the day a body's look outgrows its bound's wake.
        let regions = realm_regions_for_config(0, &UniverseConfig::visual_scale());
        let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let mut planets_checked = 0_u32;
        let mut min_ratio = f64::INFINITY;
        for r in regions.iter().filter(|r| r.parent.is_some()) {
            let Some(look) = r.look else { continue };
            let needed = look.finite_extent() * factor;
            let wake = r.aoi.spin_up_r_m();
            assert!(
                wake >= needed,
                "{:?} wakes at {wake} m but is visible from {needed} m",
                r.realm
            );
            min_ratio = min_ratio.min(wake / needed);
            if matches!(r.realm, RealmId::Planet(_)) {
                planets_checked += 1;
            }
        }
        println!("[guard_wake_covers_visibility] min wake/needed ratio = {min_ratio}");
        // …and the loop actually ran over the full Planet-kind roster (27 planets + 6 moons).
        assert_eq!(planets_checked, 33);
    }

    #[test]
    fn interest_config_build_live() {
        let live = UniverseConfig::visual_scale().interest;
        assert!(live.is_live());
        // spin_up_r = extent × the ONE visibility factor cot(θ/2) (v_child 0 ⇒ tear = spin_up widened by
        // the occupant-speed lead, since spin_up_factor == tear_down_factor collapses the geometric gap).
        let band = live.build(100.0, 0.0).expect("valid live");
        assert_eq!(
            band.spin_up_r_m(),
            100.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD)
        );
    }

    #[test]
    fn to_regions_stamps_per_realm_aoi() {
        // Walk: every region inert (byte-identity — the field never changes containment).
        for r in realm_regions_for(0) {
            assert_eq!(r.aoi, AoiConfig::inert());
        }
        // Visual: a Planet's spin_up = its own extent × the visibility factor; a bigger realm (System)
        // reaches farther (same factor, larger extent).
        let visual = realm_regions_for_config(0, &UniverseConfig::visual_scale());
        let planet = visual
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("a planet");
        assert_eq!(
            planet.aoi.spin_up_r_m(),
            planet.shape.finite_extent() * visibility_factor(VISIBILITY_THETA_MIN_RAD)
        );
        let system = visual
            .iter()
            .find(|r| r.realm == SYSTEM_A)
            .expect("system A");
        assert!(system.aoi.spin_up_r_m() > planet.aoi.spin_up_r_m());
    }

    // ===== RLM 5f-4a: the walk-demand LIVE AoI band over the walk forest =================
    // The dev demand-cluster's live AoI dynamics: a 2 m/s brisk walk (move_speed × time_multiplier),
    // 50 Hz (0.02 s/tick). Kept here (test-only) — the composer supplies the live cluster values at boot.
    fn walk_demand_regions() -> Vec<RealmRegion> {
        realm_regions_for_walk_config(0, &UniverseConfig::walk_demand(2.0, 0.02))
    }

    #[test]
    fn walk_keeps_aoi_inert_while_visual_stays_live() {
        // Byte-identity: walk keeps AoI OFF (behaviour unchanged); the compressed-real visual
        // band is LIVE under the ONE cot(θ/2) visibility factor.
        for r in realm_regions_for(0) {
            assert_eq!(r.aoi, AoiConfig::inert(), "walk regions stay AoI-inert");
        }
        assert!(
            UniverseConfig::visual_scale().interest.is_live(),
            "the visual band is live under the visibility factor"
        );
    }

    #[test]
    fn walk_demand_differs_from_walk_only_in_aoi() {
        let walk = realm_regions_for_walk_config(0, &UniverseConfig::walk_scale());
        let demand = walk_demand_regions();
        assert_eq!(walk.len(), demand.len(), "same forest topology");
        for (w, d) in walk.iter().zip(demand.iter()) {
            assert_eq!(w.realm, d.realm, "realm unchanged");
            assert_eq!(w.center, d.center, "center unchanged");
            assert_eq!(w.frame, d.frame, "frame unchanged");
            assert_eq!(w.shape, d.shape, "shape unchanged");
            assert_eq!(w.band, d.band, "containment band unchanged");
            assert_eq!(w.parent, d.parent, "parent unchanged");
            assert_eq!(w.aoi, AoiConfig::inert(), "walk region is AoI-inert");
            assert_ne!(
                d.aoi,
                AoiConfig::inert(),
                "walk-demand region has a LIVE AoI band"
            );
        }
    }

    #[test]
    fn walk_config_builders_delegate_byte_identically() {
        // The existing fns delegate to the config builders with walk_scale ⇒ byte-identical output.
        assert_eq!(
            realm_regions_for(0),
            realm_regions_for_walk_config(0, &UniverseConfig::walk_scale()),
            "realm_regions_for delegates byte-identically"
        );
        let all = realm_regions_for(0);
        let a_planet = all
            .iter()
            .find_map(|r| matches!(r.realm, RealmId::Planet(_)).then_some(r.realm))
            .expect("a planet in the walk forest");
        let sets = [
            std::collections::BTreeSet::from([SYSTEM_A]),
            std::collections::BTreeSet::from([a_planet]),
            std::collections::BTreeSet::from([SYSTEM_A, a_planet]),
        ];
        // The FIXTURE scoper agrees with its own single-realm twin over the same roster. It deliberately
        // does NOT agree with the config scoper any more: that one reads the GENERATED world, which holds
        // no player-built station or area. Asserting they match is what let a fixture masquerade as the
        // world in the first place.
        for held in &sets {
            let scoped = realm_neighbourhood_for_held(0, held);
            for r in held {
                for one in realm_neighbourhood_for(0, *r) {
                    assert!(
                        scoped.iter().any(|s| s.realm == one.realm),
                        "the union over {held:?} covers every member's own neighbourhood"
                    );
                }
            }
        }
    }

    #[test]
    fn direct_child_levels_from_seed() {
        // Visual System A hosts N orbiting planets; the roster is exactly those planet levels.
        let config = UniverseConfig::visual_scale();
        let levels = direct_child_levels(0, &config, SYSTEM_A);
        assert_eq!(
            levels.len(),
            world_n_planets() as usize + 1,
            "9 planets + the star (T2)"
        );
        assert_eq!(
            levels
                .iter()
                .filter(|l| l.kind == RealmKindTag::Planet)
                .count(),
            world_n_planets() as usize
        );
        assert_eq!(
            levels
                .iter()
                .filter(|l| l.kind == RealmKindTag::Star)
                .count(),
            1
        );
    }

    /// The visual-scale system forest at seed 0 (helper for the tests below).
    fn visual_forest() -> Vec<GeneratedBody> {
        generate_system_forest(0, &UniverseConfig::visual_scale())
    }

    // (The compressed-visual derive helpers `vis_planet_soi`/`vis_outer_sma`/`vis_central_mass`
    // died with the re-solve; the world's own bodies state their derived values now.)

    // ===== RLM realistic-demo Slice 0: the one visibility constant + compressed-real geometry =========

    /// The single most-distant planet region + its epoch ORBIT DISTANCE — the OUTER planet, the one the
    /// star-view visibility rule culls until an occupant closes in. A moving realm's region.center is ZERO
    /// (position authored via the frame), so the orbit distance is derived from the mover ELEMENTS, not the
    /// region center.
    fn outer_planet_orbit(config: &UniverseConfig) -> (RealmRegion, f64) {
        let regions = realm_regions_for_config(0, config);
        moving_children_for_config(0, config, SYSTEM_A)
            .iter()
            .map(|(realm, el)| {
                let region = *regions
                    .iter()
                    .find(|r| r.realm == *realm)
                    .expect("every mover has a region");
                (region, orbital_state(el, 0.0).position.length())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("a planet mover is present")
    }

    /// FINDING 28's measurement: THE world produces `Placement::Orbital` in production — at least one
    /// per system, through the same `UniverseConfig::world` every shipped shard boots. The claim used
    /// to live as a doc comment saying the opposite ("no production producer") under an
    /// `#[allow(dead_code)]`; a gate can be wrong but it cannot be stale.
    #[test]
    fn the_world_produces_an_orbital_planet_in_every_system() {
        let config = UniverseConfig::world(15.0, 0.02);
        let bodies = generate_system_forest(0, &config);
        let systems: Vec<RealmId> = bodies
            .iter()
            .filter(|b| {
                matches!(b.realm, RealmId::System(_))
                    && bodies
                        .iter()
                        .any(|c| c.parent == Some(b.realm) && orbital_of(c.placement).is_some())
            })
            .map(|b| b.realm)
            .collect();
        let stars: Vec<RealmId> = bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
            .map(|b| b.realm)
            .collect();
        assert_eq!(
            systems, stars,
            "every star system of THE world holds at least one Orbital planet"
        );
        assert!(!stars.is_empty(), "THE world holds star systems");
        // …and T3's moon-hosting planets hold Orbital children of their own (the same shape
        // one level down — a moon is a planet under a planet).
        let moon_hosts = bodies
            .iter()
            .filter(|b| {
                matches!(b.realm, RealmId::Planet(_))
                    && bodies
                        .iter()
                        .any(|c| c.parent == Some(b.realm) && orbital_of(c.placement).is_some())
            })
            .count();
        assert_eq!(moon_hosts, 5, "the census's five moon-hosting planets");
    }

    #[test]
    fn visibility_factor_is_cot_half_theta() {
        // cot(θ/2) at θ_min = 1.5° — the ONE visibility constant (≈ 76.390), frozen non-self-referentially.
        assert_eq!(
            visibility_factor(VISIBILITY_THETA_MIN_RAD),
            FROZEN_VISIBILITY_FACTOR
        );
    }

    // ===== The generator visibility check (owner ruling 2026-08-15, item 5 + the re-solve addendum) ==

    /// THE MEASUREMENT ON THE WORLD, pinned verbatim — the two-level bound HOLDS after the
    /// 2026-08-15 shell solve.
    ///
    /// HISTORY (the failing measurement this green pin replaces, kept as provenance). Before the
    /// solve the shell was containment-only (`ring + 2·system_soi = 12_331.398_328_646_887 m`) and
    /// THE world (seed 0) MEASURED FAILING on 2026-08-15: every planet of BOTH ring-placed systems
    /// stayed visible from just outside the galaxy — a 3.954_173_752_999_557_8 m planet visible out
    /// to 302.058_663_384_242_95 m while the shell passed within d_min 161.127_605_697_744_3 …
    /// 280.730_889_197_954 m of the ten ring planets' worst-instant positions (worst_dist
    /// 12_046.713_265_695_933 … 12_166.316_549_196_143 m; the origin system's planets passed). The
    /// owner ruled (addendum to items 5/10): the generator SOLVES the margin as a general
    /// constraint — [`galaxy_shell_r_m`] grows the shell by the worst descendant's two-level
    /// clearance; the ring could not move inward because it already sits at the wake law's LOWER
    /// bound. The exact pre-solve offence stays a live measurement in
    /// `the_guard_refuses_a_shell_that_hugs_its_ring`, which restores the old shell and pins the
    /// first offence verbatim.
    #[test]
    fn the_two_level_bound_re_solved_on_the_world_no_body_is_visible_past_any_two_level_ancestor() {
        let config = UniverseConfig::world(15.0, 0.05);
        // The storage-fence chain's shell, frozen non-self-referentially (real-scale addendum
        // §A2.2 — the interim upward solve retired; the two-level bound below now holds with ~9
        // orders of slack at the 0.2377 ly gap instead of the interim 4 m margin).
        assert_eq!(config.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
        let pairs = grandchild_visibility_pairs(
            &generate_system_forest(0, &config),
            VISIBILITY_THETA_MIN_RAD,
        );
        // Non-vacuity: every planet AND every star is judged against its galaxy AND the
        // universe (30 + 30), every system against the universe (3), and every MOON against
        // its system, galaxy and universe (3 × 7) — the walk really visited every two-level
        // pair that carries a picture (the look-less ambients are not subjects).
        assert_eq!(pairs.len(), 81);
        // The WORST margin across every pair of THE world — a ring system's planet against the
        // galaxy shell. At the 0.2377 ly gap the margin IS the reserved clearance class
        // (~3.069e11 m — the §A2.2 clearance showing through, where the interim world measured
        // 11.128 m). Pinned EXACTLY as measured so any re-solve flips this loudly.
        let worst_margin_m = pairs
            .iter()
            .map(|p| p.d_min_m - p.required_m)
            .fold(f64::INFINITY, f64::min);
        println!("[two-level] worst margin = {worst_margin_m}");
        assert!(
            worst_margin_m > 0.0,
            "no body is visible past any two-level ancestor"
        );
        // …and the check the boot fence's predecessor ran: no offence anywhere in THE world —
        // now expressed through the MEASURED climb (look_horizon slice 2): the fence passes at
        // the landed carrier's arity and refuses one below it (both arms driven).
        assert_eq!(
            grandchild_visibility_offences(
                &generate_system_forest(0, &config),
                VISIBILITY_THETA_MIN_RAD,
            ),
            vec![]
        );
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
        // With the look split the whole world fits even a one-level carrier (max climb 1);
        // the fence's refusal arm stays covered by the hugging-shell test below.
        assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
    }

    /// The refusal the boot fence makes, measured on the exact PRE-SOLVE geometry: restoring the
    /// containment-only shell (`ring + 2·system_soi` — the world as measured failing 2026-08-15)
    /// makes the guard name the FIRST ring planet with the very numbers of that measurement —
    /// history kept live, and the guard's `Err` arm covered on the boot-facing wrapper.
    #[test]
    fn the_guard_refuses_a_shell_that_hugs_its_ring() {
        let mut config = UniverseConfig::world(15.0, 0.05);
        config.scale.galaxy_r_m =
            config.stellar.system_ring_r_m + 2.0 * config.stellar.system_soi_r_m;
        // The hugging-shell offence at the REAL-SCALE placement radius, pinned verbatim (the
        // INTERIM-SCALE record — worst_dist 12_046.713_265_695_933 m, d_min 280.730_889_197_954 m
        // on the 12 031 m ring — is kept here as history per the re-solve's provenance rule; the
        // live measurement below is the same first ring planet on the 0.2377 ly placement).
        let offences = grandchild_visibility_offences(
            &generate_system_forest(0, &config),
            VISIBILITY_THETA_MIN_RAD,
        );
        assert_eq!(
            offences.first().copied(),
            Some(GrandchildVisibleOutside {
                body: RealmId::Planet(2790672799213891506),
                ancestor: GALAXY,
                worst_dist_m: 2_248_492_745_656_386.8,
                extent_m: 19_349_648.343_888_18,
                d_min_m: -2_261_384_776.593_888_3,
                required_m: 1_478_116_360.282_928_5,
            })
        );
        // …and the BOOT-facing fence (look_horizon slice 2 — the climb measurement): under the
        // hugging shell a ring SYSTEM's star stays visible from outside the whole galaxy, so
        // its picture must travel TWO levels — more than a one-level carrier holds. (At the
        // true-size world a PLANET's climb stops at its own solved system shell regardless of
        // the galaxy — the §3.2 identity doing its job — so the two-level subject under a
        // hugging shell is the system itself.) The landed arity-2 carrier still serves the
        // hugging world; the refusal arm is driven one level down.
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
        let refused = guard_visibility_climb_bounded(0, &config, 1)
            .expect_err("a hugging shell exceeds a one-level carrier");
        eprintln!("[climb] the hugging-shell refusal, verbatim: {refused}");
        assert_eq!(refused.levels, 2);
        assert_eq!(refused.top, GALAXY, "visible from outside even the galaxy");
        assert_eq!(refused.arity, 1);
        assert!(matches!(refused.body, RealmId::System(_)));
    }

    /// G-CLIMB, RE-DERIVED at the IN-SYSTEM TRUE-SIZE RE-SOLVE (the taxonomy arc's flag day —
    /// a proof rewrite, never a weakening): with the BOUND/LOOK split landed, THE world's
    /// measured visibility climb is **max 1 over all 30 bodies** — the design's prediction
    /// (real-scale addendum §A2.5; celestial_taxonomy_design §4.5.1), now a measurement:
    /// - every body's picture is its LOOK (a planet's derived radius, a system's star), and the
    ///   §3.2 clearance identity gives every child a strictly positive stopping slack at its
    ///   OWN parent — levels 1, top == self, everywhere;
    /// - the ambient galaxy/universe carry `look = None` — no picture, no climb (the outer
    ///   re-solve's named levels-2 root artifact DISSOLVES here, exactly as its ledger said
    ///   it would the day `look: None` landed on the ambients).
    #[test]
    fn g_climb_the_worlds_measured_climb_at_the_true_size_resolve() {
        let config = UniverseConfig::world(15.0, 0.05);
        let climbs = measure_visibility_climb(0, &config);
        // One climb per LOOK-carrying parented body: 3 systems + 27 planets + 3 stars + the
        // 6 census moons (T3; the ambient galaxy/universe draw nothing; the universe is the
        // root).
        assert_eq!(climbs.len(), 39);
        assert_eq!(climbs.iter().map(|c| c.levels).max(), Some(1));
        for c in &climbs {
            assert_eq!(c.levels, 1, "{c:?}");
            assert_eq!(c.top, c.body, "{c:?}");
            assert!(c.slack_m > 0.0, "{c:?}");
        }
        // THE RESERVED-CLEARANCE IDENTITY, system half: a ring system's stopping slack ==
        // (R_gal − placement) − R★·(1 + cot(θ/2)) — the reserved clearance showing through.
        // For the LARGEST star that is TARGET_SYSTEM_BOUND_MAX_M exactly (§A2.5's cancellation).
        let clearance_m = config.scale.galaxy_r_m - config.stellar.system_ring_r_m;
        let factor_plus_one = 1.0 + FROZEN_VISIBILITY_FACTOR;
        let bodies = generate_system_forest(0, &config);
        let mut ring_systems = 0u32;
        for c in climbs
            .iter()
            .filter(|c| matches!(c.body, RealmId::System(_)))
        {
            let body = bodies
                .iter()
                .find(|b| b.realm == c.body)
                .expect("a climbed body is in the forest");
            let look_m = body.look.expect("a system draws its star").finite_extent();
            let placement_m = placement_offset(body.placement).length();
            if placement_m == 0.0 {
                continue; // the home anchor's slack is the whole galaxy, not the clearance class
            }
            ring_systems += 1;
            let identity_m = clearance_m - look_m * factor_plus_one;
            // 1 m association tolerance at 2.25e15 magnitudes (a few ulp per f64 association).
            assert!(
                (c.slack_m - identity_m).abs() < 1.0,
                "the reserved-clearance identity: {c:?} vs {identity_m}"
            );
        }
        assert_eq!(ring_systems, 2, "both seeded siblings were judged");
        let largest_slack_m = climbs
            .iter()
            .filter(|c| matches!(c.body, RealmId::System(_)))
            .map(|c| c.slack_m)
            .fold(f64::INFINITY, f64::min);
        assert!(
            (largest_slack_m - TARGET_SYSTEM_BOUND_MAX_M).abs() < 1.0,
            "the largest star's ring slack IS the reserved system bound (§A2.5):              {largest_slack_m} vs {TARGET_SYSTEM_BOUND_MAX_M}"
        );
        // THE WORST TRUE-PLANET STOPPING SLACK: strictly positive by the per-rung solve;
        // printed and floor-pinned at the solve's own reserved clearance class (> 1e10 m on
        // THE world). MOONS (Planet-kind under a planet, T3) are printed separately — their
        // slack lives at the PLANET's scale, and its floor is its own gate below.
        let is_moon = |realm: RealmId| {
            bodies
                .iter()
                .find(|b| b.realm == realm)
                .and_then(|b| b.parent)
                .is_some_and(|p| matches!(p, RealmId::Planet(_)))
        };
        let planet_slack_m = climbs
            .iter()
            .filter(|c| matches!(c.body, RealmId::Planet(_)) && !is_moon(c.body))
            .map(|c| c.slack_m)
            .fold(f64::INFINITY, f64::min);
        let moon_slack_m = climbs
            .iter()
            .filter(|c| matches!(c.body, RealmId::Planet(_)) && is_moon(c.body))
            .map(|c| c.slack_m)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "[G-CLIMB] max 1 over {} bodies; worst planet slack {planet_slack_m} m; worst MOON \
             slack {moon_slack_m} m; ring system slack {largest_slack_m} m (reserved clearance \
             {clearance_m} m)",
            climbs.len()
        );
        assert!(planet_slack_m > 1.0e10);
        assert!(
            moon_slack_m > 0.0,
            "every moon stops at its own planet with slack"
        );
        // The four-number pins, re-asserted here so G-CLIMB stays self-contained.
        assert_eq!(config.scale.universe_r_m, FROZEN_REAL_UNIVERSE_R_M);
        assert_eq!(config.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
        assert_eq!(config.stellar.system_ring_r_m, FROZEN_REAL_PLACEMENT_R_M);
        // …and the fence at the landed carrier's arity: passes at 2 with a FULL SPARE LEVEL —
        // and even at arity 1 now (both fences green is itself the flag-day measurement; the
        // refusal arm stays covered by the hugging-shell test below).
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
        assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
    }

    #[test]
    fn measure_visibility_climb_the_ordered_first_measurement() {
        let config = UniverseConfig::world(15.0, 0.05);
        let climbs = measure_visibility_climb(0, &config);
        eprintln!("[CLIMB — THE ORDERED FIRST MEASUREMENT, verbatim]");
        for c in &climbs {
            eprintln!(
                "  body={:?} top={:?} levels={} slack_m={}",
                c.body, c.top, c.levels, c.slack_m
            );
        }
        let max_levels = climbs.iter().map(|c| c.levels).max();
        eprintln!("  MAX LEVELS = {max_levels:?} over {} bodies", climbs.len());
        // The landed carrier still bounds it (arity 2) — the boot fence's condition.
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    }

    /// ▲ THE FOUR OUTER GEOMETRY NUMBERS, pinned bit-for-bit against the addendum's derivations
    /// (real-scale addendum §A2.2/§A2.3) — plus the reserved-clearance identity and the storage
    /// budget's exact occupancy/headroom, each a measurement that could have failed.
    #[test]
    fn the_four_outer_geometry_numbers_are_the_addendums_derivations() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        // ▲ 1 the universe: 2⁵¹ m exactly (the storage fence's solve).
        assert_eq!(cfg.scale.universe_r_m, FROZEN_REAL_UNIVERSE_R_M);
        assert_eq!(cfg.scale.universe_r_m, (1u64 << 51) as f64);
        // ▲ 2 the galaxy: R_uni − the τ-free outset (the CORRECTED formula, addendum §A2.2 —
        // measured equal to the addendum's printed 2.2487974139336678e15 m).
        assert_eq!(cfg.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
        let outset_m = (2.0 * FROZEN_REAL_UNIVERSE_R_M / T_TRAVERSE_S)
            * GEOMETRY_TICK_DT_S
            * BAND_TICKS_N
            * BAND_TAU_HEADROOM;
        assert_eq!(cfg.scale.galaxy_r_m, FROZEN_REAL_UNIVERSE_R_M - outset_m);
        // ▲ 3 the placement radius: R_gal − the reserved clearance (the owner-ruled star gap,
        // 0.2376656 ly; RE-MEASURED at the flag day — the reservation now covers the SOLVED
        // shells, superseding the addendum's 2.248490504408914e15 by −7.88e5 m).
        assert_eq!(cfg.stellar.system_ring_r_m, FROZEN_REAL_PLACEMENT_R_M);
        let clearance_m = child_clearance_m(
            TARGET_SYSTEM_BOUND_MAX_M,
            TARGET_STAR_LOOK_MAX_M,
            VISIBILITY_THETA_MIN_RAD,
        );
        assert_eq!(
            cfg.stellar.system_ring_r_m,
            cfg.scale.galaxy_r_m - clearance_m
        );
        // …and the clearance itself equals the addendum's printed 3.069095247536e11 m class.
        assert_eq!(
            clearance_m, 306_910_312_489.256_4,
            "the reserved clearance, re-measured at the flag day (covers the SOLVED shells)"
        );
        // ▲ 4 the compression: χ = real mean NN separation / placement radius = 16.378×.
        assert_eq!(real_compression_chi(), FROZEN_REAL_COMPRESSION_CHI);
        // The storage budget, EXACT: occupancy 50.0000 %, headroom 2.0000× — the equality is the
        // construction (2 × 2⁶¹ == CELL_DOMAIN_MAX + 1), never slack.
        let budget = guard_root_representable(&cfg).expect("THE world is representable");
        assert_eq!(budget.occupancy, 0.5);
        assert_eq!(budget.headroom, 2.0);
        eprintln!(
            "[GEOMETRY] universe {} m | galaxy {} m | placement {} m (0.2376656 ly) | chi {} | \
             occupancy {:.4}% headroom {:.4}x",
            cfg.scale.universe_r_m,
            cfg.scale.galaxy_r_m,
            cfg.stellar.system_ring_r_m,
            real_compression_chi(),
            100.0 * budget.occupancy,
            budget.headroom,
        );
    }

    /// The storage fence's REFUSAL ARM, driven (addendum §A6.3's `g_root_representable`: "without
    /// the refusal arm the fence is untested"): a synthetic root ONE OCTAVE larger is refused with
    /// its numbers — THE NAMED P10 TRIGGER (R3) — and the error text names the cure.
    #[test]
    fn guard_root_representable_refuses_the_next_octave_naming_p10() {
        let mut cfg = UniverseConfig::world(15.0, 0.05);
        cfg.scale.universe_r_m = 2.0 * FROZEN_REAL_UNIVERSE_R_M;
        let refused = guard_root_representable(&cfg).expect_err("one octave up must refuse");
        assert_eq!(refused.root_r_m, 2.0 * FROZEN_REAL_UNIVERSE_R_M);
        assert_eq!(refused.k_span, K_SPAN);
        assert_eq!(refused.occupancy_pct, 100.0);
        assert!(refused.to_string().contains("galaxy cell lattice (P10)"));
    }

    /// ★ DISCOVERY-PERMANENCE, measured (the module-doc law's pin): the 3-D placement pair is
    /// APPENDED at per-system stream positions 32–33 — the first 31 draws are byte-identical to
    /// the pre-placement stream (the planet elements, the star, the albedos — their own pins
    /// stand beside this), and the NEXT TWO draws are exactly the pair the shipped placement law
    /// consumed. A reorder fails here before it can re-roll a world.
    #[test]
    fn the_placement_draws_are_appended_after_the_albedo_pass_and_shift_nothing() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        for (ix, seed) in [
            (0u32, SYSTEM_A_SEED),
            (1, system_seed_at(1)),
            (2, system_seed_at(2)),
        ] {
            let mut stream = realm_stream(0, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
            // Consume the pre-placement prefix exactly as the generator draws it: 5 planets × 5
            // element draws, 1 star draw, 5 albedo draws = 31 draws.
            for _ in 0..(5 * 5 + 1 + 5) {
                let _ = stream.next_f64();
            }
            // Draws 32–33 ARE the placement direction pair.
            let want = system_center_at(&cfg, ix, stream.next_f64(), stream.next_f64());
            let got = bodies
                .iter()
                .find(|b| b.realm == RealmId::System(seed))
                .map(|b| placement_offset(b.placement))
                .expect("every system is in the forest");
            assert_eq!(
                got, want,
                "system index {ix}: the placement pair is draws 32-33"
            );
        }
    }

    /// ★ THE 3-D SEEDED PLACEMENTS (owner ruling Q-B) and the RE-DERIVED SEPARATION FENCE,
    /// measured on THE world: the home anchored at the origin; both siblings at EXACTLY the
    /// placement radius in genuinely three-dimensional directions (out of the old ring's y = 0
    /// plane, and NOT collinear — the pair's chord differs from both the sum and difference of
    /// their radii); every pair disjoint by the 3-D fence, with the wake half (asleep at
    /// departure) measured beside it; and the fence's refusal arm driven on a synthetic overlap.
    #[test]
    fn the_seeded_placements_are_three_dimensional_and_the_fence_judges_the_point_set() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let centres: Vec<(RealmId, DVec3, f64)> = generate_system_forest(0, &cfg)
            .iter()
            .filter(|b| b.parent == Some(GALAXY))
            .map(|b| {
                (
                    b.realm,
                    placement_offset(b.placement),
                    b.shape.circumscribed_extent(),
                )
            })
            .collect();
        assert_eq!(
            centres.len(),
            3,
            "the census still says three (Q-B: count unchanged)"
        );
        // The home anchor.
        assert_eq!(centres[0].0, SYSTEM_A);
        assert_eq!(centres[0].1, DVec3::ZERO * -1.0);
        // Both siblings at exactly the placement radius…
        // |unit·r| re-associates once per component; a 2-ulp bound at this magnitude is the
        // float-identity claim "AT the placement radius", not a loosened one.
        let ulp_m = FROZEN_REAL_PLACEMENT_R_M * f64::EPSILON;
        assert!((centres[1].1.length() - FROZEN_REAL_PLACEMENT_R_M).abs() <= 2.0 * ulp_m);
        assert!((centres[2].1.length() - FROZEN_REAL_PLACEMENT_R_M).abs() <= 2.0 * ulp_m);
        // …in genuinely 3-D directions: off the retired ring's plane…
        assert!(centres[1].1.y.abs() > 0.0);
        assert!(centres[2].1.y.abs() > 0.0);
        // …and NOT collinear: the sibling chord is neither 2r (diametric) nor 0 (coincident).
        let chord = (centres[2].1 - centres[1].1).length();
        assert!(chord > 0.0);
        assert!((chord - 2.0 * FROZEN_REAL_PLACEMENT_R_M).abs() > 1.0e12);
        eprintln!(
            "[3D PLACEMENTS] sibling 1 {:?}; sibling 2 {:?}; chord {chord} m",
            centres[1].1, centres[2].1
        );
        // The fence over the real point set…
        assert_eq!(guard_seeded_systems_disjoint(0, &cfg), Ok(()));
        // …the wake half: every pair separated by far more than a system's spin-up reach, so a
        // system is ASLEEP at departure from any sibling (the interim ring's second fence,
        // re-measured on the seeded set).
        let spin_up_m = cfg
            .interest
            .build(cfg.stellar.system_soi_r_m, 0.0)
            .expect("a live system band builds")
            .spin_up_r_m();
        for (i, (_, a, _)) in centres.iter().enumerate() {
            for (_, b, _) in centres.iter().skip(i + 1) {
                assert!((*b - *a).length() > spin_up_m);
            }
        }
        // …and the refusal arm, driven: two synthetic siblings closer than their extents.
        let overlap = vec![
            (SYSTEM_A, DVec3::ZERO, 150.0),
            (RealmId::System(99), DVec3::new(200.0, 0.0, 0.0), 150.0),
        ];
        assert_eq!(
            seeded_systems_disjoint_3d(&overlap),
            Err(SiblingsOverlap {
                a: SYSTEM_A,
                b: RealmId::System(99),
                parent: GALAXY,
            })
        );
    }

    /// ★ REALM EXTENT = GRAVITY SOI (owner ruling 2026-08-18, D-REAL-1) — FLIPPED from
    /// blocker-measured to EQUALITY-PINNED by the taxonomy arc's in-system re-solve: every
    /// planet shell of THE world now EQUALS `celestial::planet_soi` at that planet's DRAWN
    /// mass around its star's REAL drawn mass. The half-worst-instant-gap clamp is the
    /// inert-but-live second arm: MEASURED never to bind on any lawful draw (the fence below
    /// prints the closest ratio and fails if it ever does silently).
    #[test]
    fn realm_shell_equals_the_gravitational_soi_at_the_drawn_mass_d_real_1() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        // The parent's REAL mass: a planet's is its star's drawn mass; a MOON's is its parent
        // PLANET's drawn mass (T3 — the same one law, different arguments, kind-blind).
        let parent_mass_kg = |parent: Option<RealmId>| -> f64 {
            let p = bodies
                .iter()
                .find(|b| Some(b.realm) == parent)
                .expect("every taxon-bearing body has a rostered parent");
            p.taxon.map_or_else(
                || {
                    p.photometrics
                        .expect("a system carries its star draw")
                        .mass_msun
                        * crate::taxonomy::M_SUN_KG
                },
                |t| t.mass_kg,
            )
        };
        let mut planets = 0usize;
        let mut worst_clamp_ratio = 0.0_f64;
        for b in bodies.iter().filter(|b| b.taxon.is_some()) {
            let taxon = b.taxon.expect("filtered on presence");
            let central_kg = parent_mass_kg(b.parent);
            let elements = orbital_of(b.placement).expect("a generated planet is Orbital");
            let soi = crate::celestial::planet_soi(elements.sma, taxon.mass_kg, central_kg);
            // THE EQUALITY (bit-for-bit): the emitted shell IS the gravitational SOI.
            assert_eq!(
                b.shape,
                Boundary::Shell { r: soi },
                "{:?}: realm extent == gravitational SOI (D-REAL-1)",
                b.realm
            );
            // The drawn mass round-trips the central mass law (a moon orbits its planet's
            // REAL drawn mass; a planet its star's).
            assert_eq!(elements.central_mass, central_kg);
            // The clamp arm's margin: soi against the half-worst-instant gap it is min'd with.
            let ratio = soi / b.shape.finite_extent();
            worst_clamp_ratio = worst_clamp_ratio.max(ratio);
            planets += 1;
        }
        assert_eq!(
            planets, 33,
            "every planet AND every moon of THE world was judged (27 + 6)"
        );
        // The clamp never bound: shell == unclamped soi everywhere (ratio exactly 1.0), so the
        // min's second arm is inert-but-live on THE world — printed, fenced.
        println!("[d-real-1] worst soi/shell ratio = {worst_clamp_ratio}");
        assert_eq!(worst_clamp_ratio, 1.0);
    }

    #[test]
    fn plant_seed_of_reads_seed_lineage_kinds_and_refuses_the_rest() {
        // The plant field's resting state IS "nothing built" (the serde default and every
        // shipped constructor agree).
        assert_eq!(FixturePlant::default(), FixturePlant::None);
        assert_eq!(plant_seed_of(RealmId::System(7)), Some(7));
        assert_eq!(plant_seed_of(RealmId::Planet(9)), Some(9));
        assert_eq!(plant_seed_of(RealmId::Station(3)), None);
        assert_eq!(plant_seed_of(RealmId::Area(4)), None);
        assert_eq!(
            plant_seed_of(RealmId::Ship(vd_core::ids::EntityId(1))),
            None
        );
    }

    /// look_horizon.md slice 5 — the fixture plant is ADDITIVE: with the plant selected, every
    /// generated body (id, orbit, photometric draw, order) is byte-identical to the plant-free
    /// world, and exactly the named pair is appended after them. The plain world carries no
    /// player-built kind at all.
    #[test]
    fn the_fixture_plant_is_appended_last_and_the_plain_world_is_untouched() {
        let plain_cfg = UniverseConfig::world(15.0, 0.05);
        let planted_cfg = plain_cfg.with_station_area_plant();
        let plain = generate_system_forest(0, &plain_cfg);
        let planted = generate_system_forest(0, &planted_cfg);
        assert_eq!(planted.len(), plain.len() + 2);
        assert_eq!(
            &planted[..plain.len()],
            &plain[..],
            "every generated body is byte-identical under the plant"
        );
        let spec = station_area_plant(0, &planted_cfg);
        assert_eq!(planted[plain.len()].realm, spec.station);
        assert_eq!(planted[plain.len() + 1].realm, spec.area);
        // The plain world has no player-built kind: its only non-plantable bodies are the
        // three STAR realms (nothing is built inside the dust-sublimation radius — T2), which
        // are seed-lineage keyed all the same.
        assert!(
            plain
                .iter()
                .filter(|b| plant_seed_of(b.realm).is_none())
                .all(|b| matches!(b.realm, RealmId::Star(_))),
        );
        assert_eq!(
            plain
                .iter()
                .filter(|b| plant_seed_of(b.realm).is_none())
                .count(),
            3
        );
        // The accessor derives the SAME spec from the plain and the planted config (it strips the
        // plant before deriving, so the spec can never be derived from planted content).
        assert_eq!(spec, station_area_plant(0, &plain_cfg));
    }

    /// look_horizon.md slice 5 (G-IDENTICAL) — the planted pair's measured climbs: BOTH stop at
    /// two levels (the carrier's arity serves the whole planted world), the generated numbers are
    /// untouched, and the ADMISSION fence would admit both members as candidates — the fixture
    /// plants only what build admission would accept.
    #[test]
    fn g_identical_the_planted_pair_measures_climb_two_and_leaves_the_worlds_numbers_alone() {
        let plain = UniverseConfig::world(15.0, 0.05);
        let config = plain.with_station_area_plant();
        let spec = station_area_plant(0, &config);
        let climbs = measure_visibility_climb(0, &config);
        assert_eq!(climbs.len(), 41, "39 generated climbs + the 2 planted");
        assert_eq!(
            climbs.iter().map(|c| c.levels).max(),
            Some(1),
            "the true-size world serves the planted pair at climb 1 (a full spare level)"
        );
        let station = climbs
            .iter()
            .find(|c| c.body == spec.station)
            .expect("the station is measured");
        // The 10 km city stops at its OWN system now (the §6.3 admission margin measured):
        // its picture travels one hop, with the whole system clearance as slack.
        assert_eq!(
            (station.levels, station.top),
            (1, spec.station),
            "{station:?}"
        );
        let area = climbs
            .iter()
            .find(|c| c.body == spec.area)
            .expect("the area is measured");
        // The 20 m structure stops at its OWN planet (slack ~half the planet's SOI).
        assert_eq!((area.levels, area.top), (1, spec.area), "{area:?}");
        eprintln!(
            "[G-IDENTICAL plant] station climb levels {} top {:?} slack {:.3} m; area climb \
             levels {} top {:?} slack {:.3} m",
            station.levels, station.top, station.slack_m, area.levels, area.top, area.slack_m,
        );
        // The plant changes NO generated number: the worst planet slack is the SAME measured bit
        // pattern G-CLIMB pins on the plain world (the reserved-clearance class at the outer
        // re-solve).
        let plain_planet_slack_m = measure_visibility_climb(0, &plain)
            .iter()
            .filter(|c| matches!(c.body, RealmId::Planet(_)))
            .map(|c| c.slack_m)
            .fold(f64::INFINITY, f64::min);
        let planet_slack_m = climbs
            .iter()
            .filter(|c| matches!(c.body, RealmId::Planet(_)))
            .map(|c| c.slack_m)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(
            planet_slack_m, plain_planet_slack_m,
            "the plant moves no generated number (bit-identical worst planet slack)"
        );
        // The boot fence on the planted world: green at the landed arity, with a spare level.
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
        assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
        // THE ADMISSION CROSS-CHECK: both planted members, put to the build-admission fence as
        // candidates over the PLAIN world, are ACCEPTED at the landed arity — the fixture path
        // plants nothing the future build path would refuse.
        for (realm, parent, r, offset_m) in [
            (
                spec.station,
                spec.station_parent,
                spec.station_extent_m,
                spec.station_offset_m,
            ),
            (
                spec.area,
                spec.area_parent,
                spec.area_extent_m,
                spec.area_offset_m,
            ),
        ] {
            let candidate = CandidateRegion {
                realm,
                parent,
                shape: Boundary::Shell { r },
                offset_m,
            };
            assert_eq!(
                guard_candidate_climb_bounded(&candidate, 0, &plain, 2),
                Ok(()),
                "{realm:?}"
            );
        }
    }

    /// look_horizon.md slice 5 — the planted pair NESTS (the real boot fence, with the boot's own
    /// worst-instant reach map) and stands MEASURED-clear of the orbital plane, both ±Z polar
    /// flight axes, and its own shell wall.
    #[test]
    fn the_planted_pair_nests_and_stands_clear_of_the_plane_and_the_polar_axes() {
        let config = UniverseConfig::world(15.0, 0.05).with_station_area_plant();
        let spec = station_area_plant(0, &config);
        let bodies = generate_system_forest(0, &config);
        let regions = realm_regions_for_config(0, &config);
        // The boot's own reach map: a mover at its closed-form apoapsis, a static child at its
        // authored offset — the same shape the bins boot states (`child_reaches`).
        let reaches: std::collections::BTreeMap<_, _> = bodies
            .iter()
            .filter(|b| b.parent.is_some())
            .map(|b| {
                let reach = match b.placement {
                    Placement::Orbital(e) => vd_core::geometry::ChildReach::Excursion(
                        Motion::Kepler(e).max_excursion_m(Tier::Fine),
                    ),
                    Placement::StaticOffset(at) => vd_core::geometry::ChildReach::Fixed(at),
                };
                (b.realm, reach)
            })
            .collect();
        // 64 = the membership-bitset width every shard boot passes (`vd_sim::stub::MAX_REGIONS`).
        assert_eq!(
            vd_core::geometry::guard_regions_nest(&regions, 64, &reaches),
            Ok(())
        );
        // Station: inside the shell with margin; clear of BOTH ±Z polar flight axes (the licensed
        // exit corridor and the pixel gate's own park legs) by far more than its extent plus one
        // occupant step; above the orbital plane's MEASURED worst |z| (apoapsis at the
        // eccentricity cap times sin(inclination), over the home system's actual elements).
        let off = spec.station_offset_m;
        let off_len = off.length();
        let extent = spec.station_extent_m;
        let shell = regions
            .iter()
            .find(|r| r.realm == spec.station_parent)
            .expect("the home system is rostered")
            .shape
            .finite_extent();
        let step_m = config.interest.occupant_v_max_mps * config.interest.tick_dt_s;
        // Precomputed locals + inline captures (HR5 test discipline — a multi-line lazy format
        // argument is a line only a FAILING assert executes).
        let off_x = off.x;
        let off_z = off.z;
        assert!(
            off_len + extent < shell,
            "the station nests: {off_len} + {extent} < {shell}",
        );
        assert!(
            off_x > extent + step_m,
            "clear of the polar axes: x {off_x} vs extent {extent} + step {step_m}",
        );
        let worst_plane_z = bodies
            .iter()
            .filter(|b| b.parent == Some(spec.station_parent))
            .filter_map(|b| orbital_of(b.placement))
            .map(|e| e.sma * (1.0 + config.planet.ecc_cap) * e.inclination.sin())
            .fold(0.0, f64::max);
        assert!(
            off_z - extent > worst_plane_z,
            "clear of the orbital plane: z {off_z} − extent {extent} vs measured worst plane \
             |z| {worst_plane_z}",
        );
        eprintln!(
            "[G-IDENTICAL plant] station at {off:?} (|off| {off_len:.3} m, extent \
             {extent:.4} m); measured worst orbital-plane |z| {worst_plane_z:.4} m; occupant \
             step {step_m} m",
        );
        // Area: nests at half the inner planet's own solved shell plus its 20 m extent —
        // strictly inside, derived from the roster.
        let planet_shell = regions
            .iter()
            .find(|r| r.realm == spec.area_parent)
            .expect("the inner planet is rostered")
            .shape
            .finite_extent();
        assert_eq!(spec.area_offset_m.z, 0.5 * planet_shell);
        assert!(spec.area_offset_m.z + spec.area_extent_m < planet_shell);
    }

    /// look_horizon.md slice 5 — the planted interior bands: the home system's stays
    /// PLANET-dominated (the pinned 444.104489631 m — the station's term is smaller), and the
    /// inner planet GAINS a live interior band that brackets its own shell (the park band the
    /// pixel gate stands in exists), stamped from the plant with no message crossing (§3.4.4).
    #[test]
    fn the_planted_interior_bands_bracket_their_shells_and_the_systems_stays_planet_dominated() {
        let config = UniverseConfig::world(15.0, 0.05).with_station_area_plant();
        let spec = station_area_plant(0, &config);
        let regions = realm_regions_for_config(0, &config);
        let home = regions
            .iter()
            .find(|r| r.realm == spec.station_parent)
            .expect("the home system is rostered");
        // Precomputed locals + inline captures (HR5 test discipline): a multi-line lazy
        // format argument is a line only a FAILING assert executes — an uncoverable region.
        let home_spin = home.interior_band.spin_up_r_m();
        // Planet-dominated still: the outer planet's worst-instant excursion + its LOOK's
        // visibility reach out-reaches the 10 km city's offset + reach — cross-derived from
        // the forest (the interim 444.104489631 m pin retired with the interim world).
        let bodies = generate_system_forest(0, &config);
        let ecc_cap = config.planet.ecc_cap;
        let expected_home = bodies
            .iter()
            .filter(|b| b.parent == Some(spec.station_parent))
            .map(|b| {
                worst_hop_excursion_capped_m(&b.placement, ecc_cap)
                    + b.look.map_or(0.0, |look| {
                        vd_core::geometry::visibility_reach_m(
                            look.finite_extent(),
                            VISIBILITY_THETA_MIN_RAD,
                        )
                    })
            })
            .fold(0.0, f64::max);
        assert!(
            (home_spin - expected_home).abs() < 1.0e-6,
            "measured {home_spin} vs derived {expected_home}",
        );
        let station_term =
            spec.station_offset_m.length() + spec.station_extent_m * config.interest.spin_up_factor;
        assert!(
            home_spin > station_term,
            "planet-dominated: the city's term {station_term} is smaller than {home_spin}",
        );
        let planet = regions
            .iter()
            .find(|r| r.realm == spec.area_parent)
            .expect("the inner planet is rostered");
        // The planet's interior reach = the area's offset + its visibility reach (extent times
        // the one cot(θ/2) factor) — the §3.4.4 stamp, cross-derived here.
        let expected =
            spec.area_offset_m.length() + spec.area_extent_m * config.interest.spin_up_factor;
        let planet_spin = planet.interior_band.spin_up_r_m();
        let planet_shell = planet.shape.circumscribed_extent();
        assert!(
            (planet_spin - expected).abs() < 1.0e-9,
            "measured {planet_spin} vs derived {expected}",
        );
        // TRUE-SCALE INVERSION, expected and lawful (celestial_taxonomy_design §4.5.2 one
        // level up): the interior reach is a small FRACTION of the shell now — an occupant
        // crosses INTO the realm long before its interior is visible, so the realm itself
        // (holding the occupant) wakes its children by the ordinary direct-child AoI rule and
        // the interest cascade never fires (G-NO-CASCADE). The interim "park band outside the
        // shell" bracket was a tiny-world artifact.
        assert!(
            planet_spin < planet_shell,
            "true scale: the interior reach sits INSIDE the shell: spin {planet_spin} vs \
             shell {planet_shell}",
        );
        eprintln!(
            "[G-IDENTICAL plant] planet interior spin-up {planet_spin:.9} m (shell \
             {planet_shell:.4} m); system interior spin-up {home_spin:.9} m",
        );
        // Frames: the station lowers to its own StationLocal; the area to AreaLocal WITH its
        // planet's provenance — the one total map, fed the planted parent.
        let st = regions
            .iter()
            .find(|r| r.realm == spec.station)
            .expect("the station is rostered");
        assert_eq!(st.frame.realm(), Some(spec.station));
        assert_eq!(st.parent, Some(spec.station_parent));
        let ar = regions
            .iter()
            .find(|r| r.realm == spec.area)
            .expect("the area is rostered");
        assert_eq!(ar.parent, Some(spec.area_parent));
        assert_eq!(
            ar.frame,
            vd_core::pose::frame_for_realm(spec.area, Some(spec.area_parent))
                .expect("an area under a planet has the lawful AreaLocal frame")
        );
    }

    /// The monotone-slack arm on a HAND-BUILT forest with a ZERO-MARGIN level (look_horizon.md
    /// slice 2's gate): a level whose slack is exactly zero still COUNTS AS VISIBLE (the same
    /// equality convention as the offence filter — the margin the solve reserves is what keeps a
    /// lawful world strictly clear), so the climb passes it; one strictly-positive level stops
    /// it. Numbers chosen to stay exact in f64 (single-binade sums), so the zero is a ZERO.
    #[test]
    fn a_zero_margin_level_still_climbs_and_a_positive_one_stops_the_walk() {
        let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let extent = 10.0;
        let offset = 5.0;
        let required = extent * factor;
        let root = RealmId::System(800);
        let zero_parent = RealmId::Planet(801);
        let body = RealmId::Station(802);
        let forest = |parent_r: f64| {
            vec![
                GeneratedBody {
                    realm: root,
                    parent: None,
                    shape: Boundary::Shell { r: 1.0e9 },
                    taxon: None,
                    look: Some(Boundary::Shell { r: 1.0e9 }),
                    placement: Placement::StaticOffset(DVec3::ZERO),
                    photometrics: None,
                },
                GeneratedBody {
                    realm: zero_parent,
                    parent: Some(root),
                    shape: Boundary::Shell { r: parent_r },
                    taxon: None,
                    look: Some(Boundary::Shell { r: parent_r }),
                    placement: Placement::StaticOffset(DVec3::ZERO),
                    photometrics: None,
                },
                GeneratedBody {
                    realm: body,
                    parent: Some(zero_parent),
                    shape: Boundary::Shell { r: extent },
                    taxon: None,
                    look: Some(Boundary::Shell { r: extent }),
                    placement: Placement::StaticOffset(DVec3::new(offset, 0.0, 0.0)),
                    photometrics: None,
                },
            ]
        };
        // (a) THE ZERO-MARGIN LEVEL: the parent's shell sized so `d_min == required` exactly —
        // equality is VISIBLE, the climb passes it and stops at the (enormous) root.
        let zero_r = offset + extent + required;
        let climbs = visibility_climbs(&forest(zero_r), VISIBILITY_THETA_MIN_RAD, 0.0);
        let c = climbs.iter().find(|c| c.body == body).expect("measured");
        assert_eq!(
            c.levels, 2,
            "a zero-margin level climbs — equality counts as visible: {c:?}"
        );
        assert_eq!(c.top, zero_parent);
        assert!(
            c.slack_m > 0.0,
            "the stop happened at the root, with the root's own slack: {c:?}"
        );
        // (b) ONE ULP-SCALE POSITIVE MARGIN stops the walk at the parent: levels 1, the body's
        // own degenerate top, and the slack IS the margin (the monotone arm's other side).
        let stopped_r = zero_r + 1.0;
        let climbs = visibility_climbs(&forest(stopped_r), VISIBILITY_THETA_MIN_RAD, 0.0);
        let c = climbs.iter().find(|c| c.body == body).expect("measured");
        assert_eq!(c.levels, 1, "a positive margin stops the climb: {c:?}");
        assert_eq!(c.top, body, "nobody outside sees it");
        assert_eq!(c.slack_m, 1.0, "the slack IS the stated margin");
    }

    /// THE Q3 EVIDENCE AS A TEST (look_horizon.md slice 2's gate + the owner's ruling
    /// 2026-08-17): a ~20 m surface structure planted on a planet of THE world at today's
    /// compressed scale needs its picture carried THREE levels — the build-admission fence
    /// REFUSES the placement (never the boot), printing the body and its numbers. The
    /// near-real-scale re-solve is the scheduled cure; its first gate run must include
    /// `measure_visibility_climb`; the carrier goes to 3 only if that measurement demands it.
    /// THE §3.4.4 CLAIM, SETTLED BY MEASUREMENT (look horizon slice 4; Q1 APPROVED, owner
    /// 2026-08-17 — the design named this exact unit): the generator stamps each region's
    /// INTERIOR BAND at boot, from the FULL forest, BEFORE scoping — so a GALAXY shard's boot
    /// roster row for a star system carries the system's interior reach with **no message ever
    /// crossing a boundary** (Ask D stays deferred). On THE world that reach is
    /// `444.104489631` m (planet worst excursion at the ecc cap `142.045826247` + planet
    /// visibility reach `302.058663384`), BIT-IDENTICAL to the same two terms the climb
    /// measurement walks with (§3.3.2's one-formula identity). A leaf (a planet) carries the
    /// INERT band — nothing inside, nothing to reach — and so does every walk-scale region
    /// (the AoI machinery is inert there: the byte-identity arm).
    #[test]
    fn the_boot_roster_stamps_each_systems_interior_reach_no_message_crossing() {
        let config = UniverseConfig::world(15.0, 0.05);
        let world = WorldView::generated(0, &config);
        let regions = world.neighbourhood(&std::collections::BTreeSet::from([GALAXY]));
        let system_rows: Vec<_> = regions
            .iter()
            .filter(|r| r.parent == Some(GALAXY))
            .collect();
        assert_eq!(system_rows.len(), 3, "THE world's galaxy holds 3 systems");
        let bodies = generate_system_forest(0, &config);
        for row in &system_rows {
            let expected = bodies
                .iter()
                .filter(|c| c.parent == Some(row.realm))
                .map(|c| {
                    // The look split routed the visibility term onto the child's LOOK
                    // (real-scale design §3.0) — same two terms, same worst case.
                    worst_hop_excursion_capped_m(&c.placement, config.planet.ecc_cap)
                        + c.look.map_or(0.0, |look| {
                            vd_core::geometry::visibility_reach_m(
                                look.finite_extent(),
                                VISIBILITY_THETA_MIN_RAD,
                            )
                        })
                })
                .fold(0.0, f64::max);
            assert_eq!(
                row.interior_band.spin_up_r_m(),
                expected,
                "the stamped reach is BIT-IDENTICAL to the climb walk's own two terms: {row:?}"
            );
            let spin = row.interior_band.spin_up_r_m();
            // (The interim 444.104489631 m literal retired with the interim world; the
            // bit-identity above IS the §3.4.4 claim, at any scale.)
            assert!(
                spin > 0.0,
                "a planet-holding system has a live interior reach"
            );
            assert!(
                row.interior_band.tear_down_r_m() > spin,
                "the tear-down adds the derived lead: {row:?}"
            );
        }
        // A LEAF states no interior band — the zero-reach arm.
        let sys = system_rows[0].realm;
        let planet_rows = world.neighbourhood(&std::collections::BTreeSet::from([sys]));
        let planets: Vec<_> = planet_rows
            .iter()
            .filter(|r| r.parent == Some(sys) && matches!(r.realm, RealmId::Planet(_)))
            .collect();
        assert_eq!(
            planets.len(),
            9,
            "the derived 9 planets per system on THE world"
        );
        // T3 re-baseline (celestial_taxonomy_design §4.5.2's own named pin move): planets WITH
        // moons stop being leaves and carry their MEASURED interior band; planets WITHOUT still
        // assert exactly 0.0 — BOTH arms driven, a stronger pin than before.
        let bodies = generate_system_forest(0, &config);
        let mut moon_hosting = 0u32;
        let mut leaves = 0u32;
        for p in planets {
            let has_moons = bodies.iter().any(|b| b.parent == Some(p.realm));
            if has_moons {
                moon_hosting += 1;
                assert!(
                    p.interior_band.spin_up_r_m() > 0.0,
                    "a moon-hosting planet carries its measured interior reach: {p:?}"
                );
            } else {
                leaves += 1;
                assert_eq!(
                    p.interior_band.spin_up_r_m(),
                    0.0,
                    "a childless leaf is inert — no interior, no interest: {p:?}"
                );
            }
        }
        assert_eq!(
            (moon_hosting, leaves),
            (1, 8),
            "home: one moon host, eight leaves"
        );
        // Walk scale: parents exist, but the AoI machinery is inert ⇒ the band is inert too
        // (the `!interest.is_live()` arm; byte-identity where nothing demands).
        let walk = WorldView::generated(0, &UniverseConfig::walk_scale());
        let walk_regions = walk.neighbourhood(&std::collections::BTreeSet::from([GALAXY]));
        assert!(
            walk_regions
                .iter()
                .all(|r| r.interior_band.spin_up_r_m() == 0.0),
            "walk-scale regions carry the inert interior band"
        );
    }

    /// D-LOOK-1, THE SCHEDULED CURE MEASURED (owner Q3 ruling 2026-08-17: "(b)-first-then-
    /// measure — the near-real-scale re-solve is the scheduled cure"): at the TRUE-SIZE world
    /// the same ~20 m surface structure the interim world REFUSED (climb 3 > arity 2, the
    /// pinned Q3 evidence) is now ADMITTED — its picture stops at its own planet with ~half an
    /// SOI of slack. The refusal arm stays covered by a SYNTHETIC oversize candidate (a body
    /// whose look out-reaches its system's clearance — beyond the §6.3 build ceiling, which THE
    /// world's own mass domain cannot produce; stated as synthetic, HR5's named-arm discipline).
    #[test]
    fn q3_the_twenty_metre_structure_is_admitted_at_true_scale_and_the_fence_still_refuses() {
        let config = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &config);
        let origin_system = bodies
            .iter()
            .find(|b| {
                matches!(b.realm, RealmId::System(_))
                    && b.parent == Some(GALAXY)
                    && worst_hop_excursion_m(&b.placement) == 0.0
            })
            .expect("THE world's system 0 sits at the galactic origin")
            .realm;
        let planet = bodies
            .iter()
            .find(|b| b.parent == Some(origin_system))
            .expect("the origin system holds planets");
        // The Q3 case: a 20 m structure ON the planet's surface (its look radius out).
        let surface_m = planet.look.expect("a planet draws itself").finite_extent();
        let candidate = CandidateRegion {
            realm: RealmId::Station(7777),
            parent: planet.realm,
            shape: Boundary::Shell { r: 20.0 },
            offset_m: DVec3::new(surface_m, 0.0, 0.0),
        };
        assert_eq!(
            guard_candidate_climb_bounded(&candidate, 0, &config, 2),
            Ok(()),
            "the scheduled cure: the 20 m structure is ADMITTED at true scale"
        );
        // The 10 km city case (§6.3's other named margin) — admitted too.
        let city = CandidateRegion {
            realm: RealmId::Station(7778),
            parent: origin_system,
            shape: Boundary::Shell { r: 1.0e4 },
            offset_m: DVec3::new(surface_m * 2.0, 0.0, 0.0),
        };
        assert_eq!(guard_candidate_climb_bounded(&city, 0, &config, 2), Ok(()));
        // THE REFUSAL ARM (synthetic, named): a body whose picture out-reaches its planet AND
        // its system — beyond any lawful build, driven so the fence's Err arm stays real.
        let oversize = CandidateRegion {
            realm: RealmId::Station(7779),
            parent: planet.realm,
            shape: Boundary::Shell { r: 4.0e9 },
            offset_m: DVec3::new(surface_m, 0.0, 0.0),
        };
        let refused = guard_candidate_climb_bounded(&oversize, 0, &config, 2)
            .expect_err("a synthetic oversize candidate exceeds the landed carrier");
        eprintln!("[Q3] the admission refusal, verbatim: {refused}");
        assert_eq!(refused.body, oversize.realm);
        assert!(refused.levels > 2, "visible past its system");
        assert_eq!(refused.arity, 2);
        // …never the boot: THE world itself still passes the same fence.
        assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    }

    // (`the_shell_solve_binds_on_the_visibility_clearance_at_the_interim_scale` and
    // `the_shell_solve_is_slack_at_near_real_scale_containment_binds` retired WITH the upward
    // interim solve they pinned (`two_level_clearance_m`/`galaxy_shell_r_m` — real-scale addendum
    // §9.2). Their successors are the ▲-chain pins:
    // `the_four_outer_geometry_numbers_are_the_addendums_derivations` and the re-derived climb
    // pins below.)

    // ===== T2 — THE STAR REALM's gates (celestial_taxonomy_design §9 slice T2) ==========

    /// The star's extent: the dust-sublimation bound, pinned per star as measured, with the
    /// SCALE-FREE identity `bound/a₀` EQUAL for all three stars (both scale with √L — a
    /// measurement that can fail), and the photosphere fence green plus its SWEPT minimum
    /// printed over the whole IMF domain (1.993 at the hydrogen-burning limit — the "always
    /// ≥ 2" claim is FALSE at the edge and is not asserted; the fence's `> 1` is).
    #[test]
    fn g_star_extent_the_dust_bound_its_scale_free_identity_and_the_swept_photosphere_fence() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        let mut ratios = Vec::new();
        let mut stars = 0u32;
        for b in &bodies {
            let RealmId::Star(_) = b.realm else { continue };
            stars += 1;
            let star = b.photometrics.expect("a star carries its photometrics");
            let bound_m = b.shape.finite_extent();
            let a0_m = cfg.planet.orbital_a0_au
                * habitable_zone_radius_au(star.luma_lsun, 1.0)
                * crate::taxonomy::AU_M;
            ratios.push(bound_m / a0_m);
            // The bound IS flux_radius at T_sub, A = 0 — the one law, inverted.
            assert_eq!(
                bound_m,
                crate::taxonomy::flux_radius_m(
                    star.luma_lsun * crate::taxonomy::L_SUN_W,
                    crate::taxonomy::DUST_SUBLIMATION_K,
                    0.0,
                )
            );
            // The look is the photosphere — the SAME value the system's look states (§5.2:
            // one function, two call sites, no double-draw).
            let look_m = b.look.expect("a star draws itself").finite_extent();
            assert_eq!(look_m, crate::taxonomy::star_radius_m(star.mass_msun));
            let system_look = bodies
                .iter()
                .find(|p| Some(p.realm) == b.parent)
                .and_then(|p| p.look)
                .expect("the system's look is its star");
            assert_eq!(look_m, system_look.finite_extent());
            eprintln!(
                "[T2 star] {:?}: bound {bound_m} m, photosphere {look_m} m, bound/look {}",
                b.realm,
                bound_m / look_m
            );
        }
        assert_eq!(stars, 3);
        // The scale-free identity: the same ratio for every star at every seed.
        assert!((ratios[0] - ratios[1]).abs() < 1e-12, "{ratios:?}");
        assert!((ratios[0] - ratios[2]).abs() < 1e-12, "{ratios:?}");
        eprintln!("[T2 star] FROZEN_STAR_BOUND_OVER_A0 = {}", ratios[0]);
        assert!((ratios[0] - 0.086_075_043).abs() < 1e-8);
        // The boot fence, green on THE world…
        assert_eq!(guard_star_bound_exceeds_photosphere(0, &cfg), Ok(()));
        // …and SWEPT over the whole IMF mass domain, minimum printed (never asserted ≥ 2 —
        // the domain edge measures 1.993).
        let mut min_ratio = f64::INFINITY;
        for i in 0..=120 {
            let mass = cfg.stellar.mass_lo_msun
                + (cfg.stellar.mass_hi_msun - cfg.stellar.mass_lo_msun) * f64::from(i) / 120.0;
            let luma = main_sequence_luminosity(mass, &cfg.stellar.mlr_segments);
            let bound = crate::taxonomy::flux_radius_m(
                luma * crate::taxonomy::L_SUN_W,
                crate::taxonomy::DUST_SUBLIMATION_K,
                0.0,
            );
            let photosphere = crate::taxonomy::star_radius_m(mass);
            min_ratio = min_ratio.min(bound / photosphere);
            assert!(bound > photosphere, "the fence holds at {mass} Msun");
        }
        eprintln!("[T2 star] swept min(bound/photosphere) over [0.08, 120] Msun = {min_ratio}");
        assert!((min_ratio - 1.993).abs() < 0.01);
    }

    /// `g_star_shell_unmoved`: the star's clearance arm is LIVE in the shell solve and
    /// MEASURED never to bind (§5.3.2 Prediction A) — each system's shell still equals its
    /// pinned reserved bound, and the star's owed clearance loses to the binding planet term
    /// by the printed margin.
    #[test]
    fn g_star_shell_unmoved_the_stars_clearance_arm_never_binds() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        let mut shells: Vec<f64> = bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
            .map(|b| b.shape.finite_extent())
            .collect();
        shells.sort_by(f64::total_cmp);
        assert_eq!(
            shells[0], TARGET_SYSTEM_BOUND_HOME_M,
            "the home shell is unmoved"
        );
        assert_eq!(
            shells[2], TARGET_SYSTEM_BOUND_MAX_M,
            "the largest shell is unmoved"
        );
        for b in bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Star(_)))
        {
            let star = b.photometrics.expect("a star carries its photometrics");
            let star_term = child_clearance_m(
                b.shape.finite_extent(),
                crate::taxonomy::star_radius_m(star.mass_msun),
                VISIBILITY_THETA_MIN_RAD,
            );
            let shell = bodies
                .iter()
                .find(|p| Some(p.realm) == b.parent)
                .expect("parented")
                .shape
                .finite_extent();
            eprintln!(
                "[T2 star] {:?}: clearance owed {star_term} m vs shell {shell} m ({}x margin)",
                b.realm,
                shell / star_term
            );
            assert!(star_term * 2.0 < shell, "the star's arm never binds");
        }
    }

    /// `siblings_disjoint_static_vs_orbital` (§4.5.5 — the comparison the static fence
    /// structurally skips): the STATIC star against every ORBITAL sibling's worst-instant
    /// periapsis annulus, margin printed (≈5.8× at home WITH the sibling's own shell counted
    /// — not the 10.2× the source designs quoted without it).
    #[test]
    fn siblings_disjoint_static_vs_orbital_the_star_clears_every_planets_annulus() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        let mut worst_margin = f64::INFINITY;
        let mut pairs = 0u32;
        for star in bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Star(_)))
        {
            let star_bound = star.shape.finite_extent();
            for planet in bodies
                .iter()
                .filter(|p| p.parent == star.parent && matches!(p.realm, RealmId::Planet(_)))
            {
                let el = orbital_of(planet.placement).expect("a planet is Orbital");
                let peri_reach = el.sma * (1.0 - cfg.planet.ecc_cap) - planet.shape.finite_extent();
                let margin = peri_reach / star_bound;
                worst_margin = worst_margin.min(margin);
                pairs += 1;
                assert!(
                    peri_reach > star_bound,
                    "{:?} vs {:?}: the static star and the orbital annulus are disjoint",
                    star.realm,
                    planet.realm
                );
            }
        }
        assert_eq!(pairs, 27);
        eprintln!("[T2 star] worst static-vs-orbital margin = {worst_margin}x");
        assert!(worst_margin > 5.0, "the ≈5.8× home margin class holds");
    }

    // ===== T3 — THE MOON gates (celestial_taxonomy_design §9 slice T3) ==================

    /// ★ g_moon_census: THE world's moon roster, pinned as `f(seed)` — the MEASURED counts
    /// (7 world-wide, on the outermost one-two planets of each system — the design's
    /// §4.2.3 prediction, measured true), every moon's derived quantities in lawful ranges,
    /// its shell == its OWN gravitational SOI around its planet's REAL drawn mass, and the
    /// world's region budget printed against the 64 fence.
    #[test]
    fn g_moon_census_seven_moons_on_the_outer_planets_pinned_as_f_of_seed() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let bodies = generate_system_forest(0, &cfg);
        let moon_of = |b: &GeneratedBody| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_)));
        // Per-system, per-rung counts: [0×8,1] / [0×7,1,2] / [0×7,1,2] — the census.
        let mut census: Vec<(RealmId, Vec<usize>)> = Vec::new();
        for i in 0..3u32 {
            let sys = RealmId::System(system_seed_at(i));
            let mut planets: Vec<(RealmId, f64)> = bodies
                .iter()
                .filter(|b| b.parent == Some(sys) && matches!(b.realm, RealmId::Planet(_)))
                .map(|b| {
                    (
                        b.realm,
                        orbital_of(b.placement).expect("a planet is Orbital").sma,
                    )
                })
                .collect();
            planets.sort_by(|a, b| a.1.total_cmp(&b.1));
            let counts: Vec<usize> = planets
                .iter()
                .map(|(p, _)| bodies.iter().filter(|b| b.parent == Some(*p)).count())
                .collect();
            census.push((sys, counts));
        }
        eprintln!("[T3 census] {census:?}");
        assert_eq!(census[0].1, vec![0, 0, 0, 0, 0, 0, 0, 0, 1], "home");
        assert_eq!(census[1].1, vec![0, 0, 0, 0, 0, 0, 0, 1, 2]);
        // The third system's inner rung-8 moon was COUNTED but not EMITTED: its drawn share
        // of the budget sits under the potato radius — the reject arm driven by THE world's
        // own mass domain (the design's §4.3 point exactly; its predicted 7th moon assumed
        // the cap-mass emission table, and the measurement wins).
        assert_eq!(census[2].1, vec![0, 0, 0, 0, 0, 0, 0, 1, 1]);
        let moons: Vec<&GeneratedBody> = bodies.iter().filter(|b| moon_of(b)).collect();
        assert_eq!(
            moons.len(),
            6,
            "THE WORLD GETS SIX MOONS (measured; design predicted ~7)"
        );
        for m in &moons {
            let el = orbital_of(m.placement).expect("a moon is Orbital");
            let taxon = m.taxon.expect("a moon carries its taxon");
            let parent_mass_kg = bodies
                .iter()
                .find(|b| Some(b.realm) == m.parent)
                .and_then(|b| b.taxon)
                .expect("a moon's planet carries its taxon")
                .mass_kg;
            // Its shell IS its own gravitational SOI around its planet (D-REAL-1, one level
            // down — the same one function, different arguments).
            assert_eq!(
                m.shape.finite_extent(),
                crate::celestial::planet_soi(el.sma, taxon.mass_kg, parent_mass_kg)
            );
            assert_eq!(el.central_mass, parent_mass_kg);
            // Above the potato floor by emission; below its planet's mass by construction;
            // tidally-damped elements (the measured regular-satellite sigmas).
            assert!(taxon.radius_m >= crate::taxonomy::MOON_MIN_RADIUS_M);
            assert!(taxon.mass_kg < parent_mass_kg);
            assert!(el.ecc <= crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS);
            eprintln!(
                "[T3 census] {:?} under {:?}: a={} e={} i={} mass={} kg radius={} m soi={} m",
                m.realm,
                m.parent,
                el.sma,
                el.ecc,
                el.inclination,
                taxon.mass_kg,
                taxon.radius_m,
                m.shape.finite_extent()
            );
        }
        // The region budget with the plant: 2 ambient + 3 systems + 27 planets + 3 stars +
        // 7 moons + 2 planted = 44 against the 64 fence — 20 spare, printed.
        let planted = cfg.with_station_area_plant();
        let regions = realm_regions_for_config(0, &planted);
        eprintln!("[T3 census] world regions = {} against 64", regions.len());
        assert_eq!(regions.len(), 43);
        // Two moons of one planet stay disjoint at the worst instant (the §4.5.5 margin —
        // measured over every two-moon planet).
        let mut hosts: std::collections::BTreeMap<RealmId, Vec<&GeneratedBody>> =
            std::collections::BTreeMap::new();
        for m in &moons {
            hosts
                .entry(m.parent.expect("parented"))
                .or_default()
                .push(m);
        }
        for (host, ms) in hosts {
            if ms.len() < 2 {
                continue;
            }
            let mut anns: Vec<(f64, f64)> = ms
                .iter()
                .map(|m| {
                    let el = orbital_of(m.placement).expect("Orbital");
                    (el.sma, m.shape.finite_extent())
                })
                .collect();
            anns.sort_by(|a, b| a.0.total_cmp(&b.0));
            let ecc_cap = crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS;
            for w in anns.windows(2) {
                let gap = w[1].0 * (1.0 - ecc_cap) - w[0].0 * (1.0 + ecc_cap);
                let need = w[0].1 + w[1].1;
                eprintln!(
                    "[T3 census] {host:?}: adjacent moon annuli gap {gap} vs shells {need} \
                     ({}x)",
                    gap / need
                );
                assert!(
                    gap > need,
                    "two moons of one planet are disjoint at the cap"
                );
            }
        }
    }

    /// The potato floor's BOTH arms, driven from mass domains the generator itself produces:
    /// THE world emits (7 moons — the emit arm, above); a synthetic mass-floor config draws
    /// planets so light every ladder rung's moon falls under hydrostatic equilibrium and
    /// NOTHING is emitted (the reject arm — the design's own named synthetic).
    #[test]
    fn the_potato_floor_rejects_sub_equilibrium_moons_and_the_count_law_still_ran() {
        let mut cfg = UniverseConfig::visual_scale();
        // Every planet at the Mercury floor: the log-uniform draw with lo == cap yields the
        // floor exactly; the moon budget 1e-4·M then sits under the potato radius everywhere.
        cfg.planet.mass_cap_mearth = cfg.planet.mass_lo_mearth;
        let bodies = generate_system_forest(0, &cfg);
        let moons = bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        assert_eq!(moons, 0, "no Mercury-mass planet keeps a hydrostatic moon");
        // …and the world's own census (the emit arm) is the 7-moon pin above — both arms live.
    }

    /// The depth-4 machinery walked at the unit tier (T3): a moon's neighbourhood scope is its
    /// own five-level ancestor chain + nothing else; its coord resolves the full lineage; its
    /// frame is the SAME total map arm every planet takes (a moon IS a planet — HR4's sharpest
    /// statement, they are literally the same kind).
    #[test]
    fn a_moon_is_a_planet_at_depth_four_scope_coord_and_frame() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let regions = realm_regions_for_config(0, &cfg);
        let moon = regions
            .iter()
            .find(|r| {
                matches!(r.realm, RealmId::Planet(_))
                    && r.parent.is_some_and(|p| matches!(p, RealmId::Planet(_)))
            })
            .expect("THE world holds moons");
        // Scope: ancestors ∪ direct children — a moon shard evaluates exactly its chain.
        let scope = vd_core::worldgen::neighbourhood_scope(
            &regions,
            &std::collections::BTreeSet::from([moon.realm]),
        );
        assert_eq!(scope.len(), 5, "universe, galaxy, system, planet, moon");
        // Coord: the full root-rooted five-level lineage (depth 4).
        let coord = vd_core::worldgen::coord_of_realm(&regions, moon.realm)
            .expect("a moon's lineage resolves");
        assert_eq!(coord.path().levels().len(), 5);
        // Frame: the SAME one-field lift every planet takes — nothing can tell them apart.
        assert_eq!(
            moon.frame,
            vd_core::pose::frame_for_realm(moon.realm, moon.parent).expect("total"),
        );
        assert_eq!(
            vd_core::worldgen::level_of(moon.realm)
                .expect("seed-keyed")
                .kind,
            vd_core::realm_path::RealmKindTag::Planet,
            "a moon IS a planet — no Moon kind exists to be told apart"
        );
    }

    // ===== T4 — THE EARTH-LIKE PREDICATE gates (celestial_taxonomy_design §9 slice T4) ===

    /// Each of the six predicate clauses driven TRUE and FALSE (HR5), from hand-built rows
    /// around a real G-star candidate shape.
    #[test]
    fn earth_like_every_clause_is_driven_both_ways() {
        use crate::taxonomy::{Atmosphere, BodyTaxon, MU_N2, PlanetType, R_EARTH_M};
        let g_star = StarPhotometrics {
            mass_msun: 0.9,
            class: classify_spectral(0.9, &SpectralClass::MASS_BOUNDS),
            luma_lsun: main_sequence_luminosity(0.9, &SpectralClass::MLR_SEGMENTS),
        };
        assert_eq!(g_star.class, SpectralClass::G, "0.9 Msun is a G star");
        let candidate = BodyTaxon {
            class: PlanetType::Rocky,
            mass_kg: crate::taxonomy::M_EARTH_KG,
            radius_m: R_EARTH_M,
            insolation_rel: 0.748_314_795,
            t_eq_k: 236.785_700_196_447_92,
            bond_albedo: 0.30,
            atmosphere: Some(Atmosphere {
                mean_molecular_weight: MU_N2,
                scale_height_m: 8.0e3,
                reference_density_kgm3: None,
            }),
        };
        assert!(
            earth_like(&g_star, &candidate),
            "the reference candidate passes"
        );
        // 1. NOT a yellow sun (an M dwarf).
        let m_star = StarPhotometrics {
            mass_msun: 0.1,
            class: classify_spectral(0.1, &SpectralClass::MASS_BOUNDS),
            luma_lsun: main_sequence_luminosity(0.1, &SpectralClass::MLR_SEGMENTS),
        };
        assert!(!earth_like(&m_star, &candidate));
        // 2. NOT rocky.
        let sub_neptune = BodyTaxon {
            class: PlanetType::SubNeptune,
            ..candidate
        };
        assert!(!earth_like(&g_star, &sub_neptune));
        // 3. Out of the radius band (both edges).
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                radius_m: 0.79 * R_EARTH_M,
                ..candidate
            }
        ));
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                radius_m: 1.26 * R_EARTH_M,
                ..candidate
            }
        ));
        // 4. Out of the Kopparapu flux band (both edges — ruling B: [0.53, 1.10]).
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                insolation_rel: 0.52,
                ..candidate
            }
        ));
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                insolation_rel: 1.11,
                ..candidate
            }
        ));
        // 5. Out of the derived temperate band (both edges).
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                t_eq_k: 216.0,
                ..candidate
            }
        ));
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                t_eq_k: 261.0,
                ..candidate
            }
        ));
        // 6. Airless.
        assert!(!earth_like(
            &g_star,
            &BodyTaxon {
                atmosphere: None,
                ..candidate
            }
        ));
        // The derived temperate band converts the SAME flux limits once (the corrected
        // literals of §8.1: ~216.75 / ~260.16 K at the class table's 0.30 Bond albedo).
        let (t_lo, t_hi) = (earth_like_t_bound_k(0.53), earth_like_t_bound_k(1.10));
        assert!((t_lo - 217.2).abs() < 0.1, "measured {t_lo}");
        assert!((t_hi - 260.7).abs() < 0.1, "measured {t_hi}");
    }

    /// §8.2 THE NO-OP HONESTY PIN: given `rocky + G` the insolation and temperature clauses
    /// can never discriminate — the ladder is quantised and seed-free (rung 2 = 0.748 S⊕ at
    /// every seed) and a rocky rung-2 `T_eq` is one of two class-albedo values, both in band.
    /// MEASURED over a seed sweep: every rocky body in the flux band also passes the
    /// temperature clause — the two clauses' pass sets coincide exactly.
    #[test]
    fn earth_like_no_op_clauses_are_measured_as_no_ops_given_rocky_and_g() {
        use crate::taxonomy::PlanetType;
        let cfg = UniverseConfig::world(15.0, 0.05);
        let mut flux_passes = 0u32;
        let mut temp_passes = 0u32;
        let mut rung2_insolations: Vec<f64> = Vec::new();
        for seed in 0..64u64 {
            for b in generate_system_forest(seed, &cfg) {
                let Some(t) = b.taxon else { continue };
                if t.class != PlanetType::Rocky {
                    continue;
                }
                let (s_lo, s_hi) = crate::taxonomy::KOPPARAPU_FLUX_CONSERVATIVE;
                if (s_lo..=s_hi).contains(&t.insolation_rel) {
                    flux_passes += 1;
                    rung2_insolations.push(t.insolation_rel);
                    // Counted, never branched: the temperature clause's "false" arm is the very
                    // thing this pin measures as unreachable given rocky + G, so writing it as
                    // an `if` would ask the coverage gate to exercise an arm the world cannot
                    // produce.
                    temp_passes += u32::from(
                        (earth_like_t_bound_k(s_lo)..=earth_like_t_bound_k(s_hi))
                            .contains(&t.t_eq_k),
                    );
                }
            }
        }
        assert!(flux_passes > 0, "the sweep found rocky flux-band bodies");
        assert_eq!(
            flux_passes, temp_passes,
            "temperature discriminates NOTHING beyond the flux clause"
        );
        // ONE quantised temperate insolation across every star at every seed — the √L
        // cancellation, measured to f64 association (the division `L/(0.4·√L·r²)²` rounds a
        // few ulp differently per drawn L; the QUANTITY is seed-free, the bits are not).
        for s_rel in &rung2_insolations {
            assert!(
                (s_rel - 0.748_314_795).abs() < 1e-9,
                "rung 2 is 0.748315 S⊕ for every star at every seed: {s_rel}"
            );
        }
    }

    /// THE SL5 FIREWALL (§8.4): the ranking weights are TOOL policy and live in the seed-search
    /// binary alone — `UniverseConfig`'s own serialized field set carries no weight, no rank,
    /// no search knob (asserted against the serde field names, so a smuggled knob fails here).
    #[test]
    fn the_ranking_weights_are_absent_from_the_one_config() {
        let json = serde_json::to_value(UniverseConfig::world(15.0, 0.05))
            .expect("the one config serializes");
        let mut names = Vec::new();
        fn collect(prefix: &str, v: &serde_json::Value, out: &mut Vec<String>) {
            if let serde_json::Value::Object(map) = v {
                for (k, child) in map {
                    out.push(format!("{prefix}{k}"));
                    collect(&format!("{prefix}{k}."), child, out);
                }
            }
        }
        collect("", &json, &mut names);
        for name in &names {
            let lower = name.to_lowercase();
            assert!(!lower.contains("weight"), "a weight knob leaked: {name}");
            assert!(!lower.contains("rank"), "a rank knob leaked: {name}");
            assert!(!lower.contains("search"), "a search knob leaked: {name}");
        }
        assert!(!names.is_empty());
    }

    #[test]
    fn worst_hop_excursion_is_the_offset_for_a_static_and_the_apoapsis_for_a_mover() {
        // Static: the authored offset's magnitude, exactly.
        assert_eq!(
            worst_hop_excursion_m(&Placement::StaticOffset(DVec3::new(3.0, 0.0, 4.0))),
            5.0
        );
        // Mover: THE one closed-form worst-instant accessor — never a re-derived `a·(1+e)` beside it.
        let bodies = generate_system_forest(0, &UniverseConfig::world(15.0, 0.05));
        let mover = bodies
            .iter()
            .find(|b| matches!(b.placement, Placement::Orbital(_)))
            .expect("THE world has orbital movers");
        let elements = orbital_of(mover.placement).expect("a mover is Orbital");
        assert_eq!(
            worst_hop_excursion_m(&mover.placement),
            Motion::Kepler(elements).max_excursion_m(Tier::Fine)
        );
    }

    #[test]
    fn a_visible_grandchild_is_named_with_its_exact_numbers() {
        // A synthetic guilty forest — the arm THE (green) world can never take: root shell 1000 m,
        // a child 100 m off the root's centre, and a 20 m grandchild 50 m off the child's centre.
        // From just outside the root the grandchild can close to 1000 − (100+50) − 20 = 830 m, and
        // the band keeps a 20 m body visible out to 20·cot(θ_min/2) ≈ 1527.8 m — an offence.
        let root = RealmId::System(900);
        let child = RealmId::Planet(901);
        let grand = RealmId::Station(902);
        let bodies = vec![
            GeneratedBody {
                realm: root,
                parent: None,
                shape: Boundary::Shell { r: 1000.0 },
                taxon: None,
                look: Some(Boundary::Shell { r: 1000.0 }),
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: child,
                parent: Some(root),
                shape: Boundary::Shell { r: 200.0 },
                taxon: None,
                look: Some(Boundary::Shell { r: 200.0 }),
                placement: Placement::StaticOffset(DVec3::new(100.0, 0.0, 0.0)),
                photometrics: None,
            },
            GeneratedBody {
                realm: grand,
                parent: Some(child),
                shape: Boundary::Shell { r: 20.0 },
                taxon: None,
                look: Some(Boundary::Shell { r: 20.0 }),
                placement: Placement::StaticOffset(DVec3::new(0.0, 0.0, 50.0)),
                photometrics: None,
            },
        ];
        let offences = grandchild_visibility_offences(&bodies, VISIBILITY_THETA_MIN_RAD);
        let expected = GrandchildVisibleOutside {
            body: grand,
            ancestor: root,
            worst_dist_m: 150.0,
            extent_m: 20.0,
            d_min_m: 1000.0 - 150.0 - 20.0,
            required_m: 20.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD),
        };
        assert_eq!(offences, vec![expected]);
        // The fail-loud shape moved to the MEASURED climb (look_horizon slice 2): the same
        // guilty forest measures a grandchild climb of 3 (visible past its parent AND its
        // grandparent — it runs out of ancestors, so the slack is the ROOT's own non-positive
        // figure), and the fence refuses it at the landed arity while passing an arity that
        // could carry it (both arms, named).
        let climbs = visibility_climbs(&bodies, VISIBILITY_THETA_MIN_RAD, 0.0);
        let grand_climb = climbs
            .iter()
            .find(|c| c.body == grand)
            .expect("the grandchild is measured");
        assert_eq!(grand_climb.levels, 3);
        assert_eq!(grand_climb.top, root, "visible from outside even the root");
        assert!(
            grand_climb.slack_m <= 0.0,
            "the climb never stopped inside the forest: {grand_climb:?}"
        );
        assert_eq!(
            first_climb_over(&climbs, 3),
            Ok(()),
            "an arity that can carry the climb passes"
        );
        let refused = first_climb_over(&climbs, 2).expect_err("the landed arity refuses");
        assert_eq!(refused.body, grand);
        assert_eq!(refused.levels, 3);
        assert_eq!(refused.arity, 2);
    }

    #[test]
    fn world_geometry_is_true_size_and_the_ladder_is_luminosity_anchored() {
        let c = UniverseConfig::visual_demand(15.0, 0.02);
        // The derived planet count: 9, scale-free (the disc edge over the ladder ratio).
        assert_eq!(c.planet.n_planets, 9);
        assert_eq!(
            c.planet.n_planets,
            derived_planet_count(
                c.planet.orbital_a0_au,
                c.planet.orbital_ratio,
                (NEPTUNE_SMA_AU / crate::taxonomy::FROST_COEFF_AU) * c.planet.frost_coeff_au,
            )
        );
        assert_eq!(c.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
        // THE LADDER LAW: every home sma == a0·√L·ratio^n in TRUE metres — χ = 1 in-system,
        // exactly (no compression factor exists to be anything else).
        let bodies = generate_system_forest(0, &c);
        let star = bodies
            .iter()
            .find(|b| b.realm == SYSTEM_A)
            .and_then(|b| b.photometrics)
            .expect("the home star");
        let a0_m = c.planet.orbital_a0_au
            * habitable_zone_radius_au(star.luma_lsun, 1.0)
            * crate::taxonomy::AU_M;
        let smas: Vec<f64> = bodies
            .iter()
            .filter(|b| b.parent == Some(SYSTEM_A) && matches!(b.realm, RealmId::Planet(_)))
            .map(|b| orbital_of(b.placement).expect("a planet is Orbital").sma)
            .collect();
        assert_eq!(smas.len(), 9);
        for (n, sma) in smas.iter().enumerate() {
            let expect = orbital_axis_au(n as u32, 1.0, c.planet.orbital_ratio) * a0_m;
            assert!(
                (sma / expect - 1.0).abs() < 1e-12,
                "rung {n}: {sma} vs ladder {expect}"
            );
        }
        // Rung 2 is the temperate rung of every star in the universe: S = 6.25/2.89^2.
        let s2 = star.luma_lsun / (smas[2] / crate::taxonomy::AU_M).powi(2);
        assert!((s2 - 0.748_314_795).abs() < 1e-9, "measured {s2}");
        // The SAME geometry as visual_scale (the static-render twin): one game geometry.
        let vs = UniverseConfig::visual_scale();
        assert_eq!(c.stellar.system_ring_r_m, vs.stellar.system_ring_r_m);
        assert_eq!(c.planet.n_planets, vs.planet.n_planets);
        assert_eq!(c.planet.mass_lo_mearth, vs.planet.mass_lo_mearth);
    }

    #[test]
    fn visual_demand_band_is_crossable_between_stars() {
        // NON-VACUITY, the other half of `a_planet_is_visible_from_anywhere_inside_its_own_system`. Together
        // they sandwich the band: a planet is awake everywhere inside its own system (so arriving shows you
        // a populated system), and ASLEEP from the neighbouring star (so the interstellar leg crosses its
        // band and the spin-up machinery is actually exercised). Without this the wider angle could swell
        // until every planet in the galaxy is permanently awake and nothing would ever be measured waking.
        //
        // This USED to read `spin_up < orbit` — the outer planet asleep at its OWN star — which is the very
        // property that made a system look empty on arrival. The band did not stop being crossable; it moved
        // outward, so the crossing is now measured where it belongs, on the way in from another star.
        let cfg = UniverseConfig::visual_demand(15.0, 0.02);
        let (outer, orbit) = outer_planet_orbit(&cfg);
        // The closest a neighbouring star ever gets to this planet: the ring, less its orbit at worst phase.
        let from_the_next_star = cfg.stellar.system_ring_r_m - orbit;
        let reach = outer.aoi.tear_down_r_m();
        assert!(
            reach < from_the_next_star,
            "the outer planet must be asleep from the next star"
        );
    }

    #[test]
    fn equal_visibility_factor_with_zero_v_rel_is_err_not_panic() {
        // THE equal-factor tripwire: under a single visibility factor (spin_up_factor == tear_down_factor)
        // the geometric dead-zone collapses, so a ZERO-relative-velocity band has spin_up == tear_down →
        // Err(InvalidEdges), never a valid inert band (and to_regions' .expect would PANIC at boot on one).
        // The precondition band validity requires occupant_v_max + v_child > 0 — a live occupant. Equality
        // asserted via expect_err (NOT matches!). No epsilon is added to keep the factors exactly equal.
        let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let err = AoiConfig::for_velocity_safe(
            100.0,
            factor,
            factor,
            0.0,
            AOI_TICK_DT_S,
            VISUAL_AOI_GRACE_TICKS,
            VISUAL_AOI_K_SAFETY_EXTRA,
        )
        .expect_err("equal factors + zero v_rel collapse the dead-zone → InvalidEdges");
        assert_eq!(err, BandError::InvalidEdges);
    }

    #[test]
    fn visual_scale_preset_is_walk_physics_with_derived_true_size_geometry() {
        let c = UniverseConfig::visual_scale();
        // The galaxy holds the seeded placement radius with every system's reach inside it.
        assert!(c.scale.galaxy_r_m > c.stellar.system_ring_r_m + TARGET_SYSTEM_BOUND_MAX_M);
        // …and it is the storage-fence chain's shell exactly (real-scale addendum §A2.2, frozen).
        assert_eq!(c.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
        assert_eq!(
            c.planet.ecc_cap,
            ECC_SIGMA * ECC_CAP_SIGMAS,
            "the GEOMETRY cap (4σ), not the solver bound"
        );
        assert_eq!(c.planet.ecc_sigma, ECC_SIGMA);
        assert_eq!(c.planet.incl_sigma, INCL_SIGMA);
        assert_eq!(c.planet.n_planets, world_n_planets());
    }

    #[test]
    fn generate_system_forest_emits_the_ambient_forest_plus_n_orbital_planets() {
        let bodies = visual_forest();
        // 2 ambient shells + every system + that system's planets. The galaxy's population is DRAWN from
        // its census, so this reads the census rather than restating a number in two places.
        let n_sys = WORLD_SYSTEM_COUNT as usize;
        // 2 ambient + per system: itself + its planets + its STAR (T2) + the census moons (T3
        // — the MEASURED 7, pinned exactly by `g_moon_census…`).
        let moons = bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        assert_eq!(
            bodies.len(),
            2 + n_sys * (1 + world_n_planets() as usize + 1) + moons
        );
        assert_eq!(moons, 6, "the T3 census");
        assert_eq!(orbital_of(bodies[0].placement), None); // Universe
        assert_eq!(orbital_of(bodies[1].placement), None); // Galaxy
        // Every remaining body is either a system (static, under the galaxy) or one of its planets
        // (orbital, under a system) — no third kind, and no planet parented anywhere but a star.
        let systems: Vec<_> = bodies
            .iter()
            .filter(|b| b.parent == Some(GALAXY))
            .map(|b| b.realm)
            .collect();
        assert_eq!(systems.len(), n_sys);
        assert!(
            systems.contains(&SYSTEM_A),
            "system 0 keeps the named identity"
        );
        // The FIVE lawful shapes (§7.3, extended by T3): static system under galaxy; orbital
        // planet under a system; STATIC STAR at its system's origin (T2); ORBITAL MOON —
        // Planet-kind under a planet (T3); nothing else in the plain world.
        let planet_ids: std::collections::BTreeSet<RealmId> = bodies
            .iter()
            .filter(|b| {
                matches!(b.realm, RealmId::Planet(_))
                    && b.parent.is_some_and(|p| systems.contains(&p))
            })
            .map(|b| b.realm)
            .collect();
        for b in bodies.iter().skip(2) {
            if systems.contains(&b.realm) {
                assert_eq!(orbital_of(b.placement), None, "a system does not orbit");
            } else if matches!(b.realm, RealmId::Star(_)) {
                assert!(systems.contains(&b.parent.expect("a star nests in its system")));
                assert_eq!(orbital_of(b.placement), None, "the star sits at the origin");
                assert_eq!(placement_offset(b.placement), DVec3::ZERO);
            } else if planet_ids.contains(&b.realm) {
                assert!(orbital_of(b.placement).is_some());
            } else {
                // A MOON: Planet-kind, parented to a planet, orbiting it.
                assert!(matches!(b.realm, RealmId::Planet(_)));
                assert!(planet_ids.contains(&b.parent.expect("a moon has a planet")));
                assert!(orbital_of(b.placement).is_some());
            }
        }
    }

    #[test]
    fn a_galaxy_of_several_systems_gives_each_its_own_seed_place_and_planets() {
        // THE GENERALISATION. The previous shape named ONE system in code and hung the planets off a
        // constant, so a second star could not exist at any scale — which is how the login side and the
        // shard side ended up describing two different worlds. Every system now comes off the same loop.
        let mut cfg = UniverseConfig::visual_scale();
        cfg.galaxy.system_count_lo = 4;
        cfg.galaxy.system_count_hi = 4;
        cfg.stellar.system_ring_r_m = 4.0 * cfg.stellar.system_soi_r_m;
        let bodies = generate_system_forest(0, &cfg);

        let systems: Vec<_> = bodies.iter().filter(|b| b.parent == Some(GALAXY)).collect();
        assert_eq!(systems.len(), 4, "a galaxy is N systems, not one");
        // System 0 keeps the identity every existing fixture and label already names.
        assert_eq!(systems[0].realm, SYSTEM_A);
        // …and no two systems share an id, so their planets can never collide either.
        let ids: std::collections::BTreeSet<_> = systems.iter().map(|s| s.realm).collect();
        assert_eq!(ids.len(), 4, "system identities are distinct: {ids:?}");

        // Each system carries its OWN planets, and a planet belongs to exactly one star.
        for sys in &systems {
            let mine = bodies
                .iter()
                .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
                .count();
            assert_eq!(
                mine,
                world_n_planets() as usize,
                "{:?} has its own planets",
                sys.realm
            );
            // …and exactly ONE star child (T2): the body-bearing near-star realm.
            let stars = bodies
                .iter()
                .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Star(_)))
                .count();
            assert_eq!(stars, 1, "{:?} holds its star as a child realm", sys.realm);
        }
        let planets: std::collections::BTreeSet<_> = bodies
            .iter()
            .filter(|b| {
                b.parent.is_some_and(|p| ids.contains(&p)) && matches!(b.realm, RealmId::Planet(_))
            })
            .map(|b| b.realm)
            .collect();
        assert_eq!(
            planets.len(),
            4 * world_n_planets() as usize,
            "every planet across every system is a distinct realm"
        );

        // A DIFFERENT SEED DRAWS DIFFERENT ORBITS but the SAME AMBIENT + SYSTEM + PLANET +
        // STAR structure — the world is a pure function of the seed. The MOON census is
        // seed-DEPENDENT by design (T3: the count follows each star's drawn ladder and each
        // planet's drawn mass through the potato floor), so the structural comparison counts
        // the non-moon prefix kinds.
        let non_moon = |forest: &[GeneratedBody]| {
            forest
                .iter()
                .filter(|b| !b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
                .count()
        };
        let other = generate_system_forest(99, &cfg);
        assert_eq!(
            non_moon(&other),
            non_moon(&bodies),
            "structure is seed-independent"
        );
        assert_ne!(
            orbital_of(other[3].placement),
            orbital_of(bodies[3].placement),
            "orbits are seed-DERIVED, not fixed"
        );
    }

    #[test]
    fn a_forest_whose_star_systems_overlap_is_refused() {
        // THE FENCE THAT MAKES SEVERAL SYSTEMS SAFE. Authority is "the deepest realm containing you". Two
        // overlapping systems give a position two equally valid owners, and which shard simulates you
        // would come down to iteration order — a coin flip deciding where your input lands.
        let mut cfg = UniverseConfig::visual_scale();
        cfg.galaxy.system_count_lo = 4;
        cfg.galaxy.system_count_hi = 4;

        // A ring TIGHTER than the systems on it: neighbours intersect. The shells are SOLVED
        // per system now, so the spread derives from the solved bound, not a config radius.
        cfg.stellar.system_ring_r_m = TARGET_SYSTEM_BOUND_MAX_M;
        let overlapping = generate_system_forest(0, &cfg);
        let err = siblings_disjoint(&overlapping).expect_err("touching systems must be refused");
        assert_eq!(
            err.parent, GALAXY,
            "the ambiguity is between children of the galaxy"
        );

        // Spread them and the same forest is accepted — so the refusal is about the GEOMETRY, not about
        // having more than one star.
        cfg.stellar.system_ring_r_m = 4.0 * TARGET_SYSTEM_BOUND_MAX_M;
        assert_eq!(siblings_disjoint(&generate_system_forest(0, &cfg)), Ok(()));

        // And the single-system world every existing rig boots is accepted unchanged.
        assert_eq!(
            siblings_disjoint(&generate_system_forest(0, &UniverseConfig::visual_scale())),
            Ok(())
        );
    }

    #[test]
    fn the_sibling_fence_declines_to_judge_orbits_rather_than_guessing() {
        // AN HONEST LIMIT, pinned so nobody mistakes silence for a guarantee. An orbiting body's region
        // sits at its frame ORIGIN — its position is authored live each tick — so every planet looks
        // co-located to any static comparison. Judging orbits needs their SHELLS compared, which is a
        // separate check over the moving roster; until it exists, orbital overlap is UNCHECKED.
        let cfg = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &cfg);
        let orbiting = bodies
            .iter()
            .filter(|b| orbital_of(b.placement).is_some())
            .count();
        let moons = bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        assert_eq!(
            orbiting,
            (WORLD_SYSTEM_COUNT * world_n_planets()) as usize + moons,
            "the fixture really does orbit (planets + the T3 moons)"
        );
        // The planets pass — NOT because they are proven disjoint, but because this fence does not judge
        // orbits at all.
        assert_eq!(siblings_disjoint(&bodies), Ok(()));
    }

    // ---- the render-origin PIN classifier ----

    #[test]
    fn d_fo_7_no_static_region_sits_under_a_varying_ancestor_chain() {
        // D-FO-7 TRIPWIRE: the A4c realm feed ships its movers-only rows and does NOT widen — a STATIC child's
        // authored row rides its frame-local `center`, never a per-tick absolute. That is correct ONLY while no
        // static realm hangs under a MOVING ancestor (which would ride the parent's orbit the center cannot
        // express). Through P4 `generate_system_forest` gives orbiting planets NO children, so the case does not
        // exist. This asserts it: the day P4 first hangs a station under an orbiting planet, THIS fails, and the
        // D-FO-7 decision (parent per-tick rows behind a realm-lane AoI cull, vs the child shard authoring its
        // own box row) must be taken before the widening lands.
        // The invariant, stated as SET EQUALITY (HR5-clean — no branch on the never-true "static-under-varying"
        // condition, whose true arm would be uncoverable): a realm moves-or-inherits-motion IFF the realm is
        // ITSELF a mover. A static realm inheriting a moving ancestor is exactly the case where the two sets
        // diverge. Both filter arms are genuinely exercised — a planet (Orbital ⇒ true) and an ambient body
        // (Static ⇒ false) exist in every forest.
        //
        // This used to ask the question through the seed-derived ORIGIN CHAIN, which is gone: no realm folds
        // its own absolute any more. The question itself survives untouched — it is about the FOREST's shape,
        // not about anybody's absolute — so it is now asked directly of the parent links.
        let moving_anywhere_above = |bodies: &[GeneratedBody], realm: RealmId| -> bool {
            let mut cur = realm;
            for _ in 0..bodies.len() {
                let Some(body) = bodies.iter().find(|b| b.realm == cur) else {
                    return false; // unknown realm — no ancestry to inherit motion from
                };
                if orbital_of(body.placement).is_some() {
                    return true;
                }
                match body.parent {
                    None => return false, // the ambient root — the walk is complete
                    Some(parent) => cur = parent,
                }
            }
            false // hop cap — a cycle, which the boot guard rejects; a safe stop, never a hang
        };
        // The walk's own edge arms, driven where the forest cannot produce them: an UNKNOWN realm has
        // no ancestry to inherit motion from; a CYCLE (which the boot guard rejects at boot) stops at
        // the hop cap instead of hanging — both answer "no motion", never a panic.
        let unknown_only = [GeneratedBody {
            realm: RealmId::System(1),
            parent: None,
            shape: Boundary::Shell { r: 1.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 1.0 }),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
        }];
        assert!(
            !moving_anywhere_above(&unknown_only, RealmId::Planet(999)),
            "an unknown realm inherits nothing"
        );
        let cycle = [
            GeneratedBody {
                realm: RealmId::System(1),
                parent: Some(RealmId::System(2)),
                shape: Boundary::Shell { r: 1.0 },
                taxon: None,
                look: Some(Boundary::Shell { r: 1.0 }),
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: RealmId::System(2),
                parent: Some(RealmId::System(1)),
                shape: Boundary::Shell { r: 1.0 },
                taxon: None,
                look: Some(Boundary::Shell { r: 1.0 }),
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
        ];
        assert!(
            !moving_anywhere_above(&cycle, RealmId::System(1)),
            "a cycle stops at the hop cap — a safe no, never a hang"
        );
        let config = UniverseConfig::visual_scale();
        for seed in [0u64, 1, 7, 42, 100] {
            let bodies = generate_system_forest(seed, &config);
            let mut varying_chain: Vec<RealmId> = bodies
                .iter()
                .filter(|b| moving_anywhere_above(&bodies, b.realm))
                .map(|b| b.realm)
                .collect();
            let mut movers: Vec<RealmId> = bodies
                .iter()
                .filter(|b| orbital_of(b.placement).is_some())
                .map(|b| b.realm)
                .collect();
            varying_chain.sort();
            movers.sort();
            assert_eq!(
                varying_chain, movers,
                "D-FO-7 (seed {seed}): a realm inherits motion IFF it is itself a mover — a divergence means a \
                 static realm now hangs under a moving ancestor, which the realm feed's movers-only filter \
                 would silently drop. Take the D-FO-7 decision before the widening lands."
            );
        }
    }

    #[test]
    fn realm_regions_for_config_gives_each_moving_planet_a_zero_center() {
        let config = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &config);
        let regions = realm_regions_for_config(0, &config);
        assert_eq!(regions.len(), bodies.len());
        // A moving (Orbital) planet authors its position LIVE through its frame, so its region carries NO
        // baked position — center is the ZERO origin of its own frame. (The crossing-flap fix: a nonzero
        // epoch center would be double-counted against the live frame placement in region_signed_distance.)
        for (body, region) in bodies
            .iter()
            .zip(&regions)
            .filter(|(b, _)| matches!(b.realm, RealmId::Planet(_)))
        {
            assert!(orbital_of(body.placement).is_some(), "a planet is Orbital");
            assert_eq!(region.center.cell(), glam::I64Vec3::ZERO);
            assert_eq!(region.center.offset(), DVec3::ZERO);
        }
    }

    /// FRAME-COHERENT CONTAINMENT (the moving-realm crossing fix): a moving planet's position is authored
    /// ONCE — into the placement book, off its injected `MotionFn` — and its region `center`
    /// is ZERO (the boundary sits at the body's OWN frame origin). So an occupant sitting exactly at the
    /// planet's live orbital position is judged INSIDE its SOI, and an occupant at the star (17.9 m away) is
    /// OUTSIDE — the SAME geometry both the parent shard (planet as a moving child) and the planet's own
    /// shard (planet at the identity) compute, so a crossing cannot flap. Regression guard against the epoch
    /// `center` being double-counted against the frame placement (which put the SOI ~17.9 m off the planet).
    #[test]
    fn a_moving_planet_soi_is_centered_on_its_live_position_not_double_counted() {
        use vd_core::frame::FramePlacement;
        use vd_core::geometry::region_signed_distance;
        use vd_core::kinematics::secs_since_epoch;
        use vd_core::placement::PlacementBook;
        use vd_core::pose::StampedPose;
        let config = UniverseConfig::visual_scale();
        let regions = realm_regions_for_config(0, &config);
        let movers = moving_children_for_config(0, &config, SYSTEM_A);
        let (realm, elements) = movers
            .iter()
            .min_by(|a, b| {
                orbital_state(&a.1, 0.0)
                    .position
                    .length()
                    .total_cmp(&orbital_state(&b.1, 0.0).position.length())
            })
            .cloned()
            .expect("the visual forest has planet movers");
        let region = regions
            .iter()
            .find(|r| r.realm == realm)
            .expect("the mover realm has a region");
        let root = regions
            .iter()
            .find(|r| r.parent.is_none())
            .expect("the forest has a root")
            .frame;
        let tick_hz = 20.0;
        let tick = vd_core::UniverseTick(200);
        // A moving realm carries NO baked position — its center is the origin of its own frame.
        assert_eq!(
            region.center.offset(),
            DVec3::ZERO,
            "a moving planet's region.center must be ZERO (position authored via the frame)",
        );
        // The parent shard's authored book: the planet's row is its live orbital state at this tick —
        // the ONE writer's output, which every consumer (this measurement included) reads as data.
        let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
        let ctx = PlacementBook::new(
            root,
            tick,
            vec![(
                region.frame,
                FramePlacement::moving(state.position, state.velocity),
            )],
        );
        let live = state.position;
        let soi = region.shape.finite_extent();
        // Occupant sitting EXACTLY at the planet's live position (root frame) ⇒ INSIDE the SOI.
        let at_planet = StampedPose::at_rest(root, live, tick);
        let d_at =
            region_signed_distance(&at_planet, region, &ctx).expect("the planet frame resolves");
        // Occupant at the star origin (17.9 m from the planet) ⇒ OUTSIDE.
        let at_star = StampedPose::at_rest(root, DVec3::ZERO, tick);
        let d_star =
            region_signed_distance(&at_star, region, &ctx).expect("the planet frame resolves");
        // An occupant AT the planet's live position is inside its SOI (distance ≈ −SOI). Split asserts
        // (not `a && b`) so neither short-circuit leaves an uncovered branch (HR5).
        assert!(d_at < 0.0);
        assert!((d_at + soi).abs() < 1e-9);
        // An occupant at the star is outside by (orbit − SOI).
        assert!(d_star > 0.0);
        assert!((d_star - (live.length() - soi)).abs() < 1e-6);
    }

    #[test]
    fn moving_children_for_config_lists_every_planet_as_a_mover() {
        let config = UniverseConfig::visual_scale();
        let movers = moving_children_for_config(0, &config, SYSTEM_A);
        assert_eq!(movers.len(), world_n_planets() as usize);
        // Each mover pairs the planet realm with its exact elements (the FIRST non-empty roster —
        // the orbital_of Some-arm + moving_children filter-true in a live path).
        let bodies = generate_system_forest(0, &config);
        for (body, mover) in bodies.iter().skip(3).zip(&movers) {
            assert_eq!(mover.0, body.realm);
            assert_eq!(Some(mover.1), orbital_of(body.placement));
        }
    }

    #[test]
    fn moving_children_for_config_excludes_non_children_and_empty_hosts() {
        let config = UniverseConfig::visual_scale();
        // The Galaxy's only child (System A) is StaticOffset ⇒ no mover (filter-true + orbital_of None).
        assert!(moving_children_for_config(0, &config, GALAXY).is_empty());
        // A realm hosting nothing ⇒ no mover (parent-filter false arm).
        assert!(moving_children_for_config(0, &config, RealmId::System(999)).is_empty());
    }

    #[test]
    fn planet_ecc_is_branchlessly_capped_both_ways() {
        // A HUGE ecc_sigma lets the Rayleigh draw exceed the cap ⇒ `.min` returns the cap exactly.
        let mut hot = UniverseConfig::visual_scale();
        hot.planet.ecc_sigma = 5.0;
        let mut stream = realm_stream(0, &SYSTEM_A_LINEAGE);
        let hot_eccs: Vec<f64> = (0..64)
            .map(|_| planet_element_draws(&hot, &mut stream).ecc)
            .collect();
        assert!(hot_eccs.iter().all(|&e| e <= hot.planet.ecc_cap));
        assert!(
            hot_eccs.contains(&hot.planet.ecc_cap),
            "a large sigma must hit the cap",
        );
        // The real sigma (0.03) draws well below the cap ⇒ `.min` returns the sample.
        let cool = UniverseConfig::visual_scale();
        let mut s2 = realm_stream(0, &SYSTEM_A_LINEAGE);
        for _ in 0..world_n_planets() {
            assert!(planet_element_draws(&cool, &mut s2).ecc < cool.planet.ecc_cap);
        }
    }

    #[test]
    fn generate_system_forest_is_deterministic_and_in_domain() {
        // Same seed ⇒ byte-identical forest (the HR1 replay property).
        assert_eq!(
            generate_system_forest(0, &UniverseConfig::visual_scale()),
            generate_system_forest(0, &UniverseConfig::visual_scale()),
        );
        // Every planet's elements are in-domain across seeds (each assert split — no `&&`).
        for seed in [0u64, 1, 42, 999] {
            for body in generate_system_forest(seed, &UniverseConfig::visual_scale())
                .iter()
                .filter(|b| matches!(b.realm, RealmId::Planet(_)))
            {
                let e = orbital_of(body.placement).expect("a planet is Orbital");
                assert!(e.ecc >= 0.0);
                assert!(e.ecc <= KEPLER_ECC_MAX);
                assert!(e.inclination >= 0.0);
                assert!(e.inclination.is_finite());
                assert!(e.raan >= 0.0);
                assert!(e.raan < TAU);
                assert!(e.arg_periapsis < TAU);
                assert!(e.mean_anomaly_epoch < TAU);
            }
        }
    }

    #[test]
    fn generate_system_forest_differs_by_seed() {
        // Genuinely f(seed): different universe seeds yield different orbits/angles.
        assert_ne!(
            generate_system_forest(1, &UniverseConfig::visual_scale()),
            generate_system_forest(2, &UniverseConfig::visual_scale()),
        );
    }

    #[test]
    fn the_worlds_orbits_are_real_kepler_around_the_real_drawn_star_mass() {
        // The synthetic Kepler-tuned mass is DEAD: every planet's central mass is its own
        // star's drawn mass in kg, and the home periods run real Kepler — 1.7 days at rung 0
        // to ~2.8 years at rung 8 (real-scale design §3.3.7), MEASURED in band here.
        let bodies = visual_forest();
        let star = bodies
            .iter()
            .find(|b| b.realm == SYSTEM_A)
            .and_then(|b| b.photometrics)
            .expect("the home star");
        let day_s = 86_400.0;
        let mut periods: Vec<f64> = bodies
            .iter()
            .filter(|b| b.parent == Some(SYSTEM_A) && matches!(b.realm, RealmId::Planet(_)))
            .map(|b| orbital_of(b.placement).expect("a planet is Orbital"))
            .inspect(|el| {
                assert_eq!(el.central_mass, star.mass_msun * crate::taxonomy::M_SUN_KG);
            })
            .map(|el| el.period())
            .collect();
        periods.sort_by(f64::total_cmp);
        assert_eq!(periods.len(), 9);
        let inner_d = periods[0] / day_s;
        assert!(
            (1.5..2.0).contains(&inner_d),
            "inner ~1.7 d, measured {inner_d} d"
        );
        let outer_yr = periods[8] / (365.25 * day_s);
        assert!(
            (2.5..3.1).contains(&outer_yr),
            "outer ~2.8 yr, measured {outer_yr} yr"
        );
    }

    #[test]
    fn true_size_containment_and_sibling_annulus_non_overlap() {
        // Containment: every planet's worst-instant apoapsis + its shell + its clearance sits
        // inside its system's solved shell (the §3.2 solve, restated as the measurement).
        let bodies = visual_forest();
        for sys in bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
        {
            let shell = sys.shape.finite_extent();
            let ecc_cap = UniverseConfig::visual_scale().planet.ecc_cap;
            for p in bodies
                .iter()
                .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
            {
                let el = orbital_of(p.placement).expect("a planet is Orbital");
                let apo = el.sma * (1.0 + ecc_cap);
                assert!(
                    apo + p.shape.finite_extent() < shell,
                    "{:?} at worst instant stays inside {:?}",
                    p.realm,
                    sys.realm
                );
            }
            // NON-OVERLAP at the WORST INSTANT: adjacent annuli (a·(1±ecc_cap) widened by each
            // shell) never touch — provable at every instant for every seed (§3.3.4).
            let mut rungs: Vec<(f64, f64)> = bodies
                .iter()
                .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
                .map(|p| {
                    let el = orbital_of(p.placement).expect("Orbital");
                    (el.sma, p.shape.finite_extent())
                })
                .collect();
            rungs.sort_by(|a, b| a.0.total_cmp(&b.0));
            for w in rungs.windows(2) {
                let (a_in, soi_in) = w[0];
                let (a_out, soi_out) = w[1];
                assert!(
                    a_in * (1.0 + ecc_cap) + soi_in < a_out * (1.0 - ecc_cap) - soi_out,
                    "adjacent SOI annuli are disjoint at the eccentricity cap"
                );
            }
        }
    }

    #[test]
    fn walk_path_is_untouched_and_visual_planet_ids_are_distinct() {
        // Byte-identity: the walk boot path still yields the frozen 7-body forest + empty mover roster
        // (the new generator is uncalled by any walk path).
        assert_eq!(realm_regions_for(0).len(), 7);
        assert!(moving_children_for(0, SYSTEM_A).is_empty());
        // The 5 visual planet ids are mutually distinct and NONE aliases the walk Planet(7) — the
        // child_seed salt/index avalanche keeps them off the roster ids (no silent alias).
        // EVERY planet of EVERY star, not just one star's — the salt/index avalanche must keep them
        // distinct ACROSS systems too, or two stars would quietly claim the same planet realm.
        let forest = visual_forest();
        let moons = forest
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        let ids: Vec<RealmId> = forest
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Planet(_)))
            .map(|b| b.realm)
            .collect();
        let expect = (WORLD_SYSTEM_COUNT * world_n_planets()) as usize + moons;
        assert_eq!(ids.len(), expect);
        let mut distinct = ids.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            expect,
            "every planet id is distinct across every system"
        );
        for id in &ids {
            assert_ne!(*id, PLANET_A);
        }
    }

    #[test]
    fn default_home_realm_walks_root_galaxy_system_and_refuses_a_degenerate_forest() {
        // The login fallback home: the FIRST system under the first galaxy under the root — pure
        // forest-walk, no seed knowledge. A degenerate forest (no root, or no chain below it) is
        // `None`, so a cluster booted on one fails where it can be seen instead of placing every
        // account somewhere arbitrary.
        let world = boot_world_for_tests();
        let home = default_home_realm(world.regions()).expect("THE world has a home system");
        assert!(
            matches!(home, RealmId::System(_)),
            "the fallback home is a star system: {home:?}"
        );
        assert_eq!(default_home_realm(&[]), None, "an empty forest has no home");
        // A root with nothing under it: the galaxy hop refuses.
        let bare_root = [world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .copied()
            .expect("the world has a root")];
        assert_eq!(default_home_realm(&bare_root), None);
    }

    #[test]
    fn the_forest_is_a_valid_single_root_containment_tree() {
        let rs = regions();
        assert_eq!(rs.iter().filter(|r| r.parent.is_none()).count(), 1);
        let mut realms: Vec<RealmId> = rs.iter().map(|r| r.realm).collect();
        realms.sort();
        realms.dedup();
        assert_eq!(realms.len(), rs.len(), "every region has a distinct realm");
        assert_eq!(
            rs.len(),
            7,
            "the 7-region forest (5 shells + Station + Area)"
        );
        // The mandate depths: Universe 0, Galaxy 1, System 2, Planet 3; the sibling system is same-depth.
        assert_eq!(region_depth(&rs, UNIVERSE), 0);
        assert_eq!(region_depth(&rs, GALAXY), 1);
        assert_eq!(region_depth(&rs, SYSTEM_A), 2);
        assert_eq!(region_depth(&rs, PLANET_A), 3);
        assert_eq!(region_depth(&rs, SYSTEM_B), 2);
        // Station A nests directly under System A (depth 3); Area A nests under Planet A (depth 4, the
        // deepest region). The kind-agnostic detector re-homes into either with no station/area code.
        assert_eq!(region_depth(&rs, STATION_A), 3);
        assert_eq!(region_depth(&rs, AREA_A), 4);
        // SIBLING TOPOLOGY LOCKED (task #135 C-6b): System B is a CHILD OF THE GALAXY — a SIBLING of
        // System A reached through the shared parent — NOT a child of System A. This is the canonical forest
        // the LIVE boot uses; the interim `override_regions_for_boundaries` child-of-source model (a
        // playground-only, env-gated fixture) must never drift into it. Depth 2 already implies this, but
        // pin the parent explicitly so a future refactor cannot silently promote the child model.
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == SYSTEM_B)
                .expect("System B present")
                .parent,
            Some(GALAXY),
            "System B is a sibling under the Galaxy, NOT a child of System A",
        );
        // PARENTAGE LOCKED for the first-class Station/Area realms (task #133): a Station nests under its
        // star SYSTEM, an Area under its PLANET (the Area's frame REQUIRES a Planet parent — pin it so a
        // refactor cannot silently re-parent it and break `frame_for_realm`).
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == STATION_A)
                .expect("Station A present")
                .parent,
            Some(SYSTEM_A),
            "Station A nests directly under System A",
        );
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == AREA_A)
                .expect("Area A present")
                .parent,
            Some(PLANET_A),
            "Area A nests under Planet A (its frame provenance)",
        );
    }

    #[test]
    fn region_depth_of_an_unknown_realm_is_zero() {
        assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
    }

    #[test]
    fn walk_demand_band_is_crossable_for_every_separated_child() {
        // Every expectation is DERIVED from the generated forest, never hand-typed.
        let regions = walk_demand_regions();
        for r in &regions {
            let ext = r.shape.finite_extent();
            let su = r.aoi.spin_up_r_m();
            // (a) An occupant INSIDE the child (within its own extent) is always in range.
            assert!(
                su > ext,
                "spin-up must exceed the child's own extent (in range inside the child)"
            );
            let Some(parent_id) = r.parent else { continue };
            let parent = regions
                .iter()
                .find(|p| p.realm == parent_id)
                .expect("a region's parent is in the forest");
            // Flatten the NORMALIZED centre (the walk offsets are dyadic, so the residual
            // `.offset()` is exactly ZERO — reading it here silenced the separated arm).
            let d = r
                .center
                .delta_m(LatticePos::ORIGIN, r.frame.tier())
                .length();
            let td = r.aoi.tear_down_r_m();
            // (c) Releasable: a point inside the parent exists from which the child is out of tear-down
            // range (tear_down < the farthest-in-parent distance = separation + the parent's own extent).
            assert!(
                td < d + parent.shape.finite_extent(),
                "tear-down must release within the parent's reach (child is releasable)"
            );
            if d > 0.0 {
                // (b) SEPARATED child: out of range AT the parent origin ⇒ a walk toward it CROSSES the band.
                assert!(
                    su < d,
                    "a separated child must be out of range at the parent origin (crossable)"
                );
            } else {
                // (d) CO-LOCATED ancestor (offset 0): always in range at the parent origin.
                assert!(
                    su > d,
                    "a co-located child must be in range at the parent origin"
                );
            }
        }
    }

    #[test]
    fn planet_a_is_releasable_at_its_parents_origin() {
        // L3's precondition (spin-DOWN): Planet A releases while the occupant is still AT the parent origin
        // — a STRONGER fact than the uniform (c) criterion (which does NOT hold for the Area: 5.4 vs 5).
        let regions = walk_demand_regions();
        let planet = regions
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("planet A in the walk forest");
        let d = planet
            .center
            .delta_m(LatticePos::ORIGIN, Tier::Fine)
            .length();
        assert!(
            planet.aoi.tear_down_r_m() < d,
            "planet A must release at its parent's origin (tear-down < its own offset)"
        );
    }

    #[test]
    fn planet_a_geometric_warmup_precedes_its_boundary() {
        // Gate 1 rests on this: a geometric warm-up margin exists OUTSIDE the child's boundary.
        let regions = walk_demand_regions();
        let planet = regions
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("planet A in the walk forest");
        assert!(
            planet.aoi.spin_up_r_m() > planet.shape.finite_extent(),
            "planet A must warm up before its boundary (spin-up radius exceeds its extent)"
        );
    }

    // ===== FA-5 S1: the config-driven VISUAL-scale Orbital generator =====================

    // FROZEN compressed-real geometry goldens — EXACT f64, captured once from the derive helpers at the
    // compressed-real numbers and pinned as literals here (NON-self-referential: a regression in a derive
    // helper is caught, not silently re-captured). Approx: au→render 37.96 / planet SOI 3.95 / orbit
    // semi-major axes 15,26,44,75,127 / synthetic central mass / visibility factor cot(0.75°) ≈ 76.390.
    // RE-CAPTURED at the placement arc S4 (the apoapsis-solved compression + the 0.372 gap fraction) —
    // THE WORLD'S NUMBERS MOVED, deliberately: the outer planet's worst instant now sits exactly at the
    // margin inside its system shell (was: outside it, the S0 tripwire), and a planet's visibility reach
    // still crosses its whole system (302.1 m ≥ 300 m).
    const FROZEN_VISIBILITY_FACTOR: f64 = 76.38983065807547;
    // (`FROZEN_AU_TO_RENDER_M`, `FROZEN_PLANET_SOI_R_M`, `FROZEN_CENTRAL_MASS_KG` and the
    // 5-rung `FROZEN_ORBIT_SMA_M` retired WITH their quantities at the in-system re-solve —
    // real-scale design §9.2: per-planet SOIs are the D-REAL-1 equality pin's, the central
    // mass is the star's drawn mass, and the 9-rung TRUE-metre ladder is pinned as a DERIVED
    // identity by `world_geometry_is_true_size_and_the_ladder_is_luminosity_anchored`.)
    // The 2026-08-15 SHELL SOLVE (owner ruling, items 5/10 addendum) — EXACT f64, captured once
    // from the derivation at THE world's numbers and pinned as literals (non-self-referential).
    // The two-level clearance of the worst descendant (the outer planet at the ecc-cap apoapsis,
    // 142.046 m reach + 3.954 m extent × (1 + cot(θ/2)) + the 4 m solve margin ≈ 452.06 m)
    // OUT-BINDS the 300 m containment headroom, so the shell is ring + clearance ≈ 12_483.46 m
    // (was ring + 300 = 12_331.40 m, the 2026-08-15 measured failure). The worst measured margin
    // on THE world (seed 0) is ≈ 11.13 m — the 4 m reserved margin plus the slack of the worst
    // planet's DRAWN eccentricity sitting below the cap the solve bounds against.
    // (`FROZEN_TWO_LEVEL_CLEARANCE_M = 452.058663384243`, `FROZEN_GALAXY_SHELL_R_M =
    // 12483.45699203113` and `FROZEN_TWO_LEVEL_WORST_MARGIN_M = 11.127605697744457` retired with
    // the upward interim solve — real-scale addendum §9.2; kept here verbatim as the interim-scale
    // record. Their successors are the ▲ four-number pins below.)
    /// ▲ THE FOUR OUTER GEOMETRY NUMBERS (real-scale addendum §A2.3), pinned bit-for-bit as
    /// MEASURED on THE world — each equals the addendum's printed derivation exactly.
    const FROZEN_REAL_UNIVERSE_R_M: f64 = 2_251_799_813_685_248.0; // 2⁵¹ m, exact
    const FROZEN_REAL_GALAXY_R_M: f64 = 2_248_797_413_933_667.8; // R_uni − the τ-free outset
    // R_gal − clearance = 0.2376656 ly (flag-day re-measured: the reserved clearance covers
    // the SOLVED system shells — see TARGET_SYSTEM_BOUND_MAX_M's doc).
    const FROZEN_REAL_PLACEMENT_R_M: f64 = 2_248_490_503_621_178.5;
    const FROZEN_REAL_COMPRESSION_CHI: f64 = 16.378_390_724_051_055; // real NN separation / placement

    // ===== THE WINDOW LANE Slice 0: the per-system photometric draw (the marker datum) =========
    // Owner-approved 2026-08-15/16, docs/design/window_lane.md §2.2/§2.8: a sleeping child's point
    // of light is authored by its parent from the child's OWN generation stream. Slice 0 lands the
    // draw consumer-less (nothing moves); the Slice-A marker emit reads it.

    #[test]
    fn the_worlds_systems_draw_their_pinned_photometrics() {
        // FROZEN per-system draw goldens on THE world (seed 0) — EXACT f64, captured once from the
        // taxonomy chain (sample_imf_mass → classify_spectral → main_sequence_luminosity) at THE
        // world's stellar config and pinned as literals (NON-self-referential: a stream drift, a
        // re-ordered draw, or a retuned IMF is caught, not silently re-captured). All three stars
        // land M-class — the honest Salpeter answer (α = 2.35 concentrates mass draws at the low
        // bound; the u01 that would draw a G star is a ~1e-3 sliver). Sub-solar luma is expected:
        // the marker's DERIVED brightness knob (coordinate-scale model) is a later, separate owe.
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        let all = system_photometrics_for_config(0, &cfg);
        // The SYSTEM subset carries the pinned stellar goldens; the planets' REFLECTED draws
        // (Slice C1) are coherence-checked below against the derivation, not re-pinned per body.
        let draws: Vec<(RealmId, StarPhotometrics)> = all
            .iter()
            .copied()
            .filter(|(realm, _)| matches!(realm, RealmId::System(_)))
            .collect();
        assert_eq!(
            draws,
            vec![
                (
                    RealmId::System(7),
                    StarPhotometrics {
                        mass_msun: 0.09287894638451702,
                        class: SpectralClass::M,
                        luma_lsun: 0.0009726074241780799,
                    },
                ),
                (
                    RealmId::System(10487570625701098367),
                    StarPhotometrics {
                        mass_msun: 0.1081418058358058,
                        class: SpectralClass::M,
                        luma_lsun: 0.0013801082634453568,
                    },
                ),
                (
                    RealmId::System(13979593561158050752),
                    StarPhotometrics {
                        mass_msun: 0.16179874709518627,
                        class: SpectralClass::M,
                        luma_lsun: 0.00348634764331354,
                    },
                ),
            ],
        );
        // Provenance: the three pinned realms ARE the seed lineage's systems, in forest order
        // (system 0 keeps the named SYSTEM_A_SEED; the rest avalanche off the galaxy). Plain
        // equality, no destructuring match — a non-System draw fails the vec compare (HR5: no
        // uncoverable panic arm).
        let realms: Vec<RealmId> = draws.iter().map(|(realm, _)| *realm).collect();
        assert_eq!(
            realms,
            vec![
                RealmId::System(system_seed_at(0)),
                RealmId::System(system_seed_at(1)),
                RealmId::System(system_seed_at(2)),
            ]
        );
        assert_eq!(realms[0], RealmId::System(SYSTEM_A_SEED));
        // Coherence: each pinned (class, luma) IS the taxonomy derivation of its pinned mass —
        // the chain cannot silently decouple from the one drawn u01.
        for (_, p) in &draws {
            assert_eq!(
                p.class,
                classify_spectral(p.mass_msun, &SpectralClass::MASS_BOUNDS)
            );
            assert_eq!(
                p.luma_lsun,
                main_sequence_luminosity(p.mass_msun, &cfg.stellar.mlr_segments)
            );
        }
        // THE PLANET REFLECTOR DRAWS (Slice C1 — §1.1 item 3b "per direct child"): every planet
        // of THE world carries a marker datum; its class and mass provenance are its STAR's
        // (reflected light keeps the star's color), and its luma sits inside the closed-form
        // reflected band `L★ · albedo · r²/(4d²)` over the canonical albedo table at the
        // planet's own orbit — bounded by construction, MEASURED here (never assumed).
        let planets: Vec<(RealmId, StarPhotometrics)> = all
            .iter()
            .copied()
            .filter(|(realm, _)| matches!(realm, RealmId::Planet(_)))
            .collect();
        let world_bodies = generate_system_forest(0, &cfg);
        let moons_n = world_bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        assert_eq!(
            planets.len(),
            draws.len() * cfg.planet.n_planets as usize + moons_n,
            "every planet AND every moon of every system carries a marker datum"
        );
        let world = WorldView::generated(0, &cfg);
        for (realm, p) in &planets {
            let region = world
                .regions()
                .iter()
                .find(|r| r.realm == *realm)
                .expect("a drawn planet is a region of THE world");
            // A MOON's illuminating geometry (T3): the star is still the illuminator and the
            // DILUTION DISTANCE is its parent PLANET's orbit — resolve the star through the
            // grandparent and the distance through the parent's own elements.
            let parent = region.parent.expect("parented");
            let (illuminating_system, dilution_sma_m) = match parent {
                RealmId::Planet(_) => {
                    let grandparent = world_bodies
                        .iter()
                        .find(|b| b.realm == parent)
                        .and_then(|b| b.parent)
                        .expect("a moon's planet nests in a system");
                    let planet_el = moving_children_for_config(0, &cfg, grandparent)
                        .into_iter()
                        .find(|(child, _)| *child == parent)
                        .map(|(_, el)| el)
                        .expect("the moon's planet orbits its system");
                    (grandparent, planet_el.sma)
                }
                _ => (
                    parent,
                    moving_children_for_config(0, &cfg, parent)
                        .into_iter()
                        .find(|(child, _)| *child == *realm)
                        .map(|(_, el)| el.sma)
                        .expect("a planet of THE world orbits"),
                ),
            };
            let star = draws
                .iter()
                .find(|(sys, _)| *sys == illuminating_system)
                .map(|(_, s)| *s)
                .expect("the illuminator is a pinned system");
            assert_eq!(
                p.class, star.class,
                "reflected light keeps the star's color"
            );
            assert_eq!(p.mass_msun, star.mass_msun, "the illuminator's provenance");
            // The reflector's cross-section is the body's OWN derived LOOK radius (SL3).
            let (d, r) = (
                dilution_sma_m,
                region
                    .look
                    .expect("a planet of THE world draws itself")
                    .finite_extent(),
            );
            let (lo, hi) = GEOMETRIC_ALBEDO_BOUNDS;
            let dilution = (r * r) / (4.0 * d * d);
            // Two asserts, not one `&&` (HR5: a short-circuit's false arm is uncoverable).
            assert!(
                p.luma_lsun >= star.luma_lsun * lo * dilution,
                "{realm:?}: reflected luma {} below the derivation band",
                p.luma_lsun
            );
            assert!(
                p.luma_lsun <= star.luma_lsun * hi * dilution,
                "{realm:?}: reflected luma {} above the derivation band",
                p.luma_lsun
            );
        }
    }

    #[test]
    fn the_marker_datum_frames_the_pinned_draw_through_the_one_shared_codec() {
        // Slice A → look_horizon slice 1: the parent's marker datum for a sleeping child rides
        // the ONE window-body codec (`vd_core::look::marker_bag`, now with the presence floor's
        // extent beside it) — encode every system of THE world, decode through the shared reader,
        // and get back exactly the pinned (class code, luma) pair AND the stated radius. A
        // re-framed bag, a transposed field, or a codec fork fails here, not on a live wire.
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        let draws = system_photometrics_for_config(0, &cfg);
        let moons = generate_system_forest(0, &cfg)
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count();
        assert_eq!(
            draws.len(),
            3 + 3 * (cfg.planet.n_planets as usize + 1) + moons,
            "THE world's marker roster: three systems + every planet's reflector + each \
             system's STAR child (T2) + every moon's reflector (T3)"
        );
        assert_eq!(moons, 6, "the T3 census");
        for (_, p) in &draws {
            let datum = marker_datum(p);
            assert_eq!(datum, (p.class as u8, p.luma_lsun));
            let bag = vd_core::look::marker_bag(Some(datum), cfg.stellar.system_soi_r_m);
            assert_eq!(
                vd_core::look::luma_of(&bag),
                Ok((p.class as u8, p.luma_lsun))
            );
            assert_eq!(
                vd_core::look::extent_of(&bag),
                Ok(cfg.stellar.system_soi_r_m)
            );
            // A marker bag can never answer for a look (the structural exclusivity, decoded side).
            assert_eq!(
                vd_core::look::look_of(&bag),
                Err(vd_core::tlv::TlvError::MissingRequiredTag(
                    vd_core::look::TAG_LOOK
                ))
            );
        }
    }

    #[test]
    fn the_photometric_draw_is_deterministic_and_dynamics_blind() {
        // Two generations, identical draws (pure f(seed, config) — HR1: every shard hosting the
        // galaxy authors byte-identical markers with no shared state)…
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        assert_eq!(
            system_photometrics_for_config(0, &cfg),
            system_photometrics_for_config(0, &cfg)
        );
        // …and blind to the two CLUSTER-dynamics arguments (occupant speed / tick dt): they size
        // the interest band, never the world — the same draw whatever cluster runs it (SL5).
        let other_dynamics = UniverseConfig::world(15.0, 0.02);
        assert_eq!(
            system_photometrics_for_config(0, &cfg),
            system_photometrics_for_config(0, &other_dynamics)
        );
        // A DIFFERENT universe seed draws differently (the stream is real, not a constant): seed 1
        // shares no system seed with seed 0 beyond the named system 0, whose draw must move.
        let seed1 = system_photometrics_for_config(1, &cfg);
        // The NON-MOON census is config-pinned; the MOON census is seed-derived by design
        // (T3: each star's ladder and each planet's drawn mass gate emission), so the roster
        // is bounded below by the moonless shape and every extra row is a moon reflector.
        assert!(
            seed1.len() >= 3 + 3 * (cfg.planet.n_planets as usize + 1),
            "the moonless census floor holds at any seed"
        );
        assert_ne!(
            seed1[0].1,
            system_photometrics_for_config(0, &cfg)[0].1,
            "system 0's draw must differ under a different universe seed"
        );
        // The ambient shells carry NO draw. Systems draw their own light; every planet carries
        // its reflected datum (Slice C1 — a sleeping realm appears only as its parent's marker).
        // ONE equality over the whole forest (HR5: no matches!/count with uncoverable arms): the
        // bodies carrying a draw are exactly each system followed by its planets, forest order.
        let world = WorldView::generated(0, &cfg);
        let starred: Vec<RealmId> = world
            .bodies
            .iter()
            .filter_map(|b| b.photometrics.map(|_| b.realm))
            .collect();
        let mut expected = Vec::new();
        for i in 0..3 {
            let s = system_seed_at(i);
            expected.push(RealmId::System(s));
            for n in 0..cfg.planet.n_planets {
                expected.push(RealmId::Planet(child_seed(s, PLANET_SALT, u64::from(n))));
            }
            // T2: each system's STAR child follows its planets (the §7.1 per-system push
            // order) — its datum IS the system's pinned draw, re-stated, zero new draws.
            expected.push(RealmId::Star(child_seed(s, STAR_SALT, 0)));
            // T3: then the census MOONS, planet order then rung order — read from the forest
            // (the emission is potato-gated, so the roster states what actually exists).
            for b in world
                .bodies
                .iter()
                .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            {
                let parent = b.parent.expect("a moon has a planet");
                let in_this_system = world
                    .bodies
                    .iter()
                    .find(|q| q.realm == parent)
                    .and_then(|q| q.parent)
                    == Some(RealmId::System(s));
                if in_this_system {
                    expected.push(b.realm);
                }
            }
        }
        assert_eq!(starred, expected);
        // THE APPEND ASSERTION (§7.3): filtering the T2/T3 rows out reproduces the pre-T2
        // roster order EXACTLY — the star and moon passes INSERTED per-system rows and
        // shifted nothing else; the stream (draw) prefix identity is pinned separately above.
        let moons: std::collections::BTreeSet<RealmId> = world
            .bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .map(|b| b.realm)
            .collect();
        let without_new: Vec<RealmId> = starred
            .iter()
            .copied()
            .filter(|r| !matches!(r, RealmId::Star(_)) && !moons.contains(r))
            .collect();
        let pre_t2: Vec<RealmId> = expected
            .iter()
            .copied()
            .filter(|r| !matches!(r, RealmId::Star(_)) && !moons.contains(r))
            .collect();
        assert_eq!(without_new, pre_t2);
    }

    /// The star-bound boot fence's REFUSAL arm: a star whose authority bound sits inside its own
    /// photosphere is a world nobody may boot (T2 §5.3). Built by hand — THE world never
    /// produces it (the green arm is measured by every boot) — so the refusal is exercised
    /// exactly once, here, with its stated numbers.
    #[test]
    fn the_star_bound_fence_refuses_a_bound_inside_the_photosphere() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let mut bodies = generate_system_forest(0, &cfg);
        let star = bodies
            .iter_mut()
            .find(|b| matches!(b.realm, RealmId::Star(_)))
            .expect("THE world names a star");
        let photosphere = crate::taxonomy::star_radius_m(
            star.photometrics
                .expect("a star carries photometrics")
                .mass_msun,
        );
        star.shape = vd_core::geometry::Boundary::Shell {
            r: 0.5 * photosphere,
        };
        let err = guard_star_bounds(&bodies).expect_err("the fence refuses");
        assert_eq!(err.bound_m, 0.5 * photosphere);
        assert_eq!(err.photosphere_m, photosphere);
    }

    /// `earth_like_candidates` — the T4 tool's whole read path (the bin is a four-line shell, so
    /// this is where the sweep is measured). Both continue arms ride here too: a body whose
    /// parent is not a system resolves its star through the GRANDparent (a moon), and a body
    /// whose chain names no photometric star is skipped.
    #[test]
    fn earth_like_candidates_reads_the_world_and_answers_with_its_numbers() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        // The measured best seed of the ruling-F sweep — it holds exactly one Earth-like body.
        let found = earth_like_candidates(2298, &cfg);
        assert_eq!(found.len(), 1);
        // PINNED AS MEASURED (the config here is the unit tier's 15 m/s · 0.05 s world, not
        // the DEV cluster's — the ladder is the same, the derived speed knobs are not).
        assert_eq!(
            found[0],
            EarthLikeCandidate {
                system: RealmId::System(7),
                body: RealmId::Planet(15_792_791_038_712_096_226),
                star_mass_msun: 1.031_280_722_475_864_7,
                mass_kg: 6.489_098_886_649_445e24,
                radius_m: 6_515_459.435_746_093,
                insolation_rel: 0.748_314_795_081_476_6,
                t_eq_k: 236.785_700_196_447_92,
            },
        );
        // A seed with no Earth-like body answers with an EMPTY sweep — the same read path, the
        // other verdict (the sweep is 1-in-181, so seed 0 is the ordinary case).
        assert_eq!(earth_like_candidates(0, &cfg), Vec::new());
    }

    /// `WorldView::default_home_offset_m` — the T2 spawn standoff, both arms: on THE world the
    /// home system holds a STATIC child that contains its centre (the star), so the clearing is
    /// twice that child's bound; on a world whose home realm cannot be named there is no
    /// clearing at all and the answer is the pre-T2 zero, byte-identical.
    #[test]
    fn the_home_offset_is_twice_the_centre_holding_childs_bound_or_zero() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let world = WorldView::generated(0, &cfg);
        let home = vd_core::worldgen::default_home_realm(world.regions()).expect("a home");
        let star_bound = world
            .regions()
            .iter()
            .find(|r| matches!(r.realm, RealmId::Star(_)) && r.parent == Some(home))
            .map(|r| r.shape.finite_extent())
            .expect("the home system holds its star");
        assert_eq!(
            world.default_home_offset_m(),
            vd_core::glam::DVec3::new(0.0, 0.0, 2.0 * star_bound),
        );
        // A view whose forest names no home realm: the clearing is zero.
        let empty = WorldView {
            bodies: Vec::new(),
            regions: Vec::new(),
        };
        assert_eq!(empty.default_home_offset_m(), vd_core::glam::DVec3::ZERO);
    }

    /// The two DEFENSIVE arms of the Earth-like sweep, and the MOON arm beside them. THE world
    /// never produces either refusal — every generated body's parent is in the forest and every
    /// system carries its star — so they are measured over a hand-built forest, which is also
    /// the only honest way to state "this cannot happen here".
    #[test]
    fn the_earth_like_sweep_skips_an_orphan_and_a_starless_chain() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let world = generate_system_forest(2298, &cfg);
        let earthlike = earth_like_in_forest(&world);
        assert_eq!(earthlike.len(), 1);
        let body = world
            .iter()
            .find(|b| b.realm == earthlike[0].body)
            .copied()
            .expect("the candidate is a body of this forest");
        // (a0) PARENTLESS: a taxon-bearing body with no parent at all — no chain, so no star,
        // so no verdict. THE world never emits one (every taxon rides a planet or a moon).
        let mut rootless = body;
        rootless.parent = None;
        assert_eq!(earth_like_in_forest(&[rootless]), Vec::new());
        // (a) ORPHAN: the candidate re-parented onto a planet that is not in the forest — the
        // grandparent lookup answers None and the body is skipped.
        let mut orphan = body;
        orphan.parent = Some(RealmId::Planet(0xdead_beef));
        assert_eq!(earth_like_in_forest(&[orphan]), Vec::new());
        // (b) STARLESS: the candidate under a system row that carries no photometrics.
        let mut starless_system = world
            .iter()
            .find(|b| b.realm == earthlike[0].system)
            .copied()
            .expect("the candidate's system");
        starless_system.photometrics = None;
        assert_eq!(earth_like_in_forest(&[starless_system, body]), Vec::new(),);
        // (c) THE MOON ARM: a body whose parent is a PLANET resolves its star through the
        // GRANDparent — stated by re-parenting the candidate under a planet of its own system.
        let host = world
            .iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)) && b.realm != body.realm)
            .copied()
            .expect("the system holds another planet");
        let system_row = world
            .iter()
            .find(|b| b.realm == earthlike[0].system)
            .copied()
            .expect("the system row");
        let mut moonised = body;
        moonised.parent = Some(host.realm);
        let via_grandparent = earth_like_in_forest(&[system_row, host, moonised]);
        assert_eq!(via_grandparent.len(), 1);
        assert_eq!(via_grandparent[0].system, earthlike[0].system);
    }

    /// The moon ladder's SOI-CLEARANCE clamp — the third `child_clearance_m` arm, measured
    /// 25×-slack inert on THE world (the census pin prints the counts). Reachable only over a
    /// hostile input: a planet whose stored shell is a fraction of its own Hill disc, so the
    /// first rung's worst instant plus its clearance already breaches the SOI and NO moon is
    /// minted. That is exactly what the clamp promises.
    #[test]
    fn the_moon_ladder_mints_nothing_when_the_first_rung_would_breach_the_soi() {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let world = generate_system_forest(0, &cfg);
        // A planet THE world actually gives moons to — its first ladder rung is inside the disc
        // edge, so the disc-edge break cannot pre-empt the clearance clamp under test.
        let hosts: std::collections::BTreeSet<RealmId> = world
            .iter()
            .filter(|b| matches!(b.parent, Some(RealmId::Planet(_))))
            .filter_map(|b| b.parent)
            .collect();
        let host = world
            .iter()
            .find(|b| hosts.contains(&b.realm))
            .copied()
            .expect("THE world gives some planet moons");
        let star = world
            .iter()
            .find(|b| b.realm == RealmId::System(7))
            .and_then(|b| b.photometrics)
            .expect("the home system carries its star");
        let taxon = host.taxon.expect("a generated planet carries its taxon");
        let sma_m = orbital_of(host.placement)
            .expect("a generated planet orbits")
            .sma;
        // The SAME planet, its authority shell cut to one Roche radius — small enough that the
        // clearance clamp refuses the first rung.
        let mut pinched = host;
        pinched.shape = Boundary::Shell {
            r: crate::taxonomy::roche_radius_m(taxon.mass_kg, crate::taxonomy::RHO_ROCK_KGM3),
        };
        let realm = pinched.realm;
        let mut bodies = vec![pinched];
        let minted = append_moons(
            &mut bodies,
            &cfg,
            0,
            7,
            &star,
            realm,
            // The moon stream's own salt — any seed states the same clamp; the clamp is
            // geometry, not chance.
            12_345,
            sma_m,
            taxon.mass_kg,
            0.3,
        );
        assert_eq!(minted, 0);
        assert_eq!(bodies.len(), 1);
    }
}
