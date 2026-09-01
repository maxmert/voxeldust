//! THE ONE CONFIG HOME for the seed universe generator, and the cited constants its defaults are
//! built from.
//!
//! Owns: the named sub-structs (scale, galaxy, stellar, planet, satellite, band, interest) and the
//! literature constants each default cites — an IMF slope, a Titius-Bode ratio, a Rayleigh sigma.
//! Every one carries its source, because a number nobody can trace is a number nobody can change.
//!
//! Does NOT own: a variant of the world. There is ONE universe, generated from the seed, used by the
//! game and by every test (SL5) — a config exists so THE world's numbers can be stated in one place,
//! never so a second, smaller world can be made.

use super::scale::{
    BAND_EXTENT_CLAMP, BAND_TAU_HEADROOM, BAND_TICKS_N, GEOMETRY_TICK_DT_S, GEOMETRY_V_FOOT_MPS,
    T_TRAVERSE_S,
};
use super::{
    AREA_HALF_M, AREA_OFFSET_M, CONTAINMENT_INSET_M, CONTAINMENT_OUTSET_M, FixturePlant,
    GALAXY_R_M, PLANET_A_OFFSET_M, REAL_GALAXY_R_M, REAL_UNIVERSE_R_M, STATION_A_OFFSET_M,
    STATION_HALF_M, SYSTEM_B_OFFSET_M, SYSTEM_SOI_R_M, UNIVERSE_R_M, VISIBILITY_THETA_MIN_RAD,
    aoi_v_rel_mps, derived_world_planet_count, imf_mass_hi_msun, planet_config, real_placement_r_m,
};
use crate::taxonomy::{FrostThresholds, GalaxyType, SpectralClass};
use serde::{Deserialize, Serialize};
use vd_core::flight::realm_speed_cap_mps;
use vd_core::geometry::visibility_factor;
use vd_core::geometry::{AoiConfig, BandError, Boundary, ContainmentBand, band_for_speed};
use vd_core::worldgen::{WALK_DEMAND_AOI_GRACE_S, grace_ticks_from_seconds};

// --- Stellar/orbital PHYSICS (scale-independent; walk + canonical share these) ---
/// Salpeter IMF slope α (Salpeter 1955): `dN/dM ∝ M^-2.35`.
pub(crate) const IMF_SLOPE: f64 = 2.35;
/// The stellar mass draw's LOWER bound (solar masses): the hydrogen-burning limit — below it a
/// body is a brown dwarf, not a star, and the draw has nothing to say about it. The UPPER bound is
/// no longer a literal: it is [`imf_mass_hi_msun`], the largest star THIS galaxy can host, derived
/// from the galaxy's own radius (the literal 120.0 that used to sit here named a star whose system
/// is ten times wider than the galaxy that would contain it — see [`DERIVED_MASS_CAP`]).
pub(crate) const IMF_MASS_LO_MSUN: f64 = 0.08;
/// ★ THE STELLAR MASS DRAW'S PHYSICAL UPPER BOUND (solar masses) — the initial mass function's own
/// top, restored as a NAMED PHYSICAL FACT (slice S7).
///
/// This is the literal that used to sit on [`IMF_MASS_LO_MSUN`]'s doc as the draw's upper bound before
/// the derived cap replaced it. Replacing it was right at the time and for the stated reason: a star of
/// this mass wants a system ten times wider than the galaxy that would contain it, so the galaxy could
/// not pay for one. But deleting the number lost something real — **how big a star the universe
/// actually makes** stopped being written down anywhere, and "the heaviest star" silently came to mean
/// "whatever this galaxy's radius happens to afford".
///
/// The two are different kinds of fact and they belong side by side: this one is physics, and the
/// derived cap is a fact about a coordinate step. Which of them BINDS is then a question that can be
/// asked and answered — see [`guard_galaxy_affords_its_stars`](crate::worldgen::scale::guard_galaxy_affords_its_stars).
///
/// MEASURED (slice S7, this world, our own solver): at today's millimetre step the galaxy affords
/// 16.36, so the coordinate step binds and this physical bound does not. At a one-metre step it affords
/// 866, and at the ruled two-metre step 1,287 — so from one metre upward THIS number is the binding one
/// and the coordinate step has stopped mattering, which is exactly what the step change is for.
pub(crate) const IMF_MASS_HI_PHYSICAL_MSUN: f64 = 120.0;
/// Titius-Bode orbital spacing seed (AU) + geometric ratio (Chambers 1996).
pub(crate) const ORBITAL_A0_AU: f64 = 0.4;
pub(crate) const ORBITAL_RATIO: f64 = 1.7;
/// Rayleigh scale for orbital eccentricity / inclination (Fabrycky 2014) — small so sampled
/// values stay well inside `KEPLER_ECC_MAX` (the generator also hard-caps at `ecc_cap`).
pub(crate) const ECC_SIGMA: f64 = 0.03;
/// The eccentricity cap in RAYLEIGH SIGMAS (the placement arc S4, owner-gated lever 1): `ecc_cap =
/// ECC_SIGMA · ECC_CAP_SIGMAS = 0.12`, and the AU compression is solved against the APOAPSIS at that
/// cap — so every planet's worst instant lands inside its system shell BY CONSTRUCTION, for every
/// seed, exactly. The cap replaces `KEPLER_ECC_MAX` doing geometry duty (a solver-convergence bound
/// has no business sizing a world); the clamp truncates the physical Rayleigh distribution with
/// probability `e^-(4²/2) = 3.35e-4` per planet — named, derived, never a magic number. MEASURED
/// before this lever: 2 of 3 systems' outer planets crossed their own shell at apoapsis (152.57 m and
/// 155.05 m against 150 m) — the S0 tripwire this lever turns green.
pub(crate) const ECC_CAP_SIGMAS: f64 = 4.0;
pub(crate) const INCL_SIGMA: f64 = 0.02;
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

// (`WALK_SYSTEM_COUNT` IS DELETED WITH THE COUNT KNOBS, S12/G8 2026-08-28 — it named a
// population for the hand-placed walk world, which the ruling forbids as much as any other.)

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
    // (`system_count_lo` and `system_count_hi` ARE DELETED, S12/G8 2026-08-28. They named a
    // population, and the owner's ruling forbids that: *"Count is a result… The amount also should
    // come from the seed."* A galaxy's population is the volume its shape encloses at the density it
    // drew — `galaxy_population` — and nothing states it.
    //
    // They were left dead for a few minutes during the change, which was worse than either keeping
    // or removing them: `growing_the_system_count_does_not_move_the_systems_already_placed` still
    // set them and still expected two different worlds, and got one world twice. A knob that is read
    // by nothing fails silently at every call site that still trusts it.
    //
    // TO LOOK AT A SMALL GALAXY, ASK FOR A SMALL GALAXY — a smaller rim, the same law, the count
    // following as it always does. The tests' `galaxy_holding` helper solves the radius exactly.)
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
    /// THE GALAXY'S OUTER RADIUS — its RIM. Every star system is placed somewhere INSIDE it, at a
    /// distance the seed draws, and none is placed ON it.
    ///
    /// ★ RENAMED FROM `galaxy_rim_r_m` AT S12 (2026-08-28), BECAUSE THE MEANING CHANGED AND THE OLD
    /// NAME KEPT LYING. It used to be "the radius of the ring the non-origin systems are spaced
    /// around", and under the shell that was also the exact distance to any sibling — so callers
    /// read it as "how far away the next star is" and were right. The owner refused the shell
    /// (owner_decisions_2026-08-27_galaxy_shape.md): a shell puts every star at the SAME distance,
    /// which no galaxy does. The placement now draws each system's own radius, so this number is an
    /// upper bound and nothing else.
    ///
    /// TWO GATES FAILED ON THE OLD NAME THE DAY THE SHAPE LANDED, both by reading it as a distance:
    /// the frame fixture's story distance, and the warp flight's park (which computed "stop when
    /// within rim − park of the destination" for a destination sitting at 91 % of the rim, so the
    /// ship's stop condition was already true and it never moved). The rename makes every reader a
    /// compiler error once, which is the only reliable way to make each one be looked at.
    ///
    /// ★ TO ASK HOW FAR APART TWO SYSTEMS ARE, MEASURE THEM. Read both placements off THE world and
    /// subtract; never read this.
    ///
    /// Must leave every system's boundary disjoint from every other's AND inside the galaxy —
    /// overlapping systems would make "which realm contains this position" ambiguous, which is the one
    /// question the whole authority model rests on.
    ///
    /// HOW MANY systems is NOT here: it is drawn from the galaxy's own census
    /// ([`GalaxyConfig::system_count_lo`]..=[`GalaxyConfig::system_count_hi`]) against the galaxy's seed,
    /// so two galaxies from one universe differ. A second count field here would be a second source of
    /// truth for the same fact.
    pub galaxy_rim_r_m: f64,
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

    /// ★ THE BAND SIZED FROM THE CEILING IN FORCE AT THIS BOUNDARY'S OWN SURFACE (slice S6).
    ///
    /// **Why the surface's OWN ceiling and not its parent's.** The approach governor lowers a subject's
    /// ceiling onto the body it is approaching, so a thing arrives at THIS body's speed and never at the
    /// speed the space around it allows. Sizing against the parent's ceiling would size every band for a
    /// speed no lawful subject can hold there — and, measured, would ask for bands thousands of times
    /// larger than the bodies they wrap.
    ///
    /// **Why a fixed tick and a fixed foot speed.** Both are constants of the solve. A band is part of
    /// the world's geometry, and two clusters must boot the identical world however fast they tick and
    /// however fast they let a person walk.
    ///
    /// **The floor.** A band never goes below the shipped edges, so a small realm keeps exactly the band
    /// it has today. Measured on THE world: 0 of 13,144 boundaries take the floor, and 0 have the foot
    /// speed bind — every band that ships is set by the body's own size.
    ///
    /// The 1:2 inset-to-outset shape is preserved at every size: the acquire edge sits one third of the
    /// band inside the surface and the release edge two thirds outside, which is what makes acquiring
    /// strictly harder than holding at every scale rather than only at the shipped one.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] if the resolved edges are degenerate — impossible while the floor is
    /// positive, and kept fallible so the ctor stays the only way to build a band.
    pub fn build_for_shape(&self, shape: &Boundary) -> Result<ContainmentBand, BandError> {
        let extent_m = shape.circumscribed_extent();
        let ceiling = realm_speed_cap_mps(extent_m, GEOMETRY_V_FOOT_MPS, T_TRAVERSE_S);
        // The τ-FREE form. The true band carries τ = T_WAKE, a MEASURED boot latency, and a world's
        // geometry may never be a function of how fast a shard happens to boot — that would fork the one
        // containment answer between processes. The band-solvability fence bounds the τ term at a
        // doubling, so the headroom factor is its lawful upper bound. This is the SAME expression the
        // outer geometry already consumed to place the galaxy, so the shells and the bands cannot
        // disagree about what a band costs.
        let need_m = band_for_speed(ceiling, GEOMETRY_TICK_DT_S, BAND_TICKS_N) * BAND_TAU_HEADROOM;
        let floor_m = self.inset_m + self.outset_m;
        // ★ THE CLAMP, AND WHY IT IS NOT OPTIONAL. The acquire edge sits one third of the band INSIDE
        // the surface, so a band larger than the body puts that edge past the body's own centre and the
        // realm becomes impossible to enter at any speed. That is not a theoretical worry: without this
        // clamp seven re-home tests went from one crossing to none, because a station five metres across
        // was given a sixty-metre band.
        //
        // The bound is the INSCRIBED extent — the largest sphere the shape fully contains — because that
        // is the smallest direction, and a band that fits the widest direction can still swallow the
        // narrowest. Half of it, so the acquire edge stays at five sixths of the body.
        //
        // A boundary the clamp binds on is one whose own ceiling is too fast for its own size. That is a
        // real condition and the clamp does not hide it: the band stops growing and the ceiling fence is
        // what says so.
        let cap_m = shape.inscribed_extent() * BAND_EXTENT_CLAMP;
        let width_m = need_m.max(floor_m).min(cap_m);
        let scale = width_m / floor_m;
        ContainmentBand::for_containment_velocity_safe(
            self.inset_m * scale,
            self.outset_m * scale,
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
pub(crate) const VISUAL_AOI_GRACE_TICKS: u32 = 20;
/// Extra velocity-safety margin folded into the dead-zone widening (beyond `K_SAFETY`).
pub(crate) const VISUAL_AOI_K_SAFETY_EXTRA: f64 = 0.5;
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
            },
            stellar: StellarConfig {
                system_soi_r_m: SYSTEM_SOI_R_M,
                imf_slope: IMF_SLOPE,
                mass_lo_msun: IMF_MASS_LO_MSUN,
                mass_hi_msun: imf_mass_hi_msun(),
                mlr_segments: SpectralClass::MLR_SEGMENTS,
                // Where the walk roster's second star already sat. It is no longer inert: with ONE
                // world, this preset drives the same generator as everything else, and a zero ring
                // would stack both stars on the origin — two authorities over one point.
                galaxy_rim_r_m: SYSTEM_B_OFFSET_M,
            },
            // `0` planets: the walk fixture forest is ambient-only (no `Orbital` body) — the
            // world derives N below.
            planet: planet_config(0),
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
        cfg.planet.n_planets = derived_world_planet_count(&cfg.planet);
        // ▲ THE OUTER GEOMETRY (real-scale addendum §A2 — the four changed numbers, derivations
        // at their consts): the universe from the storage fence (2⁵¹ m), the galaxy from the
        // τ-free outset (`R_uni − outset`), the placement radius from the reserved clearance
        // (`R_gal − clearance` = 0.15843 ly), the compression χ = 24.568× stated at the
        // census consts.
        cfg.scale.universe_r_m = REAL_UNIVERSE_R_M;
        cfg.scale.galaxy_r_m = REAL_GALAXY_R_M;
        cfg.stellar.galaxy_rim_r_m = real_placement_r_m();
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
