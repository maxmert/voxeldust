//! Real astrophysical taxonomy as DATA (D-45(a) Slice 2) — the classification + sampling
//! layer the seed universe generator (Slice 3) draws on.
//!
//! Three registries — [`GalaxyType`], [`SpectralClass`], [`PlanetType`] — replicate the
//! proven `entity_kind.rs` `KindDef` idiom EXACTLY: a `#[repr(u8)]` enum + `ALL` array +
//! `from_tag` (unknown tags are an ERROR, never a Default — HR2) + `def()` static
//! descriptor, guarded by a drift tripwire. Adding a variant is a compile error until it is
//! registered everywhere.
//!
//! The distributions are REAL and every sampler is a CLOSED-FORM inverse-CDF — ZERO
//! rejection loops — so each draw is exactly one `SplitMix64::next_f64` and the per-realm
//! stream stays bit-reproducible across shards (HR1). Distribution PARAMETERS are passed as
//! ARGUMENTS; the named consts here are documented, cited defaults for tests — the ONE
//! config home (`UniverseConfig`) that gathers and seed-perturbs them is Slice 3, not here.
//!
//! [`ProfileKind`] is the core-side capability TAG of a generated body; the ONE total
//! `ProfileKind -> ShardProfile` map, `capability::profile_for`, lives in vd-sim (a
//! `ShardProfile` is a sim type; the tag flows DOWN the `bins->node->sim->wire->core` arrow,
//! never a reverse edge). New body kind = one data row (HR3/HR4).
//!
//! Determinism note: the transcendental steps here (`ln`/`sqrt`/`powf` in the luminosity,
//! frost-line and Rayleigh closed forms) are the SAME libm class as `celestial.rs` and are
//! evaluated at BOOT / seed time (drawn once per system), not per tick — the cross-host
//! bit-equality gate is SPIKE-6a (step-2/P4), the same deferral the module docs there carry.

use serde::{Deserialize, Serialize};

pub use vd_core::taxonomy::UnknownTaxonTag;

// ===== Galaxy morphology (Nair & Abraham 2010, ApJS 186:427) =========================

/// Spiral fraction of the luminous-galaxy census (cumulative slot 0).
const GALAXY_C_SPIRAL: f64 = 0.72;
/// Spiral + elliptical cumulative fraction (slot 1); irregular is the 0.10 remainder.
const GALAXY_C_ELLIPTICAL: f64 = 0.90;

/// Hubble-sequence morphological class of a generated galaxy (Nair & Abraham 2010).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GalaxyType {
    Spiral = 0,
    Elliptical = 1,
    Irregular = 2,
}

/// Static per-galaxy-class record (the `KindDef` analog). NOT serde-carried: a
/// `&'static str` label cannot be produced from a transient deserialize buffer, and the
/// taxonomy is off-wire DATA (HR1) — the fieldless enum carries the wire discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GalaxyDef {
    /// Pins the descriptor to its variant's discriminant (the coherence loop checks it).
    pub tag: u8,
    pub name: &'static str,
}

pub static SPIRAL_DEF: GalaxyDef = GalaxyDef {
    tag: 0,
    name: "Spiral",
};
pub static ELLIPTICAL_DEF: GalaxyDef = GalaxyDef {
    tag: 1,
    name: "Elliptical",
};
pub static IRREGULAR_DEF: GalaxyDef = GalaxyDef {
    tag: 2,
    name: "Irregular",
};

impl GalaxyType {
    /// All galaxy classes in tag order; the classifier + tripwire are total over this set.
    pub const ALL: [GalaxyType; 3] = [
        GalaxyType::Spiral,
        GalaxyType::Elliptical,
        GalaxyType::Irregular,
    ];

    /// The census cumulative vector (Nair & Abraham 2010) passed to [`sample_galaxy_type`];
    /// Slice-3-tunable. `[c_spiral, c_spiral+c_elliptical]`, irregular is the remainder.
    pub const CANONICAL_CUMULATIVE: [f64; 2] = [GALAXY_C_SPIRAL, GALAXY_C_ELLIPTICAL];

    /// Recover a class from its tag byte; unknown tags are an error, never a Default (HR2).
    pub fn from_tag(tag: u8) -> Result<GalaxyType, UnknownTaxonTag> {
        match tag {
            0 => Ok(GalaxyType::Spiral),
            1 => Ok(GalaxyType::Elliptical),
            2 => Ok(GalaxyType::Irregular),
            other => Err(UnknownTaxonTag {
                kind: "galaxy",
                tag: other,
            }),
        }
    }

    /// This class's static descriptor; total over `ALL`.
    #[must_use]
    pub fn def(self) -> &'static GalaxyDef {
        match self {
            GalaxyType::Spiral => &SPIRAL_DEF,
            GalaxyType::Elliptical => &ELLIPTICAL_DEF,
            GalaxyType::Irregular => &IRREGULAR_DEF,
        }
    }
}

// ===== Spectral class (Pecaut & Mamajek 2013, ApJS 208:9) ============================

// MK main-sequence lower mass boundaries, descending (solar masses).
const MASS_O_LO_MSUN: f64 = 16.0;
const MASS_B_LO_MSUN: f64 = 2.1;
const MASS_A_LO_MSUN: f64 = 1.4;
const MASS_F_LO_MSUN: f64 = 1.04;
const MASS_G_LO_MSUN: f64 = 0.8;
/// K widened to the ~M2 mass; the K/M cut is fuzzy (Pecaut-Mamajek M0V ~0.57 Msun) — a
/// documented modeling choice, not an oversight.
const MASS_K_LO_MSUN: f64 = 0.45;
/// The hydrogen-burning limit; the classifier's domain floor (see [`classify_spectral`]).
const MASS_M_LO_MSUN: f64 = 0.08;

// Effective-temperature band lower edges (K); each class's upper edge is the next class up.
const TEFF_O_LO_K: f64 = 30_000.0;
const TEFF_B_LO_K: f64 = 10_000.0;
const TEFF_A_LO_K: f64 = 7_500.0;
const TEFF_F_LO_K: f64 = 6_000.0;
const TEFF_G_LO_K: f64 = 5_200.0;
const TEFF_K_LO_K: f64 = 3_700.0;
const TEFF_M_LO_K: f64 = 2_400.0;

/// Morgan-Keenan main-sequence class from stellar mass (Pecaut & Mamajek 2013).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SpectralClass {
    O = 0,
    B = 1,
    A = 2,
    F = 3,
    G = 4,
    K = 5,
    M = 6,
}

/// Static per-class record: `mass_lo_msun` is the inclusive lower mass boundary (the
/// descending-scan key); the T_eff band is coherence-checked. Not serde-carried (off-wire).
///
/// `xuv_bol_ratio` (taxonomy arc T0): the TIME-INTEGRATED X-ray/EUV over bolometric output
/// ratio of the class — the quantity that drives atmospheric escape, which for late classes
/// is orders of magnitude above the present-epoch solar value because low-mass dwarfs stay
/// magnetically saturated for gigayears. Sources: Wright et al. 2011 (ApJ 743:48, the
/// saturated `log(L_X/L_bol) = -3.13` activity floor), France et al. 2013 (ApJ 763:149,
/// M-dwarf EUV). G is the normalising class (the Sun, ~1e-6); O/B/A have radiative
/// envelopes and no dynamo (~1e-7). Consumed by [`xuv_rel_of`] for the cosmic shoreline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralDef {
    pub tag: u8,
    pub name: &'static str,
    pub mass_lo_msun: f64,
    pub teff_lo_k: f64,
    pub teff_hi_k: f64,
    pub xuv_bol_ratio: f64,
}

impl SpectralDef {
    /// Coherence rules (checked over `SpectralClass::ALL`): a positive lower mass bound, a
    /// well-ordered, positive T_eff band, and a positive XUV/bolometric ratio. Named AND'd
    /// booleans (HR5: each clause's false side is driven independently by
    /// `taxa_incoherent_defs_are_detected`). Bitwise `&` (no short-circuit region).
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        let mass_positive = self.mass_lo_msun > 0.0;
        let band_ordered = self.teff_lo_k < self.teff_hi_k;
        let band_positive = self.teff_lo_k > 0.0;
        let xuv_positive = self.xuv_bol_ratio > 0.0;
        mass_positive & band_ordered & band_positive & xuv_positive
    }
}

// Time-integrated XUV/bolometric ratios per class (see the `SpectralDef` field doc; T0).
// UNPINNED-COEFFICIENT class re-read at implementation: Wright 2011 saturated floor,
// France 2013 M-dwarf EUV — order-of-magnitude class descriptors, not per-star fits.
const XUV_BOL_RADIATIVE: f64 = 1.0e-7; // O, B, A — no convective dynamo
const XUV_BOL_F: f64 = 3.0e-7;
/// The Sun at the present epoch — the normalising class for [`xuv_rel_of`].
const XUV_BOL_G: f64 = 1.0e-6;
const XUV_BOL_K: f64 = 3.0e-6;
/// Saturated M dwarfs, gigayear-integrated (France 2013; Wright 2011 `log(Lx/Lbol) = -3.13`).
const XUV_BOL_M: f64 = 1.0e-4;

pub static O_DEF: SpectralDef = SpectralDef {
    tag: 0,
    name: "O",
    mass_lo_msun: MASS_O_LO_MSUN,
    teff_lo_k: TEFF_O_LO_K,
    teff_hi_k: f64::INFINITY,
    xuv_bol_ratio: XUV_BOL_RADIATIVE,
};
pub static B_DEF: SpectralDef = SpectralDef {
    tag: 1,
    name: "B",
    mass_lo_msun: MASS_B_LO_MSUN,
    teff_lo_k: TEFF_B_LO_K,
    teff_hi_k: TEFF_O_LO_K,
    xuv_bol_ratio: XUV_BOL_RADIATIVE,
};
pub static A_DEF: SpectralDef = SpectralDef {
    tag: 2,
    name: "A",
    mass_lo_msun: MASS_A_LO_MSUN,
    teff_lo_k: TEFF_A_LO_K,
    teff_hi_k: TEFF_B_LO_K,
    xuv_bol_ratio: XUV_BOL_RADIATIVE,
};
pub static F_DEF: SpectralDef = SpectralDef {
    tag: 3,
    name: "F",
    mass_lo_msun: MASS_F_LO_MSUN,
    teff_lo_k: TEFF_F_LO_K,
    teff_hi_k: TEFF_A_LO_K,
    xuv_bol_ratio: XUV_BOL_F,
};
pub static G_DEF: SpectralDef = SpectralDef {
    tag: 4,
    name: "G",
    mass_lo_msun: MASS_G_LO_MSUN,
    teff_lo_k: TEFF_G_LO_K,
    teff_hi_k: TEFF_F_LO_K,
    xuv_bol_ratio: XUV_BOL_G,
};
pub static K_DEF: SpectralDef = SpectralDef {
    tag: 5,
    name: "K",
    mass_lo_msun: MASS_K_LO_MSUN,
    teff_lo_k: TEFF_K_LO_K,
    teff_hi_k: TEFF_G_LO_K,
    xuv_bol_ratio: XUV_BOL_K,
};
pub static M_DEF: SpectralDef = SpectralDef {
    tag: 6,
    name: "M",
    mass_lo_msun: MASS_M_LO_MSUN,
    teff_lo_k: TEFF_M_LO_K,
    teff_hi_k: TEFF_K_LO_K,
    xuv_bol_ratio: XUV_BOL_M,
};

impl SpectralClass {
    /// All spectral classes in descending-mass tag order.
    pub const ALL: [SpectralClass; 7] = [
        SpectralClass::O,
        SpectralClass::B,
        SpectralClass::A,
        SpectralClass::F,
        SpectralClass::G,
        SpectralClass::K,
        SpectralClass::M,
    ];

    /// The default MK mass boundaries as a passable table (descending); the classifier takes
    /// it as an arg so Slice 3 can retune. Values are named consts, not inline literals.
    pub const MASS_BOUNDS: [(SpectralClass, f64); 7] = [
        (SpectralClass::O, MASS_O_LO_MSUN),
        (SpectralClass::B, MASS_B_LO_MSUN),
        (SpectralClass::A, MASS_A_LO_MSUN),
        (SpectralClass::F, MASS_F_LO_MSUN),
        (SpectralClass::G, MASS_G_LO_MSUN),
        (SpectralClass::K, MASS_K_LO_MSUN),
        (SpectralClass::M, MASS_M_LO_MSUN),
    ];

    /// Broken-power-law mass-luminosity segments `(mass_hi, coeff, exponent)` ascending
    /// (Duric 2004, "Advanced Astrophysics"): `L/Lsun = coeff * M^exponent`. Nearly
    /// continuous at the breaks (a `mass_luminosity_is_continuous` tripwire guards it).
    pub const MLR_SEGMENTS: [(f64, f64, f64); 3] =
        [(0.43, 0.23, 2.3), (2.0, 1.0, 4.0), (55.0, 1.4, 3.5)];

    /// Recover a class from its tag byte; unknown tags error (HR2), never Default.
    pub fn from_tag(tag: u8) -> Result<SpectralClass, UnknownTaxonTag> {
        match tag {
            0 => Ok(SpectralClass::O),
            1 => Ok(SpectralClass::B),
            2 => Ok(SpectralClass::A),
            3 => Ok(SpectralClass::F),
            4 => Ok(SpectralClass::G),
            5 => Ok(SpectralClass::K),
            6 => Ok(SpectralClass::M),
            other => Err(UnknownTaxonTag {
                kind: "spectral",
                tag: other,
            }),
        }
    }

    /// This class's static descriptor; total over `ALL`.
    #[must_use]
    pub fn def(self) -> &'static SpectralDef {
        match self {
            SpectralClass::O => &O_DEF,
            SpectralClass::B => &B_DEF,
            SpectralClass::A => &A_DEF,
            SpectralClass::F => &F_DEF,
            SpectralClass::G => &G_DEF,
            SpectralClass::K => &K_DEF,
            SpectralClass::M => &M_DEF,
        }
    }
}

/// Descending-threshold scan: the first class whose lower mass bound `<= mass_msun` wins.
/// `bounds` is the class->lower-bound table (`SpectralClass::MASS_BOUNDS` supplies the
/// defaults). Domain is main-sequence mass `>= MASS_M_LO_MSUN`; a mass below the
/// hydrogen-burning limit clamps to `M` (Slice 3's IMF sampler guarantees `m_lo >= 0.08`,
/// the cross-slice precondition — mirroring `KEPLER_ECC_MAX`).
#[must_use]
pub fn classify_spectral(mass_msun: f64, bounds: &[(SpectralClass, f64); 7]) -> SpectralClass {
    bounds
        .iter()
        .find(|(_, lo)| mass_msun >= *lo)
        .map(|(class, _)| *class)
        .unwrap_or(SpectralClass::M)
}

/// Broken-power-law mass-luminosity relation `L/Lsun = coeff * M^exponent` (Duric 2004).
/// `segments` = `[(mass_hi, coeff, exponent); 3]` ascending; the high tail (`M > 55`) reuses
/// the top segment. Feeds the apparent-magnitude star map (`M_abs = 4.83 - 2.5*log10(L)`).
#[must_use]
pub fn main_sequence_luminosity(mass_msun: f64, segments: &[(f64, f64, f64); 3]) -> f64 {
    segmented_power_law(mass_msun, segments)
}

/// Flux-balance habitable-zone radius `sqrt(L/flux_limit)` in AU (Kopparapu 2013): pass the
/// moist-inner `S_in~1.1` or conservative-outer `S_out~0.53` flux limit for the edges, or
/// `1.0` for the `r = sqrt(L)` centre.
#[must_use]
pub fn habitable_zone_radius_au(luminosity_lsun: f64, flux_limit: f64) -> f64 {
    (luminosity_lsun / flux_limit).sqrt()
}

// ===== Planet type (taxonomy arc T0 — the 2x2 + giant escape) ========================

/// Solar-nebula water-ice condensation radius at `L = Lsun` (Hayashi 1981), scaled by
/// `sqrt(L)`; passed to [`frost_line_radius_au`].
pub const FROST_COEFF_AU: f64 = 2.7;
/// Pollack et al. 1996 critical core mass for runaway gas accretion (Earth masses) — gates
/// whether the Jovian outcome is reachable at all before the mass break is tested.
const M_CORE_CRIT_MEARTH: f64 = 10.0;

/// Compositional/formation planet class. The classifier ([`classify_planet`]) derives it
/// from two physically motivated bits — WHERE the core formed (inside/beyond the frost
/// line) and WHETHER it kept its H/He envelope (the Kepler radius valley) — plus the
/// measured Neptunian->Jovian mass break for giants. Occurrence grounding: Fressin et al.
/// 2013 (ApJ 766:81); the radius valley: Fulton et al. 2017 (AJ 154:109).
///
/// `Ocean` (tag 1) was RELABELLED by the taxonomy arc (a Stream-Law derivation correction,
/// declared): it used to mean a 2-10 Mearth body INSIDE the frost line; it now means an
/// ice-rich core BEYOND the line whose envelope was lost (Zeng et al. 2019, PNAS 116:9723
/// water worlds). Same wire discriminant, different meaning, stated here once.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PlanetType {
    Rocky = 0,
    Ocean = 1,
    GasGiant = 2,
    IceGiant = 3,
    /// The single most common planet in the galaxy (2-4 Rearth, occurrence 0.152/star,
    /// Fressin 2013) — an envelope-retaining core inside the frost line. APPENDED at tag 4
    /// (postcard append-only discipline).
    SubNeptune = 4,
}

/// Static per-type record. `is_giant` is a clean per-type fact; `requires_frost_line` marks
/// classes whose defining ices condense only beyond the line (`Ocean`, `IceGiant`);
/// `retains_h2_envelope` marks classes whose radius includes a kept H/He envelope
/// (`SubNeptune`, `IceGiant`, `GasGiant`). The Bond albedos are MEASURED solar-system
/// values (never the drawn geometric albedo — a different physical quantity; the
/// reflected-light marker keeps its own draw): Rocky airless 0.10 (Mercury 0.088, Moon
/// 0.11), Rocky with air 0.30 (Earth 0.306), Ocean 0.30 (water/cloud), SubNeptune 0.30
/// (GJ 1214b class, UNPINNED COEFFICIENT), IceGiant 0.29 (Neptune 0.290, Uranus 0.300),
/// GasGiant 0.34 (Jupiter 0.343, Saturn 0.342). Not serde-carried (off-wire descriptor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetDef {
    pub tag: u8,
    pub name: &'static str,
    pub is_giant: bool,
    /// This type can form ONLY beyond the frost line (`Ocean`, `IceGiant`).
    pub requires_frost_line: bool,
    /// This type kept its primordial H/He envelope (`SubNeptune`, `IceGiant`, `GasGiant`).
    pub retains_h2_envelope: bool,
    /// Bond albedo when an atmosphere/envelope is present (feeds `T_eq`, never the marker).
    pub bond_albedo_with_atmosphere: f64,
    /// Bond albedo for the airless surface (differs from the atmosphere row only for Rocky).
    pub bond_albedo_airless: f64,
}

impl PlanetDef {
    /// Coherence: a giant always retains its envelope (runaway accretion IS envelope
    /// retention — the old `!requires_frost_line || is_giant` clause was retired when
    /// `Ocean` was re-grounded beyond the line), and both Bond albedos are physical
    /// (`0 <= A < 1`). Named bitwise-AND'd booleans (HR5: each clause independently driven
    /// false by `taxa_incoherent_defs_are_detected`).
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        let giant_retains = !self.is_giant | self.retains_h2_envelope;
        let albedo_air_physical = (0.0..1.0).contains(&self.bond_albedo_with_atmosphere);
        let albedo_bare_physical = (0.0..1.0).contains(&self.bond_albedo_airless);
        giant_retains & albedo_air_physical & albedo_bare_physical
    }
}

pub static ROCKY_DEF: PlanetDef = PlanetDef {
    tag: 0,
    name: "Rocky",
    is_giant: false,
    requires_frost_line: false,
    retains_h2_envelope: false,
    bond_albedo_with_atmosphere: 0.30,
    bond_albedo_airless: 0.10,
};
pub static OCEAN_DEF: PlanetDef = PlanetDef {
    tag: 1,
    name: "Ocean",
    is_giant: false,
    requires_frost_line: true,
    retains_h2_envelope: false,
    bond_albedo_with_atmosphere: 0.30,
    bond_albedo_airless: 0.30,
};
pub static GAS_GIANT_DEF: PlanetDef = PlanetDef {
    tag: 2,
    name: "GasGiant",
    is_giant: true,
    requires_frost_line: false,
    retains_h2_envelope: true,
    bond_albedo_with_atmosphere: 0.34,
    bond_albedo_airless: 0.34,
};
pub static ICE_GIANT_DEF: PlanetDef = PlanetDef {
    tag: 3,
    name: "IceGiant",
    is_giant: true,
    requires_frost_line: true,
    retains_h2_envelope: true,
    bond_albedo_with_atmosphere: 0.29,
    bond_albedo_airless: 0.29,
};
pub static SUB_NEPTUNE_DEF: PlanetDef = PlanetDef {
    tag: 4,
    name: "SubNeptune",
    is_giant: false,
    requires_frost_line: false,
    retains_h2_envelope: true,
    bond_albedo_with_atmosphere: 0.30,
    bond_albedo_airless: 0.30,
};

/// The classifier thresholds (params-as-args; `CANONICAL` supplies the cited defaults).
/// `m_ocean_lo_mearth` is DELETED (the ocean/rocky split is the frost line now, never a
/// mass cut); the valley parameters joined (Martinez et al. 2019 ApJ 875:29 anchor;
/// Lopez & Rice 2018 / Fulton & Petigura 2018 `S^0.11` scaling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrostThresholds {
    /// The Neptunian->Jovian break: `PMR_SEGMENTS[1].0` = 131.6 Mearth (Chen & Kipping
    /// 2017) — a table entry the tree already holds, never a hand-picked literal.
    pub m_gas_mearth: f64,
    /// Pollack 1996 runaway-accretion critical core mass (10 Mearth).
    pub m_core_crit_mearth: f64,
    /// The radius valley at `S = 1 S_earth` ([`RADIUS_VALLEY_1SEARTH_REARTH`]).
    pub valley_r1_rearth: f64,
    /// The valley's insolation exponent ([`RADIUS_VALLEY_INSOLATION_EXP`]).
    pub valley_insolation_exp: f64,
}

impl FrostThresholds {
    pub const CANONICAL: FrostThresholds = FrostThresholds {
        m_gas_mearth: PMR_SEGMENTS[1].0,
        m_core_crit_mearth: M_CORE_CRIT_MEARTH,
        valley_r1_rearth: RADIUS_VALLEY_1SEARTH_REARTH,
        valley_insolation_exp: RADIUS_VALLEY_INSOLATION_EXP,
    };
}

impl PlanetType {
    /// All planet types in tag order.
    pub const ALL: [PlanetType; 5] = [
        PlanetType::Rocky,
        PlanetType::Ocean,
        PlanetType::GasGiant,
        PlanetType::IceGiant,
        PlanetType::SubNeptune,
    ];

    /// Recover a type from its tag byte; unknown tags error (HR2), never Default.
    pub fn from_tag(tag: u8) -> Result<PlanetType, UnknownTaxonTag> {
        match tag {
            0 => Ok(PlanetType::Rocky),
            1 => Ok(PlanetType::Ocean),
            2 => Ok(PlanetType::GasGiant),
            3 => Ok(PlanetType::IceGiant),
            4 => Ok(PlanetType::SubNeptune),
            other => Err(UnknownTaxonTag {
                kind: "planet",
                tag: other,
            }),
        }
    }

    /// This type's static descriptor; total over `ALL`.
    #[must_use]
    pub fn def(self) -> &'static PlanetDef {
        match self {
            PlanetType::Rocky => &ROCKY_DEF,
            PlanetType::Ocean => &OCEAN_DEF,
            PlanetType::GasGiant => &GAS_GIANT_DEF,
            PlanetType::IceGiant => &ICE_GIANT_DEF,
            PlanetType::SubNeptune => &SUB_NEPTUNE_DEF,
        }
    }
}

/// Water-ice condensation radius `frost_coeff_au * sqrt(L/Lsun)` in AU (Hayashi 1981).
#[must_use]
pub fn frost_line_radius_au(luminosity_lsun: f64, frost_coeff_au: f64) -> f64 {
    frost_coeff_au * luminosity_lsun.sqrt()
}

/// The class Bond albedo for `T_eq` (the class table above — MEASURED Bond values, never
/// the drawn geometric albedo; conflating the two is up to 50% wrong into `T ~ (1-A)^0.25`).
#[must_use]
pub fn bond_albedo_of(class: PlanetType, has_atmosphere: bool) -> f64 {
    let def = class.def();
    if has_atmosphere {
        def.bond_albedo_with_atmosphere
    } else {
        def.bond_albedo_airless
    }
}

/// THE CLASSIFIER (taxonomy arc, replaces the Pollack threshold tree): two derived bits and
/// one giant escape, no albedo and no temperature on the input side (the evaluation-order
/// contract: class -> shoreline -> atmosphere bit -> Bond albedo -> `T_eq`; insolation in,
/// temperature only ever OUT).
///
/// ```text
/// giant escape:  M >= 131.6 Mearth (Chen-Kipping Jovian break) AND M >= 10 Mearth
///                (Pollack runaway core) => GasGiant, either frost side
/// BIT 1 formed:  orbit < frost line?
/// BIT 2 kept:    R_core(M) > R_valley(S)?   (rock mass-radius vs the Kepler radius valley)
///                       kept: NO            kept: YES
///   inside line:        Rocky               SubNeptune
///   beyond line:        Ocean               IceGiant
/// ```
///
/// Solar-system fidelity is 7 of 8 with Saturn the single named miss (Saturn is 0.299 M_J;
/// the published Chen-Kipping break itself puts it on the Neptunian branch) — pinned by the
/// classifier golden, not tuned away.
#[must_use]
pub fn classify_planet(
    orbit_au: f64,
    frost_line_au: f64,
    mass_mearth: f64,
    insolation_rel: f64,
    th: FrostThresholds,
) -> PlanetType {
    let runaway_core = mass_mearth >= th.m_core_crit_mearth;
    let jovian_mass = mass_mearth >= th.m_gas_mearth;
    if runaway_core & jovian_mass {
        return PlanetType::GasGiant;
    }
    let core_rearth = segmented_power_law(mass_mearth, &ROCK_MR_SEGMENTS);
    let valley_rearth = radius_valley_rearth(
        insolation_rel,
        th.valley_r1_rearth,
        th.valley_insolation_exp,
    );
    let kept_envelope = core_rearth > valley_rearth;
    let inside_frost = orbit_au < frost_line_au;
    match (inside_frost, kept_envelope) {
        (true, false) => PlanetType::Rocky,
        (true, true) => PlanetType::SubNeptune,
        (false, false) => PlanetType::Ocean,
        (false, true) => PlanetType::IceGiant,
    }
}

// ===== T0 pure laws: constants, mass-radius tables, temperature, escape ==============
//
// Every constant below is a CITED physical value (the no-magic-numbers discipline: cited
// physics constants and published fit coefficients, gathered here once, never inline).

/// Nominal solar luminosity in W — IAU 2015 Resolution B3, exactly defined.
pub const L_SUN_W: f64 = 3.828e26;
/// Nominal solar radius in m — IAU 2015 Resolution B3.
pub const R_SUN_M: f64 = 6.957e8;
/// Solar mass in kg (the `G*M_sun` split the tree's `G = 6.674e-11` implies).
pub const M_SUN_KG: f64 = 1.989e30;
/// Earth mass in kg (IAU nominal).
pub const M_EARTH_KG: f64 = 5.972e24;
/// Earth MEAN radius in m — the radius the Chen-Kipping / Zeng Earth-unit fits are stated in.
pub const R_EARTH_M: f64 = 6.371e6;
/// The astronomical unit in m — IAU 2012 Resolution B2, exactly defined.
pub const AU_M: f64 = 1.495978707e11;
/// Stefan-Boltzmann constant, W m^-2 K^-4 — CODATA 2018, exact under the SI redefinition.
pub const STEFAN_BOLTZMANN_W_M2_K4: f64 = 5.670374419e-8;
/// Boltzmann constant, J/K — SI 2019, exactly defined.
pub const BOLTZMANN_J_K: f64 = 1.380649e-23;
/// Atomic mass unit, kg — CODATA 2018.
pub const ATOMIC_MASS_KG: f64 = 1.66053906660e-27;

/// Segmented broken-power-law `coeff * x^exponent` over ascending `(x_hi, coeff, exponent)`
/// rows; `x` above every break reuses the top row. THE one evaluator every mass-radius /
/// mass-luminosity table in this file goes through (HR3 — one machinery, tables as data).
#[must_use]
pub fn segmented_power_law(x: f64, segments: &[(f64, f64, f64)]) -> f64 {
    let (_, coeff, exponent) = segments
        .iter()
        .find(|(hi, _, _)| x <= *hi)
        .copied()
        .unwrap_or(segments[segments.len() - 1]);
    coeff * x.powf(exponent)
}

/// Chen & Kipping 2017 (ApJ 834:17, "Forecaster") planet mass-radius broken power law in
/// EARTH units: Terran / Neptunian / Jovian segments; the 131.6 Mearth Neptunian->Jovian
/// break doubles as the classifier's gas-giant threshold (`FrostThresholds::CANONICAL`).
pub const PMR_SEGMENTS: [(f64, f64, f64); 3] = [
    (2.04, 1.008, 0.279),
    (131.6, 0.808_11, 0.589),
    (f64::MAX, 17.7346, -0.044),
];

/// Demircan & Kahraman 1991 (Ap&SS 181:313) stellar mass-radius segments in SOLAR units:
/// `R/Rsun = 1.06*M^0.945` below 1.66 Msun, `1.2917*M^0.555` above.
pub const STELLAR_MR_SEGMENTS: [(f64, f64, f64); 2] =
    [(1.66, 1.06, 0.945), (f64::MAX, 1.2917, 0.555)];

/// Zeng, Sasselov & Jacobsen 2016 (ApJ 819:127) rocky-core mass-radius (CMF 0.33), Earth
/// units: `R = M^0.27`.
pub const ROCK_MR_SEGMENTS: [(f64, f64, f64); 1] = [(f64::MAX, 1.0, 0.27)];

/// Zeng et al. 2019 (PNAS 116:9723) 50% H2O ice-rock mass-radius, Earth units:
/// `R = 1.24 * M^0.27` (coefficient re-read from the paper at implementation).
pub const ICE_MR_SEGMENTS: [(f64, f64, f64); 1] = [(f64::MAX, 1.24, 0.27)];

/// A star's photospheric radius in metres from its drawn mass (Demircan & Kahraman 1991
/// through the one segmented evaluator). ONE function, two call sites by design: the
/// System's look and the Star realm's own look both read THIS (taxonomy design par 5.2).
#[must_use]
pub fn star_radius_m(mass_msun: f64) -> f64 {
    R_SUN_M * segmented_power_law(mass_msun, &STELLAR_MR_SEGMENTS)
}

/// A planet's TOTAL radius in metres from its drawn mass (Chen & Kipping 2017) — the
/// population-statistical radius the re-solve uses for looks/SOIs before the taxonomy's
/// per-class composition refines it.
#[must_use]
pub fn planet_radius_m(mass_mearth: f64) -> f64 {
    R_EARTH_M * segmented_power_law(mass_mearth, &PMR_SEGMENTS)
}

// ---- The radius valley (the classifier's BIT 2) --------------------------------------

/// The Kepler radius valley at `S = 1 S_earth`, Earth radii — a normalisation of the
/// published `R_valley ~ S^0.11` slope (Lopez & Rice 2018; Fulton & Petigura 2018) to the
/// published anchor `R_valley = 1.90 Rearth at S = 121 S_earth` (Martinez et al. 2019,
/// ApJ 875:29): `1.121 * 121^0.11 = 1.8998`. The ONE invented constant in the taxonomy arc,
/// and it is a normalisation of two published values, pinned by its own test.
pub const RADIUS_VALLEY_1SEARTH_REARTH: f64 = 1.121;
/// The valley's insolation exponent (Lopez & Rice 2018 photoevaporation scaling).
pub const RADIUS_VALLEY_INSOLATION_EXP: f64 = 0.11;

/// The measured boundary between the stripped (super-Earth) and envelope-bearing
/// (sub-Neptune) populations as a function of insolation, Earth radii.
#[must_use]
pub fn radius_valley_rearth(insolation_rel: f64, r1_rearth: f64, exponent: f64) -> f64 {
    r1_rearth * insolation_rel.powf(exponent)
}

// ---- Temperature: ONE law, two consumers, two depths ---------------------------------

/// Equilibrium temperature (K) of a body at distance `d_m` from a luminosity `l_w` with
/// Bond albedo `bond_albedo`: `(L*(1-A) / (16*pi*sigma*d^2))^(1/4)`. Consumers: the
/// taxonomy row's `t_eq_k` and (inverted) the Star realm's extent.
#[must_use]
pub fn equilibrium_temperature_k(l_w: f64, d_m: f64, bond_albedo: f64) -> f64 {
    (l_w * (1.0 - bond_albedo)
        / (16.0 * core::f64::consts::PI * STEFAN_BOLTZMANN_W_M2_K4 * d_m * d_m))
        .powf(0.25)
}

/// The exact inverse of [`equilibrium_temperature_k`]: the distance (m) at which a body
/// reaches `temp_k`. At `A = 0`, `T = DUST_SUBLIMATION_K` this IS the Star realm's extent.
#[must_use]
pub fn flux_radius_m(l_w: f64, temp_k: f64, bond_albedo: f64) -> f64 {
    (l_w * (1.0 - bond_albedo)
        / (16.0 * core::f64::consts::PI * STEFAN_BOLTZMANN_W_M2_K4 * temp_k.powi(4)))
    .sqrt()
}

/// Kopparapu et al. 2013 (ApJ 765:131) CONSERVATIVE habitable-zone flux limits, Earth units
/// — the owner's ruling B (2026-08-19): the Earth-like predicate's temperate band is
/// `[0.53, 1.10] S⊕` (the owner's earlier 0.75–1.30 band is superseded — it could never
/// match the quantized ladder's rung 2 at 0.748 S⊕, at any seed).
pub const KOPPARAPU_FLUX_CONSERVATIVE: (f64, f64) = (0.53, 1.10);

/// Silicate dust sublimates at ~1500 K — the inner rim of every observed protoplanetary
/// disc (Dullemond & Monnier 2010, ARA&A 48:205; the same 1400-1600 K figure in Isella &
/// Natta 2005). [`flux_radius_m`] at this temperature and `A = 0` is the dust destruction
/// radius: the honest "solid matter is destroyed by heat here" surface.
pub const DUST_SUBLIMATION_K: f64 = 1500.0;

// ---- The envelope (Lopez & Fortney 2014) ---------------------------------------------

/// Lopez & Fortney 2014 (ApJ 792:1) envelope-thickness fit `(coeff, mass_exp, fenv_exp,
/// insolation_exp, age_exp)` about the `f_env = 0.05`, `age = 5 Gyr` normalisation point:
/// `R_env = 2.06 * M^-0.21 * (f_env/0.05)^0.59 * S^0.044 * (t/5)^-0.18` Earth radii.
pub const ENVELOPE_LF14: (f64, f64, f64, f64, f64) = (2.06, -0.21, 0.59, 0.044, -0.18);
/// The envelope fit's `f_env` normalisation point (Lopez & Fortney 2014).
pub const ENVELOPE_F_ENV_NORM: f64 = 0.05;
/// The system age fed to the envelope law — the median age of the Kepler FGK sample
/// (Silva Aguirre et al. 2015, MNRAS 452:2127). A stated constant, not a draw.
pub const SYSTEM_AGE_GYR: f64 = 5.0;
/// The drawn H/He envelope mass-fraction bounds (log-uniform): Lopez & Fortney 2014 +
/// Wolfgang & Lopez 2015 (ApJ 806:183).
pub const F_ENV_BOUNDS: (f64, f64) = (1.0e-3, 0.30);

/// Envelope thickness in Earth radii (Lopez & Fortney 2014). `R_planet = R_core + R_env`
/// for retained classes; `R_planet = R_core` for stripped ones.
#[must_use]
pub fn envelope_radius_rearth(
    mass_mearth: f64,
    f_env: f64,
    insolation_rel: f64,
    age_gyr: f64,
) -> f64 {
    let (coeff, m_exp, f_exp, s_exp, t_exp) = ENVELOPE_LF14;
    coeff
        * mass_mearth.powf(m_exp)
        * (f_env / ENVELOPE_F_ENV_NORM).powf(f_exp)
        * insolation_rel.powf(s_exp)
        * (age_gyr / SYSTEM_AGE_GYR).powf(t_exp)
}

// ---- Atmosphere: gravity, escape, scale height ---------------------------------------

/// Mean molecular weight of a solar-composition H2/He envelope (atomic units).
pub const MU_H_HE: f64 = 2.30;
/// Mean molecular weight of an N2 secondary atmosphere (atomic units) — the Jeans gauge's
/// reference species (Catling & Kasting 2017 ch. 5).
pub const MU_N2: f64 = 28.0;

/// Surface gravity `G*M/R^2` in m/s^2.
#[must_use]
pub fn surface_gravity_mps2(mass_kg: f64, radius_m: f64) -> f64 {
    crate::celestial::G * mass_kg / (radius_m * radius_m)
}

/// Escape velocity `sqrt(2*G*M/R)` in m/s.
#[must_use]
pub fn escape_velocity_mps(mass_kg: f64, radius_m: f64) -> f64 {
    (2.0 * crate::celestial::G * mass_kg / radius_m).sqrt()
}

/// Isothermal scale height `k_B*T / (mu*m_u*g)` in metres.
#[must_use]
pub fn scale_height_m(t_eq_k: f64, mu: f64, gravity_mps2: f64) -> f64 {
    BOLTZMANN_J_K * t_eq_k / (mu * ATOMIC_MASS_KG * gravity_mps2)
}

/// Photospheric reference density of a retained envelope: the thin-shell identity
/// `rho_0 = M_env / (4*pi*R^2*H)` (no free parameter).
#[must_use]
pub fn envelope_reference_density_kgm3(
    envelope_mass_kg: f64,
    radius_m: f64,
    scale_height_m: f64,
) -> f64 {
    envelope_mass_kg / (4.0 * core::f64::consts::PI * radius_m * radius_m * scale_height_m)
}

// ---- The cosmic shoreline (stripped classes' atmosphere verdict) ---------------------

/// Earth's escape velocity in m/s (the shoreline's normalising body).
pub const V_ESC_EARTH_MPS: f64 = 11_180.0;

/// The shoreline coefficient: `I_XUV < SHORELINE_COEFF * (v_esc/v_esc_earth)^4` retains
/// (Zahnle & Catling 2017, ApJ 843:122 — the LAW is derived, the COEFFICIENT is not).
/// Calibrated as the geometric mean of the binding bracket: Titan (retains, 3.540) and
/// Ganymede (airless, 10.246) — `sqrt(3.540 * 10.246) = 6.0229`. The calibration set is
/// NOT linearly separable (Mars 10.537 retains vs Ganymede 10.246 airless straddle within
/// 2.9%); exactly one body must be a miss, and it is Mars — asserted by name in the T0
/// shoreline golden, which also recomputes this coefficient from the 8-body table.
pub const SHORELINE_COEFF: f64 = 6.0229;

/// Does a stripped body retain a secondary atmosphere? `xuv_rel` is the XUV irradiation
/// relative to Earth's ([`xuv_rel_of`] — NOT bolometric insolation).
#[must_use]
pub fn cosmic_shoreline_retains(xuv_rel: f64, v_esc_mps: f64) -> bool {
    xuv_rel < SHORELINE_COEFF * (v_esc_mps / V_ESC_EARTH_MPS).powi(4)
}

/// The XUV irradiation relative to Earth's: bolometric insolation scaled by the star
/// class's time-integrated XUV enhancement over the Sun's ([`SpectralDef::xuv_bol_ratio`];
/// G is the normalising class).
#[must_use]
pub fn xuv_rel_of(insolation_rel: f64, class: SpectralClass) -> f64 {
    insolation_rel * class.def().xuv_bol_ratio / XUV_BOL_G
}

/// The Jeans safety factor: an atmosphere survives Jeans escape over gigayears only when
/// `v_esc >= 6 * v_thermal` (Catling & Kasting 2017 ch. 5).
pub const JEANS_SAFETY_FACTOR: f64 = 6.0;

/// The recorded GAUGE beside the shoreline (never a fence): thermal Jeans retention for an
/// N2 atmosphere. Scores 6 of 8 on the calibration set (misses Mercury and Ganymede); the
/// T0 golden prints both verdicts per body so a future correction has both numbers.
#[must_use]
pub fn jeans_retains(v_esc_mps: f64, t_eq_k: f64) -> bool {
    let v_thermal = (3.0 * BOLTZMANN_J_K * t_eq_k / (MU_N2 * ATOMIC_MASS_KG)).sqrt();
    v_esc_mps >= JEANS_SAFETY_FACTOR * v_thermal
}

// ---- Moon count laws (taxonomy arc par 4.2) ------------------------------------------

/// Roche 1849 fluid disruption coefficient.
pub const ROCHE_COEFF: f64 = 2.44;
/// Machida, Kokubo, Inutsuka & Matsumoto 2008 (ApJ 685:1220): the centrifugal radius of
/// the circumplanetary gas disc, `r_c ~ R_Hill/48` — the regular-satellite outer edge.
/// A derived value (empirical cross-check: Callisto 0.0354, Titan 0.0187, Oberon 0.0083
/// of R_Hill); the T0 census golden publishes the per-planet residual against it.
pub const SATELLITE_DISC_HILL_FRACTION: f64 = 1.0 / 48.0;
/// Ice-satellite bulk density (kg/m^3): Ganymede 1936, Callisto 1834, Titan 1881.
pub const RHO_ICE_KGM3: f64 = 1900.0;
/// Rock-satellite bulk density (kg/m^3): the Moon, 3344.
pub const RHO_ROCK_KGM3: f64 = 3344.0;
/// Canup & Ward 2006 (Nature 441:834) gas-starved satellite-system mass fraction; the
/// solar giants realise 1.21-2.48e-4 around it.
pub const SATELLITE_MASS_FRACTION: f64 = 1.0e-4;
/// Lineweaver & Norman 2010 "the potato radius": Mimas (198 km) is the smallest body in
/// hydrostatic equilibrium; below this no moon realm is emitted (rubble, a later slice).
pub const MOON_MIN_RADIUS_M: f64 = 2.0e5;
/// Regular-satellite eccentricity Rayleigh sigma — measured mean of Io 0.0041, Europa
/// 0.0094, Ganymede 0.0013, Callisto 0.0074, Titan 0.0288 (tidally damped).
pub const MOON_ECC_SIGMA: f64 = 0.007;
/// Regular-satellite inclination Rayleigh sigma (rad) — Galilean mean 0.22 deg.
pub const MOON_INCL_SIGMA: f64 = 0.003;

/// The fluid Roche disruption radius in metres, mass form (the planet radius CANCELS out
/// of the density-corrected `2.44*R_p*(rho_p/rho_m)^(1/3)` — taxonomy design par 4.2.2):
/// `2.44 * (3*M_p / (4*pi*rho_moon))^(1/3)`.
#[must_use]
pub fn roche_radius_m(planet_mass_kg: f64, moon_density_kgm3: f64) -> f64 {
    ROCHE_COEFF
        * (3.0 * planet_mass_kg / (4.0 * core::f64::consts::PI * moon_density_kgm3)).powf(1.0 / 3.0)
}

/// Hill 1878 sphere radius `a * (m_p/(3*M_star))^(1/3)` in metres.
#[must_use]
pub fn hill_radius_m(sma_m: f64, planet_mass_kg: f64, star_mass_kg: f64) -> f64 {
    sma_m * (planet_mass_kg / (3.0 * star_mass_kg)).powf(1.0 / 3.0)
}

/// The circumplanetary disc outer edge `fraction * R_Hill` in metres (Machida 2008).
#[must_use]
pub fn satellite_disc_edge_m(hill_radius_m: f64, fraction: f64) -> f64 {
    hill_radius_m * fraction
}

/// The moon orbit ladder `d_Roche * ratio^(k+1)` (the `+1` offset: a moon may not orbit AT
/// the disruption radius). `ratio` reuses the planet ladder's `ORBITAL_RATIO` (Galilean
/// spacings 1.59/1.60/1.75 validate it at this scale).
#[must_use]
pub fn satellite_ladder_a_m(k: u32, roche_m: f64, ratio: f64) -> f64 {
    roche_m * ratio.powf(f64::from(k) + 1.0)
}

// ===== THE ONE BODY-DERIVATION PATH (taxonomy arc T1; celestial_taxonomy_design par 4.4) =====

/// A derived atmosphere. `reference_density_kgm3` is `Some` only for RETAINED-envelope classes
/// (the thin-shell identity `f_env·M / (4·pi·R²·H)`); a stripped body that keeps a SECONDARY
/// atmosphere has no derivable surface density until the `p_surf` draw lands (D-TAX-1 — surface
/// pressure has no derivation: Earth 1 bar, Venus 92 bar, Mars 6 mbar at comparable escape
/// speeds), and `None` states that honestly rather than minting a dressed-up literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    /// (P) scale height, drag — `MU_H_HE` for a kept envelope, `MU_N2` for a secondary.
    pub mean_molecular_weight: f64,
    /// (P) the drag profile `rho(h) = rho_0·exp(-h/H)`.
    pub scale_height_m: f64,
    /// (P) drag magnitude — retained classes only in this arc (see the struct doc).
    pub reference_density_kgm3: Option<f64>,
}

/// The emitted taxonomy row — every field with a NAMED consumer (the par 3.2 filter):
/// (P) the physics charter, (R) rendering, (S) the seed search. Surface gravity and escape
/// velocity are NOT stored — both are one line from `mass_kg` and `radius_m`, and a stored
/// derivation is a second representation of one fact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyTaxon {
    /// (R) colour family later; (S) the rocky predicate; (T3) the moon budget.
    pub class: PlanetType,
    /// (P) gravity; `central_mass` for its own children.
    pub mass_kg: f64,
    /// (R) the self-look; (P) surface gravity, Roche.
    pub radius_m: f64,
    /// (S) the habitable predicate; (P) thermal load.
    pub insolation_rel: f64,
    /// (P) material limits; (S) the temperate band.
    pub t_eq_k: f64,
    /// Derived from `(class, has_atmosphere)`; feeds `t_eq_k` (the evaluation-order contract).
    pub bond_albedo: f64,
    pub atmosphere: Option<Atmosphere>,
}

/// THE ONE DERIVATION PATH (par 4.4, the HR4 shape): called with
/// `(illuminator_distance = its own orbit)` for a planet and with
/// `(illuminator_distance = its PARENT's orbit)` for a moon — identical code, identical arms,
/// zero branches on what the body is. The only type-keyed inputs are the Bond-albedo table and
/// the composition selector, both registry DATA off [`PlanetDef`], never a realm kind.
///
/// THE EVALUATION ORDER IS PART OF THE CONTRACT (par 3.5.1, no circularity):
/// `class → shoreline verdict → atmosphere bit → Bond albedo → T_eq` — the classifier consumes
/// INSOLATION, never temperature, so no albedo exists before a class does.
#[must_use]
pub fn body_params(
    mass_kg: f64,
    illuminator_l_lsun: f64,
    illuminator_class: SpectralClass,
    illuminator_distance_m: f64,
    frost_line_au: f64,
    th: FrostThresholds,
    f_env: f64,
) -> BodyTaxon {
    let mass_mearth = mass_kg / M_EARTH_KG;
    let orbit_au = illuminator_distance_m / AU_M;
    let insolation_rel = illuminator_l_lsun / (orbit_au * orbit_au);
    let class = classify_planet(orbit_au, frost_line_au, mass_mearth, insolation_rel, th);
    let radius_m = composition_radius_m(
        class,
        orbit_au < frost_line_au,
        mass_mearth,
        f_env,
        insolation_rel,
    );
    // The shoreline verdict (stripped classes' secondary-atmosphere bit) — evaluated for every
    // class (branchless input side); the retained bit wins by data (`retains_h2_envelope`).
    let v_esc_mps = escape_velocity_mps(mass_kg, radius_m);
    let xuv_rel = xuv_rel_of(insolation_rel, illuminator_class);
    let keeps_secondary = cosmic_shoreline_retains(xuv_rel, v_esc_mps);
    let keeps_envelope = class.def().retains_h2_envelope;
    let has_atmosphere = keeps_envelope | keeps_secondary;
    let bond_albedo = bond_albedo_of(class, has_atmosphere);
    let t_eq_k = equilibrium_temperature_k(
        illuminator_l_lsun * L_SUN_W,
        illuminator_distance_m,
        bond_albedo,
    );
    let atmosphere = derive_atmosphere(
        keeps_envelope,
        keeps_secondary,
        mass_kg,
        radius_m,
        t_eq_k,
        f_env,
    );
    BodyTaxon {
        class,
        mass_kg,
        radius_m,
        insolation_rel,
        t_eq_k,
        bond_albedo,
        atmosphere,
    }
}

/// The composition radius (par 3.4.3/3.4.4): rock core inside the frost line, ice core beyond
/// (Zeng), plus the Lopez–Fortney envelope for retained non-giants; a GAS GIANT takes the
/// Chen–Kipping Jovian branch (the only law covering that regime — the envelope fit does not).
///
/// THE SMALL-BODY CLAMP (`.min`, branchless): a body's radius never exceeds its own
/// UNCOMPRESSED density sphere `(3M/4πρ)^(1/3)` at its composition density — below the Zeng
/// fits' domain a potato-class body is incompressible and the `M^0.27` extrapolation
/// over-sizes it (the two laws cross near 0.07 M⊕; Mimas-class moons are pure density
/// spheres — Lineweaver & Norman 2010's own regime). Monomorphic (HR5: the routing reads
/// registry DATA, `is_giant`/`retains_h2_envelope`).
#[must_use]
pub fn composition_radius_m(
    class: PlanetType,
    inside_frost: bool,
    mass_mearth: f64,
    f_env: f64,
    insolation_rel: f64,
) -> f64 {
    if class.def().is_giant & !class.def().requires_frost_line {
        return R_EARTH_M * segmented_power_law(mass_mearth, &PMR_SEGMENTS);
    }
    let (core_segments, rho): (&[(f64, f64, f64)], f64) = if inside_frost {
        (&ROCK_MR_SEGMENTS, RHO_ROCK_KGM3)
    } else {
        (&ICE_MR_SEGMENTS, RHO_ICE_KGM3)
    };
    let core_m = (R_EARTH_M * segmented_power_law(mass_mearth, core_segments))
        .min(density_sphere_radius_m(mass_mearth * M_EARTH_KG, rho));
    let envelope_rearth = if class.def().retains_h2_envelope {
        envelope_radius_rearth(mass_mearth, f_env, insolation_rel, SYSTEM_AGE_GYR)
    } else {
        0.0
    };
    core_m + R_EARTH_M * envelope_rearth
}

/// The uncompressed constant-density sphere `(3M/(4πρ))^(1/3)` — the small-body radius law
/// and the composition radius's incompressible bound.
#[must_use]
pub fn density_sphere_radius_m(mass_kg: f64, density_kgm3: f64) -> f64 {
    (3.0 * mass_kg / (4.0 * core::f64::consts::PI * density_kgm3)).powf(1.0 / 3.0)
}

/// The atmosphere row (monomorphic; all three arms driven): a kept H/He envelope (`MU_H_HE`,
/// with the thin-shell reference density), a shoreline-surviving secondary (`MU_N2`, density
/// deferred to D-TAX-1), or airless.
fn derive_atmosphere(
    keeps_envelope: bool,
    keeps_secondary: bool,
    mass_kg: f64,
    radius_m: f64,
    t_eq_k: f64,
    f_env: f64,
) -> Option<Atmosphere> {
    let gravity = surface_gravity_mps2(mass_kg, radius_m);
    if keeps_envelope {
        let scale_height = scale_height_m(t_eq_k, MU_H_HE, gravity);
        Some(Atmosphere {
            mean_molecular_weight: MU_H_HE,
            scale_height_m: scale_height,
            reference_density_kgm3: Some(envelope_reference_density_kgm3(
                f_env * mass_kg,
                radius_m,
                scale_height,
            )),
        })
    } else if keeps_secondary {
        Some(Atmosphere {
            mean_molecular_weight: MU_N2,
            scale_height_m: scale_height_m(t_eq_k, MU_N2, gravity),
            reference_density_kgm3: None,
        })
    } else {
        None
    }
}

// ===== Inverse-CDF samplers (closed form, no rejection — bit-reproducible) ============

/// Slopes within this of 1 use the log-uniform limit (avoids a `1/(1-slope)` singularity).
const IMF_SLOPE_EPS: f64 = 1e-9;

/// Bounded power-law (Salpeter/single-segment IMF, Salpeter 1955) inverse-CDF stellar mass:
/// `u01 in [0,1)` -> mass in `[m_lo, m_hi]`, monotone non-decreasing. `slope` is the IMF
/// exponent alpha (Salpeter 2.35). Params-as-args; Slice 3 supplies them.
#[must_use]
pub fn sample_imf_mass(u01: f64, slope: f64, m_lo: f64, m_hi: f64) -> f64 {
    if (slope - 1.0).abs() > IMF_SLOPE_EPS {
        imf_power(u01, slope, m_lo, m_hi)
    } else {
        imf_log(u01, m_lo, m_hi)
    }
}

/// The `slope != 1` inverse-CDF: invert `CDF(M) = (M^b - m_lo^b)/(m_hi^b - m_lo^b)`, `b = 1-slope`.
fn imf_power(u01: f64, slope: f64, m_lo: f64, m_hi: f64) -> f64 {
    let b = 1.0 - slope;
    let lo = m_lo.powf(b);
    let hi = m_hi.powf(b);
    (lo + u01 * (hi - lo)).powf(1.0 / b)
}

/// The `slope -> 1` log-uniform analytic limit of the bounded power law.
fn imf_log(u01: f64, m_lo: f64, m_hi: f64) -> f64 {
    m_lo * (m_hi / m_lo).powf(u01)
}

/// Rayleigh inverse-CDF `sigma * sqrt(-2*ln(1-u01))` — the shared eccentricity/inclination
/// sampler (Fabrycky 2014). Using `1-u01` keeps `u=0 -> 0` and, since `u01 in [0,1)` gives
/// `1-u01 in (0,1]`, `ln` is finite (the half-open RNG never yields 1, enforced at the RNG
/// seam). NOTE: `u=0` yields `-0.0` (which compares equal to `0.0`) — do NOT add
/// `.max()/.abs()` to normalize the sign; it would inject an uncoverable branch and defeat
/// the branchless closed form.
#[must_use]
pub fn sample_rayleigh(u01: f64, sigma: f64) -> f64 {
    sigma * (-2.0 * (1.0 - u01).ln()).sqrt()
}

/// Titius-Bode geometric orbital spacing `a0 * ratio^n` in AU (Chambers 1996). NOT a `u01`
/// sampler — orbital spacing is deterministic-per-slot (`n` = the planet index); Slice 3
/// draws `a0`/`ratio` from the realm stream ONCE per system, then spaces planets by index.
#[must_use]
pub fn orbital_axis_au(n: u32, a0_au: f64, ratio: f64) -> f64 {
    a0_au * ratio.powf(f64::from(n))
}

/// Categorical inverse-CDF cumulative-threshold scan (half-open `[c_i, c_{i+1})`). `cumulative`
/// = `[c_spiral, c_spiral+c_elliptical]` (`GalaxyType::CANONICAL_CUMULATIVE` supplies the
/// census defaults). Branch-honest: a partition count, not an `&&` chain.
#[must_use]
pub fn sample_galaxy_type(u01: f64, cumulative: &[f64; 2]) -> GalaxyType {
    let idx = cumulative.iter().filter(|&&c| u01 >= c).count();
    GalaxyType::ALL[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- drift tripwires (the entity_kind.rs KindDef idiom, per registry) -------------

    #[test]
    fn galaxy_type_registry_cannot_drift() {
        for x in GalaxyType::ALL {
            match x {
                GalaxyType::Spiral | GalaxyType::Elliptical | GalaxyType::Irregular => {}
            }
        }
        let mut accepted = 0usize;
        for tag in 0..=u8::MAX {
            if let Ok(x) = GalaxyType::from_tag(tag) {
                accepted += 1;
                assert_eq!(x as u8, tag);
                assert!(GalaxyType::ALL.contains(&x));
                assert_eq!(x.def().tag, x as u8, "{x:?} def tag pins the discriminant");
            }
        }
        assert_eq!(accepted, GalaxyType::ALL.len());
        assert_eq!(
            GalaxyType::from_tag(200).expect_err("unknown").to_string(),
            "unknown taxonomy tag galaxy:200"
        );
    }

    #[test]
    fn spectral_class_registry_cannot_drift() {
        for x in SpectralClass::ALL {
            match x {
                SpectralClass::O
                | SpectralClass::B
                | SpectralClass::A
                | SpectralClass::F
                | SpectralClass::G
                | SpectralClass::K
                | SpectralClass::M => {}
            }
        }
        let mut accepted = 0usize;
        for tag in 0..=u8::MAX {
            if let Ok(x) = SpectralClass::from_tag(tag) {
                accepted += 1;
                assert_eq!(x as u8, tag);
                assert!(SpectralClass::ALL.contains(&x));
                assert_eq!(x.def().tag, x as u8);
            }
        }
        assert_eq!(accepted, SpectralClass::ALL.len());
        assert_eq!(
            SpectralClass::from_tag(200)
                .expect_err("unknown")
                .to_string(),
            "unknown taxonomy tag spectral:200"
        );
    }

    #[test]
    fn planet_type_registry_cannot_drift() {
        for x in PlanetType::ALL {
            match x {
                PlanetType::Rocky
                | PlanetType::Ocean
                | PlanetType::GasGiant
                | PlanetType::IceGiant
                | PlanetType::SubNeptune => {}
            }
        }
        let mut accepted = 0usize;
        for tag in 0..=u8::MAX {
            if let Ok(x) = PlanetType::from_tag(tag) {
                accepted += 1;
                assert_eq!(x as u8, tag);
                assert!(PlanetType::ALL.contains(&x));
                assert_eq!(x.def().tag, x as u8);
            }
        }
        assert_eq!(accepted, PlanetType::ALL.len());
        assert_eq!(
            PlanetType::from_tag(200).expect_err("unknown").to_string(),
            "unknown taxonomy tag planet:200"
        );
    }

    // ---- coherence ------------------------------------------------------------------

    #[test]
    fn every_taxon_def_is_coherent() {
        let bad_spectral: Vec<SpectralClass> = SpectralClass::ALL
            .into_iter()
            .filter(|x| !x.def().is_coherent())
            .collect();
        assert_eq!(bad_spectral, Vec::<SpectralClass>::new());
        let bad_planet: Vec<PlanetType> = PlanetType::ALL
            .into_iter()
            .filter(|x| !x.def().is_coherent())
            .collect();
        assert_eq!(bad_planet, Vec::<PlanetType>::new());
    }

    #[test]
    fn taxa_incoherent_defs_are_detected() {
        // Clause 1 false: non-positive lower mass bound.
        let bad_mass = SpectralDef {
            mass_lo_msun: 0.0,
            ..G_DEF
        };
        assert!(!bad_mass.is_coherent());
        // Clause 2 false: inverted T_eff band (lo >= hi).
        let bad_band = SpectralDef {
            teff_lo_k: 11_000.0,
            ..A_DEF
        }; // A_DEF hi = 10_000
        assert!(!bad_band.is_coherent());
        // Clause 3 false: non-positive band floor (lo <= 0, but still lo < hi).
        let bad_floor = SpectralDef {
            teff_lo_k: 0.0,
            ..A_DEF
        };
        assert!(!bad_floor.is_coherent());
        // Clause 4 false: non-positive XUV/bolometric ratio.
        let bad_xuv = SpectralDef {
            xuv_bol_ratio: 0.0,
            ..G_DEF
        };
        assert!(!bad_xuv.is_coherent());
        // PlanetDef clause 1 false: a giant that lost its envelope (runaway accretion IS
        // envelope retention).
        let bad_giant = PlanetDef {
            is_giant: true,
            retains_h2_envelope: false,
            ..GAS_GIANT_DEF
        };
        assert!(!bad_giant.is_coherent());
        // PlanetDef clause 2 false: an unphysical with-atmosphere Bond albedo.
        let bad_air_albedo = PlanetDef {
            bond_albedo_with_atmosphere: 1.0,
            ..ROCKY_DEF
        };
        assert!(!bad_air_albedo.is_coherent());
        // PlanetDef clause 3 false: an unphysical airless Bond albedo.
        let bad_bare_albedo = PlanetDef {
            bond_albedo_airless: -0.1,
            ..ROCKY_DEF
        };
        assert!(!bad_bare_albedo.is_coherent());
    }

    // ---- classifiers ----------------------------------------------------------------

    #[test]
    fn classify_spectral_golden_boundaries() {
        let b = &SpectralClass::MASS_BOUNDS;
        assert_eq!(classify_spectral(20.0, b), SpectralClass::O);
        assert_eq!(classify_spectral(5.0, b), SpectralClass::B);
        assert_eq!(classify_spectral(1.7, b), SpectralClass::A);
        assert_eq!(classify_spectral(1.2, b), SpectralClass::F);
        assert_eq!(classify_spectral(1.0, b), SpectralClass::G); // the Sun
        assert_eq!(classify_spectral(0.6, b), SpectralClass::K);
        assert_eq!(classify_spectral(0.3, b), SpectralClass::M);
        // Sub-hydrogen-burning tail: clamps to M via the unwrap_or None arm.
        assert_eq!(classify_spectral(0.05, b), SpectralClass::M);
    }

    #[test]
    fn mass_luminosity_and_hz_golden() {
        let s = &SpectralClass::MLR_SEGMENTS;
        assert!((main_sequence_luminosity(1.0, s) - 1.0).abs() < 1e-9);
        assert!((main_sequence_luminosity(0.3, s) - 0.014_424_697_75).abs() < 1e-4);
        // Mid-segment 2: distinguishes exponent 4.0 (1.5^4 = 5.0625) from a wrong 3.5.
        assert!((main_sequence_luminosity(1.5, s) - 5.0625).abs() < 1e-6);
        // Segment 2 upper edge: 1.0 * 2^4 = 16.0 (would be 11.31 under a transposed table).
        assert!((main_sequence_luminosity(2.0, s) - 16.0).abs() < 1e-6);
        // High tail (M > 55) exercises the unwrap_or None arm: 1.4 * 60^3.5.
        let tail = main_sequence_luminosity(60.0, s);
        assert!((tail - 1.4 * 60.0_f64.powf(3.5)).abs() < 1e-3);
        // HZ centre + inner/outer edges.
        assert!((habitable_zone_radius_au(1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((frost_line_radius_au(1.0, FROST_COEFF_AU) - 2.7).abs() < 1e-12);
    }

    #[test]
    fn mass_luminosity_is_continuous_at_the_segment_breaks() {
        // A transposed (coeff<->exponent) segment table jumps at the breaks; guard it.
        let seg = |i: usize, m: f64| {
            let (_, c, p) = SpectralClass::MLR_SEGMENTS[i];
            c * m.powf(p)
        };
        assert!(
            (seg(0, 0.43) / seg(1, 0.43) - 1.0).abs() < 0.1,
            "break at 0.43"
        );
        assert!(
            (seg(1, 2.0) / seg(2, 2.0) - 1.0).abs() < 0.1,
            "break at 2.0"
        );
    }

    // ---- THE SOLAR-SYSTEM CLASSIFIER GOLDEN (taxonomy design par 3.4.2) -----------
    //
    // Eight named bodies, literal masses and insolations; the verdicts are pinned with the
    // single Saturn miss NAMED (Saturn is 0.299 M_J; the published Chen-Kipping break puts
    // it on the Neptunian branch), so a wrong law cannot pass by being vague.

    /// (name, mass Mearth, sma AU, truth) — masses/orbits from the IAU planetary fact
    /// sheets; insolation = 1/a^2; frost line at L=1 is 2.7 AU.
    const SOLAR_BODIES: [(&str, f64, f64, PlanetType); 8] = [
        ("Mercury", 0.0553, 0.3871, PlanetType::Rocky),
        ("Venus", 0.8150, 0.7233, PlanetType::Rocky),
        ("Earth", 1.0, 1.0, PlanetType::Rocky),
        ("Mars", 0.1070, 1.5237, PlanetType::Rocky),
        ("Jupiter", 317.8, 5.2044, PlanetType::GasGiant),
        ("Saturn", 95.16, 9.5826, PlanetType::GasGiant),
        ("Uranus", 14.54, 19.191, PlanetType::IceGiant),
        ("Neptune", 17.15, 30.07, PlanetType::IceGiant),
    ];

    #[test]
    fn the_solar_system_classifier_golden_is_seven_of_eight_with_saturn_the_named_miss() {
        let th = FrostThresholds::CANONICAL;
        let frost = frost_line_radius_au(1.0, FROST_COEFF_AU);
        let verdicts: Vec<PlanetType> = SOLAR_BODIES
            .iter()
            .map(|(_, m, a, _)| classify_planet(*a, frost, *m, 1.0 / (a * a), th))
            .collect();
        let expected = [
            PlanetType::Rocky,    // Mercury
            PlanetType::Rocky,    // Venus
            PlanetType::Rocky,    // Earth
            PlanetType::Rocky,    // Mars
            PlanetType::GasGiant, // Jupiter
            PlanetType::IceGiant, // Saturn — THE NAMED MISS (truth: GasGiant)
            PlanetType::IceGiant, // Uranus
            PlanetType::IceGiant, // Neptune
        ];
        assert_eq!(verdicts, expected);
        // Exactly one miss, and it is Saturn.
        let misses: Vec<&str> = SOLAR_BODIES
            .iter()
            .zip(&verdicts)
            .filter(|((_, _, _, truth), v)| truth != *v)
            .map(|((name, _, _, _), _)| *name)
            .collect();
        assert_eq!(misses, vec!["Saturn"]);
    }

    #[test]
    fn classify_planet_drives_every_arm_including_the_synthetic_threshold_split() {
        let th = FrostThresholds::CANONICAL;
        // The 2x2, one arm each (frost at 2.7 AU, temperate-ish insolations):
        // inside + stripped -> Rocky (Earth analog).
        assert_eq!(classify_planet(1.0, 2.7, 1.0, 1.0, th), PlanetType::Rocky);
        // inside + kept -> SubNeptune (a 10 Mearth core at Earth insolation:
        // R_core = 10^0.27 = 1.862 > valley 1.121).
        assert_eq!(
            classify_planet(1.0, 2.7, 9.9, 1.0, th),
            PlanetType::SubNeptune
        );
        // beyond + stripped -> Ocean (a Mars-mass body beyond the line).
        assert_eq!(
            classify_planet(5.0, 2.7, 0.107, 0.037, th),
            PlanetType::Ocean
        );
        // beyond + kept -> IceGiant (Neptune).
        assert_eq!(
            classify_planet(30.07, 2.7, 17.15, 1.0 / (30.07 * 30.07), th),
            PlanetType::IceGiant
        );
        // The giant escape fires on BOTH frost sides (a migrated hot Jupiter).
        assert_eq!(
            classify_planet(0.05, 2.7, 317.8, 400.0, th),
            PlanetType::GasGiant
        );
        assert_eq!(
            classify_planet(5.2, 2.7, 317.8, 0.037, th),
            PlanetType::GasGiant
        );
        // The two named giant-escape clauses split (HR5: drive the runaway_core=false &
        // jovian_mass=true arm with a synthetic threshold table — canonically unreachable
        // because 131.6 > 10, honest here because thresholds are params-as-args).
        let synth = FrostThresholds {
            m_gas_mearth: 5.0,
            m_core_crit_mearth: 8.0,
            ..FrostThresholds::CANONICAL
        };
        // jovian by the synthetic break, but below the runaway core: NOT a giant.
        assert_eq!(
            classify_planet(1.0, 2.7, 6.0, 1.0, synth),
            PlanetType::SubNeptune
        );
        // above both: giant.
        assert_eq!(
            classify_planet(1.0, 2.7, 9.0, 1.0, synth),
            PlanetType::GasGiant
        );
        // Boundary: orbit == frost line -> beyond (the strict `<`).
        assert_eq!(
            classify_planet(2.7, 2.7, 0.107, 0.137, th),
            PlanetType::Ocean
        );
    }

    #[test]
    fn bond_albedo_reads_the_class_table_and_only_rocky_splits_on_air() {
        assert_eq!(bond_albedo_of(PlanetType::Rocky, true), 0.30);
        assert_eq!(bond_albedo_of(PlanetType::Rocky, false), 0.10);
        for class in [
            PlanetType::Ocean,
            PlanetType::GasGiant,
            PlanetType::IceGiant,
            PlanetType::SubNeptune,
        ] {
            assert_eq!(bond_albedo_of(class, true), bond_albedo_of(class, false));
        }
        assert_eq!(bond_albedo_of(PlanetType::GasGiant, true), 0.34);
        assert_eq!(bond_albedo_of(PlanetType::IceGiant, true), 0.29);
    }

    // ---- THE SHORELINE CALIBRATION TABLE (taxonomy design par 3.6.2) ------------------

    /// (name, insolation S_earth, v_esc m/s, retains_truth). Insolations from the solar
    /// distances; escape velocities from the fact sheets.
    const SHORELINE_BODIES: [(&str, f64, f64, bool); 8] = [
        ("Earth", 1.0, 11_180.0, true),
        ("Venus", 1.911_45, 10_360.0, true),
        ("Titan", 0.010_99, 2_639.0, true),
        ("Ganymede", 0.037_02, 2_741.0, false),
        ("Mars", 0.430_73, 5_027.0, true),
        ("Io", 0.037_02, 2_558.0, false),
        ("Mercury", 6.673_50, 4_250.0, false),
        ("Moon", 1.0, 2_380.0, false),
    ];

    #[test]
    fn the_shoreline_calibration_is_seven_of_eight_with_mars_the_named_miss() {
        // The coefficient IS the geometric mean of the binding bracket, recomputed from the
        // table (a measurement pinning the constant, not an argument).
        let shoreline_of = |s: f64, v: f64| s / (v / V_ESC_EARTH_MPS).powi(4);
        let titan = shoreline_of(SHORELINE_BODIES[2].1, SHORELINE_BODIES[2].2);
        let ganymede = shoreline_of(SHORELINE_BODIES[3].1, SHORELINE_BODIES[3].2);
        let derived = (titan * ganymede).sqrt();
        assert!(
            (derived - SHORELINE_COEFF).abs() < 5e-4,
            "SHORELINE_COEFF {SHORELINE_COEFF} must equal sqrt(Titan*Ganymede) = {derived}"
        );
        // Titan is the binding retainer, Ganymede the binding airless: they bracket.
        assert!(titan < SHORELINE_COEFF);
        assert!(SHORELINE_COEFF < ganymede);
        // The set is NOT linearly separable: Mars (retains) sits ABOVE Ganymede (airless).
        let mars = shoreline_of(SHORELINE_BODIES[4].1, SHORELINE_BODIES[4].2);
        assert!(
            mars > ganymede,
            "Mars {mars} vs Ganymede {ganymede}: the 2.9% straddle"
        );
        // Verdicts: 7 of 8 with Mars the ONE named miss. The gate PRINTS both gauges.
        let mut misses = Vec::new();
        for (name, s, v, truth) in SHORELINE_BODIES {
            let shoreline = cosmic_shoreline_retains(s, v);
            let jeans = jeans_retains(v, equilibrium_temperature_k(L_SUN_W, AU_M / s.sqrt(), 0.0));
            println!("[shoreline] {name}: shoreline={shoreline} jeans={jeans} truth={truth}");
            if shoreline != truth {
                misses.push(name);
            }
        }
        assert_eq!(misses, vec!["Mars"]);
    }

    #[test]
    fn the_jeans_gauge_scores_five_of_eight_and_is_recorded_never_a_fence() {
        // MEASURED under the stated convention (T_eq at the body's solar distance, A = 0,
        // N2): the gauge misses Ganymede, Io AND Mercury — 5 of 8, one worse than the
        // design pass predicted (6 of 8; its Io cell did not survive recomputation:
        // 6*v_th(122 K) = 1978 m/s < Io's 2558 m/s, so Jeans wrongly retains Io). Pinned
        // as what MEASURES; the gauge is a recorded second opinion beside the shoreline,
        // never a fence, and the shoreline golden prints both verdicts per body.
        let mut misses = Vec::new();
        for (name, s, v, truth) in SHORELINE_BODIES {
            let d_m = AU_M / s.sqrt();
            let t = equilibrium_temperature_k(L_SUN_W, d_m, 0.0);
            if jeans_retains(v, t) != truth {
                misses.push(name);
            }
        }
        assert_eq!(misses, vec!["Ganymede", "Io", "Mercury"]);
    }

    #[test]
    fn xuv_rel_is_insolation_scaled_by_the_class_enhancement_over_g() {
        // G is the normalising class: xuv_rel == insolation.
        assert_eq!(xuv_rel_of(1.0, SpectralClass::G), 1.0);
        // M dwarfs: 100x the solar ratio (within f64 division rounding).
        assert!((xuv_rel_of(1.0, SpectralClass::M) - 100.0).abs() < 1e-9);
        assert!((xuv_rel_of(0.5, SpectralClass::M) - 50.0).abs() < 1e-9);
        // Radiative envelopes: a tenth.
        assert!((xuv_rel_of(1.0, SpectralClass::B) - 0.1).abs() < 1e-12);
    }

    // ---- THE SOLAR MOON CENSUS (taxonomy design par 4.2.1) ----------------------------

    #[test]
    fn the_solar_moon_census_pins_the_per_planet_residual() {
        // (name, planet mass kg, sma AU, real regular count). Densities: ice beyond the
        // 2.7 AU frost line, rock inside. The LAW is validated (the shipped count is
        // radius-free — par 4.2.2: the planet radius cancels).
        let rows: [(&str, f64, f64, i32); 8] = [
            ("Mercury", 3.301e23, 0.3871, 0),
            ("Venus", 4.867e24, 0.7233, 0),
            ("Earth", 5.972e24, 1.0, 1),
            ("Mars", 6.417e23, 1.5237, 2),
            ("Jupiter", 1.898e27, 5.2044, 4),
            ("Saturn", 5.683e26, 9.5826, 7),
            ("Uranus", 8.681e25, 19.191, 5),
            ("Neptune", 1.024e26, 30.07, 1),
        ];
        let frost = frost_line_radius_au(1.0, FROST_COEFF_AU);
        let mut predicted = Vec::new();
        for (name, m_kg, a_au, _) in rows {
            let rho = if a_au < frost {
                RHO_ROCK_KGM3
            } else {
                RHO_ICE_KGM3
            };
            let roche = roche_radius_m(m_kg, rho);
            let disc = satellite_disc_edge_m(
                hill_radius_m(a_au * AU_M, m_kg, M_SUN_KG),
                SATELLITE_DISC_HILL_FRACTION,
            );
            let n = (0..64)
                .take_while(|&k| satellite_ladder_a_m(k, roche, 1.7) <= disc)
                .count() as i32;
            println!("[census] {name}: predicted {n}");
            predicted.push(n);
        }
        // The pinned per-planet prediction (recomputed in the design pass; MEASURED here).
        assert_eq!(predicted, vec![0, 0, 0, 1, 3, 4, 6, 7]);
        // The residual is judged PER PLANET (SAE), never as a lucky total.
        let sae: i32 = rows
            .iter()
            .zip(&predicted)
            .map(|((_, _, _, real), p)| (p - real).abs())
            .sum();
        assert_eq!(sae, 13, "sum of absolute per-planet errors");
        // Both moonless planets are predicted moonless; Neptune (+6) is documented history
        // (Triton's capture destroyed the regular system — Agnor & Hamilton 2006).
        assert_eq!(predicted[0], 0);
        assert_eq!(predicted[1], 0);
    }

    // ---- Temperature: the one law, its inverse, and the frost cross-check -------------

    #[test]
    fn equilibrium_temperature_earth_goldens() {
        // Earth at A = 0.306 vs the accepted 254 K; at A = 0 the bare 278.32 K.
        let with_air = equilibrium_temperature_k(L_SUN_W, AU_M, 0.306);
        let bare = equilibrium_temperature_k(L_SUN_W, AU_M, 0.0);
        assert!((with_air - 254.031).abs() < 0.01, "measured {with_air}");
        assert!((bare - 278.321).abs() < 0.01, "measured {bare}");
    }

    #[test]
    fn flux_radius_is_the_exact_inverse_of_equilibrium_temperature() {
        for (l, d, a) in [
            (L_SUN_W, AU_M, 0.0),
            (L_SUN_W, AU_M, 0.306),
            (0.0009726074241780799 * L_SUN_W, 5.393206e9, 0.1),
        ] {
            let t = equilibrium_temperature_k(l, d, a);
            let back = flux_radius_m(l, t, a);
            assert!((back / d - 1.0).abs() < 1e-12, "round-trip {back} vs {d}");
        }
    }

    #[test]
    fn the_frost_line_and_the_temperature_law_agree_on_ice_condensation() {
        // Two independent code paths: T_eq at the frost line vs Hayashi's ~170 K
        // condensation temperature — agree to 0.4% (169.4 K), for any luminosity (the
        // sqrt(L) scaling cancels).
        for l_lsun in [1.0, 0.0009726074241780799] {
            let frost_m = frost_line_radius_au(l_lsun, FROST_COEFF_AU) * AU_M;
            let t = equilibrium_temperature_k(l_lsun * L_SUN_W, frost_m, 0.0);
            assert!((t - 169.4).abs() < 0.1, "measured {t}");
        }
    }

    // ---- Mass-radius tables through the one evaluator ---------------------------------

    #[test]
    fn segmented_power_law_selects_segments_and_reuses_the_top_tail() {
        let segs = SpectralClass::MLR_SEGMENTS;
        // Below the first break, mid-table, above every break (the unwrap_or arm).
        assert!((segmented_power_law(0.3, &segs) - 0.23 * 0.3f64.powf(2.3)).abs() < 1e-12);
        assert!((segmented_power_law(1.5, &segs) - 5.0625).abs() < 1e-9);
        assert!((segmented_power_law(60.0, &segs) - 1.4 * 60.0f64.powf(3.5)).abs() < 1e-6);
    }

    #[test]
    fn star_radius_reproduces_the_worlds_three_stars_and_the_sun() {
        // Demircan & Kahraman through the one evaluator; THE world's pinned masses.
        assert!((star_radius_m(0.09287894638451702) - 7.805661e7).abs() < 1e2);
        assert!((star_radius_m(0.1081418058358058) - 9.012636e7).abs() < 1e2);
        assert!((star_radius_m(0.16179874709518627) - 1.318892e8).abs() < 1e2);
        // The Sun: 1.06 Rsun at 1 Msun (the fit's stated coefficient).
        assert!((star_radius_m(1.0) - 1.06 * R_SUN_M).abs() < 1.0);
        // The high segment drives (an A star).
        assert!((star_radius_m(2.0) - R_SUN_M * 1.2917 * 2.0f64.powf(0.555)).abs() < 1.0);
    }

    #[test]
    fn planet_radius_chen_kipping_goldens() {
        // The re-solve's three named rows: minimum draw, the 1.3786 Mearth geometric mean
        // of the home log-uniform, the Neptunian segment.
        assert!((planet_radius_m(0.0553) / 1.0e3 - 2_863.0).abs() < 1.0);
        assert!((planet_radius_m(1.3786) / 1.0e3 - 7_024.0).abs() < 1.0);
        // Continuity at each break is a MEASURED discontinuity, never smoothed.
        let below = planet_radius_m(2.04 - 1e-9);
        let above = planet_radius_m(2.04 + 1e-9);
        assert!(
            (below / above - 1.0).abs() < 0.02,
            "Terran->Neptunian break"
        );
        let below2 = planet_radius_m(131.6 - 1e-6);
        let above2 = planet_radius_m(131.6 + 1e-6);
        assert!(
            (below2 / above2 - 1.0).abs() < 0.05,
            "Neptunian->Jovian break"
        );
        // Stellar segments: continuity at 1.66 Msun.
        let s_below = star_radius_m(1.66 - 1e-9);
        let s_above = star_radius_m(1.66 + 1e-9);
        assert!((s_below / s_above - 1.0).abs() < 0.02, "Demircan break");
    }

    #[test]
    fn the_valley_normalisation_reproduces_its_published_anchor() {
        // 1.121 * 121^0.11 = 1.8998 — the Martinez 2019 anchor (1.90 Rearth at S = 121).
        let anchored = radius_valley_rearth(
            121.0,
            RADIUS_VALLEY_1SEARTH_REARTH,
            RADIUS_VALLEY_INSOLATION_EXP,
        );
        assert!((anchored - 1.90).abs() < 1e-3, "measured {anchored}");
        assert_eq!(
            radius_valley_rearth(
                1.0,
                RADIUS_VALLEY_1SEARTH_REARTH,
                RADIUS_VALLEY_INSOLATION_EXP
            ),
            RADIUS_VALLEY_1SEARTH_REARTH
        );
    }

    #[test]
    fn the_envelope_law_reproduces_the_bimodality_at_the_temperate_rung() {
        // Home rung 2 (S = 0.748315). The design pass's worked example used the
        // geometric-mean mass 1.3786 Mearth — but its OWN stripped ceiling at this rung is
        // 1.3565 Mearth (par 3.4.5), so at 1.3786 the bare core (1.0905) already sits a
        // hair ABOVE the valley (1.0858) and the "opposite sides" claim fails by its own
        // table. MEASURED here: the numeric pins hold; the bimodality is demonstrated at
        // 1.0 Mearth, below the ceiling, where it is real.
        let s = 0.748_314_795;
        let valley = radius_valley_rearth(
            s,
            RADIUS_VALLEY_1SEARTH_REARTH,
            RADIUS_VALLEY_INSOLATION_EXP,
        );
        assert!((valley - 1.0858).abs() < 1e-3);
        let geo_stripped = segmented_power_law(1.3786, &ROCK_MR_SEGMENTS);
        let geo_retained = geo_stripped + envelope_radius_rearth(1.3786, 0.05, s, SYSTEM_AGE_GYR);
        assert!((geo_stripped - 1.0905).abs() < 1e-3);
        assert!((geo_retained - 2.992).abs() < 2e-3);
        assert!(
            geo_stripped > valley,
            "the design example's own inconsistency, pinned"
        );
        // Below the 1.3565 Mearth stripped ceiling the two outcomes bracket the valley.
        let stripped = segmented_power_law(1.0, &ROCK_MR_SEGMENTS);
        let retained = stripped + envelope_radius_rearth(1.0, 0.05, s, SYSTEM_AGE_GYR);
        assert!(stripped < valley);
        assert!(valley < retained);
        // The age normalisation point is the identity (t = 5 Gyr => the last factor is 1).
        let at_norm = envelope_radius_rearth(1.0, ENVELOPE_F_ENV_NORM, 1.0, SYSTEM_AGE_GYR);
        assert!((at_norm - ENVELOPE_LF14.0).abs() < 1e-12);
    }

    #[test]
    fn gravity_escape_and_scale_height_goldens() {
        // Earth: g = 9.81, v_esc = 11.18 km/s, N2 scale height ~8.6 km at 254 K...
        let g = surface_gravity_mps2(M_EARTH_KG, R_EARTH_M);
        assert!((g - 9.82).abs() < 0.02, "measured {g}");
        let v = escape_velocity_mps(M_EARTH_KG, R_EARTH_M);
        assert!((v - 11_180.0).abs() < 10.0, "measured {v}");
        let h = scale_height_m(254.0, MU_N2, g);
        assert!((h - 7_680.0).abs() < 100.0, "measured {h}");
        // The thin-shell identity: rho_0 * 4 pi R^2 H == M_env exactly.
        let rho = envelope_reference_density_kgm3(1.0e20, 7.0e6, 1.0e5);
        assert!(
            (rho * 4.0 * core::f64::consts::PI * 7.0e6f64 * 7.0e6 * 1.0e5 - 1.0e20).abs() < 1.0,
        );
    }

    #[test]
    fn the_moon_laws_reproduce_the_galilean_scale() {
        // Jupiter: d_Roche ~ 1.51e8 m (ice), R_H ~ 5.31e10 m, disc edge ~ 1.11e9 m.
        let roche = roche_radius_m(1.898e27, RHO_ICE_KGM3);
        assert!((roche / 1.513e8 - 1.0).abs() < 1e-2, "measured {roche}");
        let hill = hill_radius_m(5.2044 * AU_M, 1.898e27, M_SUN_KG);
        assert!((hill / 5.315e10 - 1.0).abs() < 1e-2, "measured {hill}");
        let disc = satellite_disc_edge_m(hill, SATELLITE_DISC_HILL_FRACTION);
        assert!((disc / 1.107e9 - 1.0).abs() < 1e-2, "measured {disc}");
        // The ladder's +1 offset: rung 0 sits at ratio * Roche, never AT the limit.
        assert_eq!(satellite_ladder_a_m(0, roche, 1.7), roche * 1.7);
        assert!((satellite_ladder_a_m(2, roche, 1.7) - roche * 1.7f64.powf(3.0)).abs() < 1.0);
    }

    // ---- samplers ----------------------------------------------------------------

    #[test]
    fn sample_imf_mass_endpoints_and_midpoint() {
        // Salpeter: endpoints land on the bounds (within libm tolerance, NOT bit-exact).
        assert!((sample_imf_mass(0.0, 2.35, 0.08, 120.0) - 0.08).abs() < 1e-12);
        assert!((sample_imf_mass(1.0, 2.35, 0.08, 120.0) - 120.0).abs() < 1e-9 * 120.0);
        // Midpoint golden (recomputed: the transposed 0.1206 was wrong).
        assert!((sample_imf_mass(0.5, 2.35, 0.08, 120.0) - 0.133_677_6).abs() < 1e-4);
        // The slope==1 log-uniform false arm: 0.4 * 10^0.5.
        let log_limit = sample_imf_mass(0.5, 1.0, 0.4, 4.0);
        assert!((log_limit - 0.4 * 10.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn sample_imf_mass_is_monotone_and_in_range() {
        let (a, lo, hi) = (2.35, 0.08, 120.0);
        let mut prev = sample_imf_mass(0.0, a, lo, hi);
        for i in 1..=100u32 {
            let u = f64::from(i) / 101.0;
            let m = sample_imf_mass(u, a, lo, hi);
            assert!(m >= prev - 1e-9, "monotone non-decreasing at u={u}");
            assert!((lo..=hi).contains(&m), "in [m_lo, m_hi] at u={u}");
            prev = m;
        }
    }

    #[test]
    fn sample_rayleigh_known_u01_goldens() {
        let sigma = 0.03;
        // u=0 -> 0 (note: the closed form yields -0.0, which == 0.0; intentional, no branch).
        assert_eq!(sample_rayleigh(0.0, sigma), 0.0);
        // The u that makes x == sigma and x == 2*sigma exactly.
        let u_sigma = 1.0 - (-0.5f64).exp();
        let u_2sigma = 1.0 - (-2.0f64).exp();
        assert!((sample_rayleigh(u_sigma, sigma) - sigma).abs() < 1e-12);
        assert!((sample_rayleigh(u_2sigma, sigma) - 2.0 * sigma).abs() < 1e-12);
    }

    #[test]
    fn sample_rayleigh_is_nonnegative() {
        for i in 0..100u32 {
            let u = f64::from(i) / 100.0;
            assert!(sample_rayleigh(u, 0.05) >= 0.0);
        }
    }

    #[test]
    fn orbital_axis_au_geometric_golden() {
        assert_eq!(orbital_axis_au(0, 0.4, 1.7), 0.4); // ratio^0 == 1
        assert!((orbital_axis_au(1, 0.4, 1.7) - 0.68).abs() < 1e-12);
        assert!((orbital_axis_au(4, 0.4, 1.7) - 0.4 * 1.7_f64.powf(4.0)).abs() < 1e-12);
        // Monotone increasing in the slot index.
        assert!(orbital_axis_au(5, 0.4, 1.7) > orbital_axis_au(4, 0.4, 1.7));
    }

    #[test]
    fn sample_galaxy_type_categorical_boundaries() {
        let c = &GalaxyType::CANONICAL_CUMULATIVE; // [0.72, 0.90]
        assert_eq!(sample_galaxy_type(0.0, c), GalaxyType::Spiral);
        assert_eq!(sample_galaxy_type(0.72, c), GalaxyType::Elliptical); // on-boundary pins `>=`
        assert_eq!(sample_galaxy_type(0.80, c), GalaxyType::Elliptical);
        assert_eq!(sample_galaxy_type(0.95, c), GalaxyType::Irregular);
    }

    // ---- serde (fieldless enums only — the Defs carry a &'static str, not derivable) ---

    #[test]
    fn taxonomy_enums_serde_roundtrip() {
        for x in GalaxyType::ALL {
            let bytes = postcard::to_allocvec(&x).expect("encode");
            assert_eq!(
                postcard::from_bytes::<GalaxyType>(&bytes).expect("decode"),
                x
            );
        }
        for x in SpectralClass::ALL {
            let bytes = postcard::to_allocvec(&x).expect("encode");
            assert_eq!(
                postcard::from_bytes::<SpectralClass>(&bytes).expect("decode"),
                x
            );
        }
        for x in PlanetType::ALL {
            let bytes = postcard::to_allocvec(&x).expect("encode");
            assert_eq!(
                postcard::from_bytes::<PlanetType>(&bytes).expect("decode"),
                x
            );
        }
    }
}
