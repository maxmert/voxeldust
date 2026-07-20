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

/// A byte carried a taxonomy tag this build does not know (the shared `from_tag` error).
/// `kind` names which registry, so the three round-trip tests give distinct messages (HR2:
/// an unknown tag is an error, never decode-to-Default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown taxonomy tag {kind}:{tag}")]
pub struct UnknownTaxonTag {
    pub kind: &'static str,
    pub tag: u8,
}

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralDef {
    pub tag: u8,
    pub name: &'static str,
    pub mass_lo_msun: f64,
    pub teff_lo_k: f64,
    pub teff_hi_k: f64,
}

impl SpectralDef {
    /// Coherence rules (checked over `SpectralClass::ALL`): a positive lower mass bound and
    /// a well-ordered, positive T_eff band. Named AND'd booleans (HR5: each clause's false
    /// side is driven independently by `taxa_incoherent_defs_are_detected`).
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        let mass_positive = self.mass_lo_msun > 0.0;
        let band_ordered = self.teff_lo_k < self.teff_hi_k;
        let band_positive = self.teff_lo_k > 0.0;
        mass_positive && band_ordered && band_positive
    }
}

pub static O_DEF: SpectralDef = SpectralDef {
    tag: 0,
    name: "O",
    mass_lo_msun: MASS_O_LO_MSUN,
    teff_lo_k: TEFF_O_LO_K,
    teff_hi_k: f64::INFINITY,
};
pub static B_DEF: SpectralDef = SpectralDef {
    tag: 1,
    name: "B",
    mass_lo_msun: MASS_B_LO_MSUN,
    teff_lo_k: TEFF_B_LO_K,
    teff_hi_k: TEFF_O_LO_K,
};
pub static A_DEF: SpectralDef = SpectralDef {
    tag: 2,
    name: "A",
    mass_lo_msun: MASS_A_LO_MSUN,
    teff_lo_k: TEFF_A_LO_K,
    teff_hi_k: TEFF_B_LO_K,
};
pub static F_DEF: SpectralDef = SpectralDef {
    tag: 3,
    name: "F",
    mass_lo_msun: MASS_F_LO_MSUN,
    teff_lo_k: TEFF_F_LO_K,
    teff_hi_k: TEFF_A_LO_K,
};
pub static G_DEF: SpectralDef = SpectralDef {
    tag: 4,
    name: "G",
    mass_lo_msun: MASS_G_LO_MSUN,
    teff_lo_k: TEFF_G_LO_K,
    teff_hi_k: TEFF_F_LO_K,
};
pub static K_DEF: SpectralDef = SpectralDef {
    tag: 5,
    name: "K",
    mass_lo_msun: MASS_K_LO_MSUN,
    teff_lo_k: TEFF_K_LO_K,
    teff_hi_k: TEFF_G_LO_K,
};
pub static M_DEF: SpectralDef = SpectralDef {
    tag: 6,
    name: "M",
    mass_lo_msun: MASS_M_LO_MSUN,
    teff_lo_k: TEFF_M_LO_K,
    teff_hi_k: TEFF_K_LO_K,
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
    let (_, coeff, exponent) = segments
        .iter()
        .find(|(mass_hi, _, _)| mass_msun <= *mass_hi)
        .copied()
        .unwrap_or(segments[2]);
    coeff * mass_msun.powf(exponent)
}

/// Flux-balance habitable-zone radius `sqrt(L/flux_limit)` in AU (Kopparapu 2013): pass the
/// moist-inner `S_in~1.1` or conservative-outer `S_out~0.53` flux limit for the edges, or
/// `1.0` for the `r = sqrt(L)` centre.
#[must_use]
pub fn habitable_zone_radius_au(luminosity_lsun: f64, flux_limit: f64) -> f64 {
    (luminosity_lsun / flux_limit).sqrt()
}

// ===== Planet type (Pollack 1996 core accretion; Hayashi 1981 frost line) ============

/// Solar-nebula water-ice condensation radius at `L = Lsun` (Hayashi 1981), scaled by
/// `sqrt(L)`; passed to [`frost_line_radius_au`].
pub const FROST_COEFF_AU: f64 = 2.7;
const M_OCEAN_LO_MEARTH: f64 = 2.0;
const M_GAS_MEARTH: f64 = 10.0;
const M_CORE_CRIT_MEARTH: f64 = 10.0;

/// Formation-based planet class from orbital distance vs the frost line and planet mass
/// (Pollack 1996 core accretion).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PlanetType {
    Rocky = 0,
    Ocean = 1,
    GasGiant = 2,
    IceGiant = 3,
}

/// Static per-type record. `is_giant` is a clean per-type fact; `requires_frost_line` is
/// true ONLY for `IceGiant` (its ices condense only beyond the line) — gas giants form on
/// BOTH sides (hot Jupiters migrate inward), so this is NOT a "gas giants are always inside"
/// flag. Not serde-carried (off-wire descriptor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetDef {
    pub tag: u8,
    pub name: &'static str,
    pub is_giant: bool,
    /// This type can form ONLY beyond the frost line (true for `IceGiant` alone).
    pub requires_frost_line: bool,
}

impl PlanetDef {
    /// Coherence: anything that can form only beyond the frost line must be a giant (the ice
    /// giants). Named boolean (HR5: the false side is driven by an incoherent hand-built Def).
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        !self.requires_frost_line || self.is_giant
    }
}

pub static ROCKY_DEF: PlanetDef = PlanetDef {
    tag: 0,
    name: "Rocky",
    is_giant: false,
    requires_frost_line: false,
};
pub static OCEAN_DEF: PlanetDef = PlanetDef {
    tag: 1,
    name: "Ocean",
    is_giant: false,
    requires_frost_line: false,
};
pub static GAS_GIANT_DEF: PlanetDef = PlanetDef {
    tag: 2,
    name: "GasGiant",
    is_giant: true,
    requires_frost_line: false,
};
pub static ICE_GIANT_DEF: PlanetDef = PlanetDef {
    tag: 3,
    name: "IceGiant",
    is_giant: true,
    requires_frost_line: true,
};

/// The planet-type mass thresholds in Earth masses (params-as-args; `CANONICAL` supplies the
/// Pollack 1996 defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrostThresholds {
    pub m_ocean_lo_mearth: f64,
    pub m_gas_mearth: f64,
    pub m_core_crit_mearth: f64,
}

impl FrostThresholds {
    pub const CANONICAL: FrostThresholds = FrostThresholds {
        m_ocean_lo_mearth: M_OCEAN_LO_MEARTH,
        m_gas_mearth: M_GAS_MEARTH,
        m_core_crit_mearth: M_CORE_CRIT_MEARTH,
    };
}

impl PlanetType {
    /// All planet types in tag order.
    pub const ALL: [PlanetType; 4] = [
        PlanetType::Rocky,
        PlanetType::Ocean,
        PlanetType::GasGiant,
        PlanetType::IceGiant,
    ];

    /// Recover a type from its tag byte; unknown tags error (HR2), never Default.
    pub fn from_tag(tag: u8) -> Result<PlanetType, UnknownTaxonTag> {
        match tag {
            0 => Ok(PlanetType::Rocky),
            1 => Ok(PlanetType::Ocean),
            2 => Ok(PlanetType::GasGiant),
            3 => Ok(PlanetType::IceGiant),
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
        }
    }
}

/// Water-ice condensation radius `frost_coeff_au * sqrt(L/Lsun)` in AU (Hayashi 1981).
#[must_use]
pub fn frost_line_radius_au(luminosity_lsun: f64, frost_coeff_au: f64) -> f64 {
    frost_coeff_au * luminosity_lsun.sqrt()
}

/// Pure threshold tree (Pollack 1996). Inside the frost line: Rocky / Ocean / hot-GasGiant by
/// mass; beyond it: IceGiant below the critical core mass, GasGiant above (runaway gas
/// accretion). All thresholds come from `th` — no magic numbers. Note a GasGiant can be
/// returned on EITHER side (a migrated hot Jupiter inside, a Jupiter beyond).
#[must_use]
pub fn classify_planet(
    orbit_au: f64,
    frost_line_au: f64,
    mass_mearth: f64,
    th: FrostThresholds,
) -> PlanetType {
    if orbit_au < frost_line_au {
        if mass_mearth < th.m_ocean_lo_mearth {
            PlanetType::Rocky
        } else if mass_mearth < th.m_gas_mearth {
            PlanetType::Ocean
        } else {
            PlanetType::GasGiant
        }
    } else if mass_mearth < th.m_core_crit_mearth {
        PlanetType::IceGiant
    } else {
        PlanetType::GasGiant
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

// ===== ProfileKind — the core-side capability tag ====================================

/// The core-side capability TAG of a generated body/realm: the input to the ONE core->sim
/// map [`capability::profile_for`](../../vd_sim/capability/fn.profile_for.html) (in vd-sim,
/// where `ShardProfile` lives). This tag enum LEADS `RealmId`: `Galaxy`/`Asteroid` tag body
/// kinds whose `RealmId` arms land at Slice 4 / a later asteroid phase, and `Stub` is the
/// bare P3 empty-space subject — so some arms are exercised by the `ALL`-loop test but have
/// no Slice-3 producer yet (intended). `Galaxy` tags BOTH the Universe root and a Galaxy
/// relay realm (both -> `profiles::galaxy()`: signal-relay, no voxel).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProfileKind {
    Galaxy = 0,
    System = 1,
    Planet = 2,
    Ship = 3,
    Asteroid = 4,
    Station = 5,
    Area = 6,
    Stub = 7,
}

impl ProfileKind {
    /// All profile kinds in tag order; `profile_for` is total over this set (compiler-enforced).
    pub const ALL: [ProfileKind; 8] = [
        ProfileKind::Galaxy,
        ProfileKind::System,
        ProfileKind::Planet,
        ProfileKind::Ship,
        ProfileKind::Asteroid,
        ProfileKind::Station,
        ProfileKind::Area,
        ProfileKind::Stub,
    ];

    /// Recover a profile kind from its tag byte; unknown tags error (HR2).
    pub fn from_tag(tag: u8) -> Result<ProfileKind, UnknownTaxonTag> {
        match tag {
            0 => Ok(ProfileKind::Galaxy),
            1 => Ok(ProfileKind::System),
            2 => Ok(ProfileKind::Planet),
            3 => Ok(ProfileKind::Ship),
            4 => Ok(ProfileKind::Asteroid),
            5 => Ok(ProfileKind::Station),
            6 => Ok(ProfileKind::Area),
            7 => Ok(ProfileKind::Stub),
            other => Err(UnknownTaxonTag {
                kind: "profile",
                tag: other,
            }),
        }
    }
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
                | PlanetType::IceGiant => {}
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

    #[test]
    fn profile_kind_registry_cannot_drift() {
        for x in ProfileKind::ALL {
            match x {
                ProfileKind::Galaxy
                | ProfileKind::System
                | ProfileKind::Planet
                | ProfileKind::Ship
                | ProfileKind::Asteroid
                | ProfileKind::Station
                | ProfileKind::Area
                | ProfileKind::Stub => {}
            }
        }
        let mut accepted = 0usize;
        for tag in 0..=u8::MAX {
            if let Ok(x) = ProfileKind::from_tag(tag) {
                accepted += 1;
                assert_eq!(x as u8, tag);
                assert!(ProfileKind::ALL.contains(&x));
            }
        }
        assert_eq!(accepted, ProfileKind::ALL.len());
        assert_eq!(
            ProfileKind::from_tag(200).expect_err("unknown").to_string(),
            "unknown taxonomy tag profile:200"
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
        // PlanetDef: requires-frost-line yet not a giant.
        let bad_planet = PlanetDef {
            requires_frost_line: true,
            is_giant: false,
            ..ROCKY_DEF
        };
        assert!(!bad_planet.is_coherent());
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

    #[test]
    fn classify_planet_golden_and_boundaries() {
        let th = FrostThresholds::CANONICAL;
        let frost = 2.7;
        // Inside the frost line, by mass.
        assert_eq!(classify_planet(1.0, frost, 1.0, th), PlanetType::Rocky);
        assert_eq!(classify_planet(1.0, frost, 5.0, th), PlanetType::Ocean);
        assert_eq!(classify_planet(1.0, frost, 100.0, th), PlanetType::GasGiant); // hot Jupiter
        // Beyond the frost line, by core-critical mass.
        assert_eq!(classify_planet(30.0, frost, 8.0, th), PlanetType::IceGiant);
        assert_eq!(classify_planet(5.2, frost, 318.0, th), PlanetType::GasGiant); // Jupiter
        // Boundaries: mass == ocean floor -> Ocean (the `<` cut); orbit == frost line -> beyond.
        assert_eq!(
            classify_planet(1.0, frost, th.m_ocean_lo_mearth, th),
            PlanetType::Ocean
        );
        assert_eq!(classify_planet(frost, frost, 8.0, th), PlanetType::IceGiant);
    }

    #[test]
    fn gas_giant_can_form_on_either_side_of_the_frost_line() {
        // The PlanetDef.requires_frost_line flag must NOT be read as "gas giants are inside".
        let th = FrostThresholds::CANONICAL;
        let inside = classify_planet(1.0, 2.7, 100.0, th);
        let beyond = classify_planet(5.2, 2.7, 318.0, th);
        assert_eq!(inside, PlanetType::GasGiant);
        assert_eq!(beyond, PlanetType::GasGiant);
        assert!(!GAS_GIANT_DEF.requires_frost_line);
        assert!(ICE_GIANT_DEF.requires_frost_line);
    }

    // ---- samplers -------------------------------------------------------------------

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
        for x in ProfileKind::ALL {
            let bytes = postcard::to_allocvec(&x).expect("encode");
            assert_eq!(
                postcard::from_bytes::<ProfileKind>(&bytes).expect("decode"),
                x
            );
        }
    }
}
