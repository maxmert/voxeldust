//! The core-side taxonomy TAGS (D-45(a) Slice 2). The real astrophysical CLASSIFICATION +
//! SAMPLING half (galaxy morphology, spectral classes, planet types, the IMF and Rayleigh
//! closed forms) moved to `vd_physics::taxonomy` with the seed generator it feeds (the
//! placement arc S5: anything that reads the seed stream or mints a body lives in the motion
//! crate; core keeps the tags every layer names).
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

    #[test]
    fn profile_kind_serde_roundtrips() {
        for x in ProfileKind::ALL {
            let bytes = postcard::to_allocvec(&x).expect("encode");
            assert_eq!(
                postcard::from_bytes::<ProfileKind>(&bytes).expect("decode"),
                x
            );
        }
    }
}
