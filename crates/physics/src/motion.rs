//! THE ONE MOTION DISCRIMINANT (SL4, the placement arc). Both arms of every match here write the
//! SAME kind of value into the SAME row shape — a [`FramePlacement`] — so nobody downstream can tell
//! which arm ran: a ship, a station, a moon and a rock cross by identical code because that code
//! cannot tell them apart. Adding a way of moving is a NEW ARM HERE, ZERO consumer change (HR4 — the
//! [`Motion::Integrated`] arm is that claim's standing proof).
//!
//! Motion reaches the simulation ONLY as an opaque [`MotionFn`] closure (injected at boot, the
//! `sim::io` seam discipline) or as authored placement-book rows — never as a nameable type: the
//! crossing path has no Cargo edge to this crate (`crate_isolation` gate).

use std::sync::Arc;

use glam::DVec3;
use vd_core::frame::FramePlacement;
use vd_core::placement::MotionFn;

use crate::celestial::{OrbitalElements, orbital_state};

/// How ONE child moves in its parent's frame. PRIVATE knowledge of this crate's callers (boot
/// composition, the demand spawner): consumers read rows, never arms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    /// A child that never moves off its authored placement (a station parked in space, every
    /// walk-scale body).
    Fixed(FramePlacement),
    /// A Keplerian orbit — the closed-form Category-A ephemeris ([`orbital_state`]).
    Kepler(OrbitalElements),
    /// Integrator-carried state: an epoch placement advanced ballistically under a constant
    /// acceleration (closed form: `p + v·t + a·t²/2`, `v + a·t` — deterministic, no per-tick
    /// accumulation). The P5/P8 rapier checkpoint rides this arm; today it is the HR4/G-IDENTICAL
    /// proof arm — a thrusting child crosses by the same code as an orbiting one because the row
    /// they write is the same row.
    Integrated {
        /// The epoch placement (position + velocity at `secs == 0`).
        placement: FramePlacement,
        /// Constant parent-frame acceleration (m/s²) — a thruster burn, a uniform field.
        acceleration: DVec3,
    },
}

impl Motion {
    /// THE ONE motion evaluation: this child's placement at `secs` since epoch. Every arm writes the
    /// SAME kind of value; monomorphic, all arms covered in this crate's own tests (HR5).
    #[must_use]
    pub fn state_at(&self, secs: f64) -> FramePlacement {
        match self {
            Motion::Fixed(placement) => *placement,
            Motion::Kepler(elements) => {
                let st = orbital_state(elements, secs);
                FramePlacement::moving(st.position, st.velocity)
            }
            Motion::Integrated {
                placement,
                acceleration,
            } => FramePlacement {
                origin: placement.origin
                    + placement.velocity * secs
                    + *acceleration * (0.5 * secs * secs),
                velocity: placement.velocity + *acceleration * secs,
                ..*placement
            },
        }
    }

    /// The WORST instant, in CLOSED FORM, for the boot fence: how far this child's centre can get
    /// from its parent's origin. `|origin|` for a fixed child; APOAPSIS `a·(1+e)` for a Kepler one;
    /// the CURRENT offset for an integrated one (an integrated child's future is its controller's —
    /// the thruster BUDGET bound lands with P8, and until then a fence over an integrated child is
    /// re-judged from its checkpoint, never promised ahead).
    ///
    /// Cell anchors: every shipped placement is authored at cell ZERO (the tiered-excursion fold is
    /// D-41's), so the f64 origin IS the offset.
    #[must_use]
    pub fn max_excursion_m(&self) -> f64 {
        match self {
            Motion::Fixed(placement) => placement.origin.length(),
            Motion::Kepler(elements) => elements.sma * (1.0 + elements.ecc),
            Motion::Integrated { placement, .. } => placement.origin.length(),
        }
    }

    /// This child's own closing speed (m/s) — the scalar the AoI band is widened by. vd-core receives
    /// a NUMBER, never a motion: the fifth rival has-orbit test (`orbital_of(..).map_or(0.0, v_peri)`)
    /// collapsed into this one accessor.
    #[must_use]
    pub fn closing_speed_mps(&self) -> f64 {
        match self {
            Motion::Fixed(_) => 0.0,
            Motion::Kepler(elements) => elements.v_peri(),
            Motion::Integrated { placement, .. } => placement.velocity.length(),
        }
    }
}

/// Wrap a [`Motion`] into the opaque [`MotionFn`] seam the boot injects into the simulation's
/// placement writer. The closure captures the motion BY VALUE; the sim can run it, never name it.
#[must_use]
pub fn motion_fn(motion: Motion) -> MotionFn {
    MotionFn(Arc::new(move |secs| motion.state_at(secs)))
}

/// The Kepler convenience the boot uses for every seed-generated mover.
#[must_use]
pub fn kepler_motion_fn(elements: OrbitalElements) -> MotionFn {
    motion_fn(Motion::Kepler(elements))
}

/// A whole seed-derived mover roster, wrapped for injection — the one conversion the boot makes
/// between "what the generator knows" (elements) and "what the simulation may hold" (opaque fns).
#[must_use]
pub fn kepler_motion_fns<K: Ord>(
    movers: std::collections::BTreeMap<K, OrbitalElements>,
) -> std::collections::BTreeMap<K, MotionFn> {
    movers
        .into_iter()
        .map(|(k, e)| (k, kepler_motion_fn(e)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;

    fn elements() -> OrbitalElements {
        OrbitalElements {
            sma: 1.2e7,
            ecc: 0.3,
            inclination: 0.5,
            raan: 0.4,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 5.972e24,
        }
    }

    #[test]
    fn every_arm_writes_the_same_kind_of_row() {
        // The SL4 shape as a measurement: three ways of moving, ONE row type out, and the values are
        // each arm's own closed form — a consumer holding the rows cannot reconstruct which arm ran
        // from their type.
        let fixed = Motion::Fixed(FramePlacement::moving(
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::ZERO,
        ));
        assert_eq!(
            fixed.state_at(0.0),
            fixed.state_at(1.0e6),
            "a fixed child never moves off its authored placement"
        );

        let kepler = Motion::Kepler(elements());
        let st = orbital_state(&elements(), 137.0);
        assert_eq!(
            kepler.state_at(137.0),
            FramePlacement::moving(st.position, st.velocity),
            "a Kepler child's row IS the closed-form ephemeris"
        );

        let integrated = Motion::Integrated {
            placement: FramePlacement::moving(DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 2.0, 0.0)),
            acceleration: DVec3::new(0.0, 0.0, 4.0),
        };
        let at = integrated.state_at(3.0);
        // p + v·t + a·t²/2 and v + a·t, exactly.
        assert_eq!(at.origin, DVec3::new(1.0, 6.0, 18.0));
        assert_eq!(at.velocity, DVec3::new(0.0, 2.0, 12.0));
        assert_eq!(at.orientation, DQuat::IDENTITY);
    }

    #[test]
    fn the_worst_instant_is_closed_form_per_arm() {
        assert_eq!(
            Motion::Fixed(FramePlacement::moving(
                DVec3::new(3.0, 4.0, 0.0),
                DVec3::ZERO
            ))
            .max_excursion_m(),
            5.0
        );
        assert_eq!(
            Motion::Kepler(elements()).max_excursion_m(),
            1.2e7 * 1.3,
            "a Kepler child's worst instant is its apoapsis"
        );
        assert_eq!(
            Motion::Integrated {
                placement: FramePlacement::moving(DVec3::new(0.0, 3.0, 4.0), DVec3::X),
                acceleration: DVec3::ZERO,
            }
            .max_excursion_m(),
            5.0
        );
    }

    #[test]
    fn the_closing_speed_is_a_scalar_per_arm() {
        assert_eq!(
            Motion::Fixed(FramePlacement::identity()).closing_speed_mps(),
            0.0
        );
        assert_eq!(
            Motion::Kepler(elements()).closing_speed_mps(),
            elements().v_peri()
        );
        assert_eq!(
            Motion::Integrated {
                placement: FramePlacement::moving(DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0)),
                acceleration: DVec3::ZERO,
            }
            .closing_speed_mps(),
            5.0
        );
    }

    #[test]
    fn a_roster_wraps_every_mover_into_the_motion_made_opaque() {
        // The MAP form (`kepler_motion_fns`) — covered IN THIS CRATE, at the production `RealmId`
        // instantiation (HR5 discipline (b): llvm counts a generic's regions per monomorphization
        // per test binary, and every other instantiation site rides vd-sim's DEV-ONLY edge, which
        // the planned writer-in-node rework (D-PLACE-4) deletes — without an in-crate test this fn
        // would silently VANISH from the 100% gate rather than fail it; batch review).
        use vd_core::pose::RealmId;
        let movers = std::collections::BTreeMap::from([
            (RealmId::Planet(7), elements()),
            (RealmId::Planet(8), elements()),
        ]);
        let fns = kepler_motion_fns(movers);
        assert_eq!(fns.len(), 2, "one opaque fn per mover, keyed identically");
        for f in fns.values() {
            assert_eq!((f.0)(42.0), Motion::Kepler(elements()).state_at(42.0));
        }
    }

    #[test]
    fn a_motion_fn_is_the_motion_made_opaque() {
        // The seam: the closure's answers are the motion's answers, bit for bit — and its type names
        // nothing (the simulation can run it, never ask what it is).
        let m = Motion::Kepler(elements());
        let f = motion_fn(m);
        for secs in [0.0, 1.0, 999.5] {
            assert_eq!((f.0)(secs), m.state_at(secs));
        }
        let k = kepler_motion_fn(elements());
        assert_eq!((k.0)(42.0), Motion::Kepler(elements()).state_at(42.0));
    }
}
