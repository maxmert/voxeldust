//! Frozen bit-pin for the D-45(a) Slice-1 ephemeris — a HOST-LOCAL regression guard.
//!
//! This is deliberately an INTEGRATION test, NOT a `celestial::tests` unit test: the
//! justfile coverage recipes exclude `/tests/` from the HR5 100% region+branch gate
//! (`--ignore-filename-regex '(/bin/|/tests/)'`), so an EXACT-f64-bits assertion here can
//! never contaminate the coverage gate the way it would inside the Tier-A `core` unit
//! suite (a single-ULP libm shift between the coverage nightly and the stable dev
//! toolchain would otherwise flip a `to_bits()` literal into a spurious gate-blocking RED).
//!
//! What this pins: the exact rotation convention + math of [`orbital_state`]. A same-host
//! refactor that changes results — a flipped glam multiply order, a dropped `μ/h` scale, a
//! sign error — flips these bits and fails loudly, where the tolerant property tests (which
//! pass under small drift) would not.
//!
//! Cross-host caveat: cross-OS/CPU bit-equality of the transcendental chain
//! (`sin`/`cos`/`sqrt`/`atan2` + glam rotation) is NOT guaranteed until the SPIKE-6a
//! build-and-diff gate (step-2 / P4). Until then a mismatch on a DIFFERENT host is
//! EXPECTED, not a regression — re-capture on that host or gate it there. The bits below
//! were captured on the dev host (macOS / aarch64; the coverage `nightly-2026-06-06` and
//! stable `1.94.1` agree on this machine).

use vd_core::celestial::{OrbitalElements, orbital_state};

/// A representative non-trivial orbit — all six elements + the parent mass exercised, at
/// ~1 AU / solar mass so the numbers are astronomically realistic.
const PIN_ELEMENTS: OrbitalElements = OrbitalElements {
    sma: 1.495_978_7e11,  // ~1 AU (m)
    ecc: 0.016_7,         // Earth-like
    inclination: 0.409_0, // ~23.4° obliquity (rad)
    raan: 1.796_3,
    arg_periapsis: 1.993_3,
    mean_anomaly_epoch: 6.259_6,
    central_mass: 1.989e30, // ~1 solar mass (kg)
};
const PIN_TIME_S: f64 = 5_000_000.0;

#[test]
fn ephemeris_frozen_bit_pin() {
    let s = orbital_state(&PIN_ELEMENTS, PIN_TIME_S);
    let bits = [
        s.position.x.to_bits(),
        s.position.y.to_bits(),
        s.position.z.to_bits(),
        s.velocity.x.to_bits(),
        s.velocity.y.to_bits(),
        s.velocity.z.to_bits(),
    ];
    // Captured from the first green run on the dev host (see module docs re: cross-host).
    // Verified physically at capture: |r| = 0.991 AU ∈ [a(1−e), a(1+e)], and vis-viva +
    // specific energy hold to ~4e-16 (machine precision) — the pin encodes a REAL orbit,
    // not an arbitrary regression snapshot.
    let expected: [u64; 6] = [
        0x4208_8f8d_18e3_b438, // position.x =  1.318595e10 m
        0xc241_2813_c8da_7543, // position.y = -1.473737e11 m
        0x4200_3a5a_4c5a_bcb5, // position.z =  8.712309e9  m
        0x40da_f8c4_629f_9a14, // velocity.x =  2.761907e4  m/s
        0x4095_2248_3b28_5b10, // velocity.y =  1.352571e3  m/s
        0xc0c7_0ba3_ced2_383f, // velocity.z = -1.179928e4  m/s
    ];
    assert_eq!(
        bits, expected,
        "ephemeris bits drifted. If this is a DIFFERENT host/OS/CPU it is EXPECTED \
         pre-SPIKE-6a (re-capture there); if the SAME host, a math/rotation-order \
         regression. Actual: {bits:#018x?}"
    );
}
