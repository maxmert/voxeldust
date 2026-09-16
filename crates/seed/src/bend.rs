//! ★ THE FACE BEND — the map from a flat cube face onto the sphere, and its inverse.
//!
//! A face position `a` runs from `-1` to `+1` across a face. The bend turns it into a tangent value
//! `W(a)`, and one normalisation puts the direction `(n + W(a)·u + W(b)·v)` on the unit sphere. The
//! bend is a polynomial of degree five whose three constants sum to exactly `1.0`, so `W(1) = 1` and a
//! face edge lands exactly on the cube edge. Its derivative at the centre is `π/4`, which makes a cell
//! at the face centre exactly one metre and the radius arithmetic closed-form.
//!
//! **Everything here is a function of its inputs and of the constants below, computed with
//! operations IEEE-754 fixes on every target** (SL10 V1.4): add, subtract, multiply, divide and
//! square root; the round-to-integral family (`floor`, `ceil`, `round`); comparison, `abs`, `clamp`;
//! saturating integer casts; and integer shifts and `leading_zeros`. No transcendental function, no
//! fused multiply-add (Rust does not contract), no `f64::min`/`max`. ONE evaluation order, so the
//! server and every client name the same cell for the same position. The constants, the evaluation
//! order (Horner form), the inverse's FIRST GUESS and its STEP COUNT are part of the generator's
//! world identity (ruling V6, Format D): change any of them and every stored edit on every planet
//! re-addresses.
//!
//! **Scalars in the bend.** The bend runs on `f64` and `[f64; 3]`, under the leaf's own float fence
//! (`clippy.toml`): plain scalar arithmetic, no vector library. The boundary to the position lattice
//! lives in `vd-core`'s grid, above this crate.

/// `k₁ = π/4` — the standard library's CONSTANT (a literal bit pattern, never a function call).
pub const K1: f64 = std::f64::consts::FRAC_PI_4;
/// `k₂` — the cubic weight (the investigation's `VD-ASC5`).
pub const K2: f64 = 0.15;
/// `k₃ = 1 − k₁ − k₂` in THIS order, so that `(k₁ + k₂) + k₃` and `k₁ + (k₂ + k₃)` are both exactly
/// `1.0` in IEEE `f64` (asserted at compile time below; MEASURED 2026-09-07 in Python first, and the
/// crate's own assertion is the gate).
pub const K3: f64 = 1.0 - K1 - K2;
/// The Newton inverse runs exactly this many steps from the first guess `a₀ = t`. FOUR reaches the
/// `f64` floor from either guess (MEASURED: 2.2 × 10⁻¹⁶ worst residual over 200 001 samples); three
/// from the obvious alternative guess would leave a third of a cell at the largest legal body.
pub const INVERSE_STEPS: u32 = 4;

// THE CONSTANTS' SUM, in both evaluation orders, at compile time (U-2). A build on any target that
// cannot make these exact does not build at all.
const _: () = assert!((K1 + K2) + K3 == 1.0);
const _: () = assert!(K1 + (K2 + K3) == 1.0);

/// The forward bend `W(a) = k₁a + k₂a³ + k₃a⁵`, in Horner form: `a·(k₁ + a²·(k₂ + a²·k₃))`.
#[must_use]
pub fn bend(a: f64) -> f64 {
    let a2 = a * a;
    a * (K1 + a2 * (K2 + a2 * K3))
}

/// The inverse bend: the `a` with `W(a) = t`, by [`INVERSE_STEPS`] Newton steps from `a₀ = t`, then
/// clamped to the face. The derivative `k₁ + a²·(3k₂ + a²·5k₃)` is positive everywhere (every constant
/// is positive), so no step divides by zero. A `t` a hair outside `[−1, 1]` — which rounding at a cube
/// edge can produce — clamps to the edge rather than naming a cell off the face.
#[must_use]
pub fn unbend(t: f64) -> f64 {
    let mut a = t;
    let mut step = 0;
    while step < INVERSE_STEPS {
        let a2 = a * a;
        let f = a * (K1 + a2 * (K2 + a2 * K3)) - t;
        let df = K1 + a2 * (3.0 * K2 + a2 * (5.0 * K3));
        a -= f / df;
        step += 1;
    }
    clamp_unit(a)
}

/// Clamp to `[−1, 1]`. `f64::clamp` is two comparisons that return the input itself when it is not
/// outside the range — never `f64::min`/`max`, whose result on a signed zero the standard leaves
/// unspecified (ruling V6 D, the fenced float's rule, kept here too).
fn clamp_unit(a: f64) -> f64 {
    a.clamp(-1.0, 1.0)
}

/// One of the six cube faces. The discriminant is the `face` byte of a cell address and is part of
/// the address format (ruling V6, Format A): it never renumbers.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

/// An integer unit axis: exactly one non-zero component, `±1`. Integer, so the basis table and the
/// seam table are exact and can be built at compile time.
pub type Axis = [i8; 3];

/// A face's basis: its outward normal `n` and its two tangent axes `u` (the `i` direction) and `v`
/// (the `j` direction), right-handed as `(u, v, n)`: `u × v = n` on every face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceBasis {
    pub n: Axis,
    pub u: Axis,
    pub v: Axis,
}

/// THE FACE BASIS TABLE — part of the address format. A cell's `(i, j)` counts along `u` and `v`.
pub const BASIS: [FaceBasis; 6] = [
    // PosX: (u, v, n) = (+Y, +Z, +X)
    FaceBasis {
        n: [1, 0, 0],
        u: [0, 1, 0],
        v: [0, 0, 1],
    },
    // NegX: (u, v, n) = (+Z, +Y, −X)
    FaceBasis {
        n: [-1, 0, 0],
        u: [0, 0, 1],
        v: [0, 1, 0],
    },
    // PosY: (u, v, n) = (+Z, +X, +Y)
    FaceBasis {
        n: [0, 1, 0],
        u: [0, 0, 1],
        v: [1, 0, 0],
    },
    // NegY: (u, v, n) = (+X, +Z, −Y)
    FaceBasis {
        n: [0, -1, 0],
        u: [1, 0, 0],
        v: [0, 0, 1],
    },
    // PosZ: (u, v, n) = (+X, +Y, +Z)
    FaceBasis {
        n: [0, 0, 1],
        u: [1, 0, 0],
        v: [0, 1, 0],
    },
    // NegZ: (u, v, n) = (+Y, +X, −Z)
    FaceBasis {
        n: [0, 0, -1],
        u: [0, 1, 0],
        v: [1, 0, 0],
    },
];

impl Face {
    /// Every face, in discriminant order.
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];

    /// The face byte of an address.
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// The face for an address byte; `None` for a byte no face owns (a decoder REFUSES, never defaults).
    #[must_use]
    pub const fn from_index(b: u8) -> Option<Face> {
        match b {
            0 => Some(Face::PosX),
            1 => Some(Face::NegX),
            2 => Some(Face::PosY),
            3 => Some(Face::NegY),
            4 => Some(Face::PosZ),
            5 => Some(Face::NegZ),
            _ => None,
        }
    }

    /// This face's basis row.
    #[must_use]
    pub const fn basis(self) -> FaceBasis {
        BASIS[self as usize]
    }

    /// The face whose outward normal is `axis`. Every unit axis names exactly one face, so this is
    /// total over unit axes; a non-unit axis (never produced by the table) answers `PosX` by the
    /// fall-through, which the seam-table test excludes by construction.
    #[must_use]
    pub const fn with_normal(axis: Axis) -> Face {
        let mut f = 0;
        while f < 6 {
            if axis_eq(BASIS[f].n, axis) {
                // SAFETY-FREE: `f < 6` and every value in `0..6` is a face discriminant.
                return match f {
                    0 => Face::PosX,
                    1 => Face::NegX,
                    2 => Face::PosY,
                    3 => Face::NegY,
                    4 => Face::PosZ,
                    _ => Face::NegZ,
                };
            }
            f += 1;
        }
        Face::PosX
    }
}

/// Exact integer-axis equality, usable at compile time.
#[must_use]
pub const fn axis_eq(a: Axis, b: Axis) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

/// The negated axis.
#[must_use]
pub const fn axis_neg(a: Axis) -> Axis {
    [-a[0], -a[1], -a[2]]
}

/// `d · axis` — exact, because the axis components are `0` or `±1`.
#[must_use]
pub fn dot_axis(d: [f64; 3], axis: Axis) -> f64 {
    d[0] * f64::from(axis[0]) + d[1] * f64::from(axis[1]) + d[2] * f64::from(axis[2])
}

/// Which face a direction falls on: the axis with the largest magnitude, its sign choosing the side.
/// Ties resolve `x` before `y` before `z` (a point exactly on a cube edge belongs to ONE face), and a
/// zero component counts as positive, so the answer is total and deterministic.
#[must_use]
pub fn face_of(d: [f64; 3]) -> Face {
    let ax = d[0].abs();
    let ay = d[1].abs();
    let az = d[2].abs();
    if ax >= ay && ax >= az {
        if d[0] >= 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if d[1] >= 0.0 { Face::PosY } else { Face::NegY }
    } else if d[2] >= 0.0 {
        Face::PosZ
    } else {
        Face::NegZ
    }
}

/// The face coordinates `(t, s)` of a direction on `face`: the tangents `d·u / d·n` and `d·v / d·n`.
/// On the face [`face_of`] chose, `d·n` is the largest component and positive, so both lie in
/// `[−1, 1]` up to rounding.
#[must_use]
pub fn face_coords(face: Face, d: [f64; 3]) -> (f64, f64) {
    let b = face.basis();
    let dn = dot_axis(d, b.n);
    (dot_axis(d, b.u) / dn, dot_axis(d, b.v) / dn)
}

/// The unit direction of face position `(a, b)` on `face`: `normalize(n + W(a)·u + W(b)·v)`.
#[must_use]
pub fn direction(face: Face, a: f64, b: f64) -> [f64; 3] {
    let basis = face.basis();
    let wa = bend(a);
    let wb = bend(b);
    let x = f64::from(basis.n[0]) + wa * f64::from(basis.u[0]) + wb * f64::from(basis.v[0]);
    let y = f64::from(basis.n[1]) + wa * f64::from(basis.u[1]) + wb * f64::from(basis.v[1]);
    let z = f64::from(basis.n[2]) + wa * f64::from(basis.u[2]) + wb * f64::from(basis.v[2]);
    normalize([x, y, z])
}

/// THE INTEGER DIRECTION (ruling F7): the unit direction of cell `(face, i, j)` at 40 fraction bits
/// from the recipe's integer bend, for a body whose cell-count reciprocal at this rung is `inv_n`
/// (`vd_recipe::bend::inv_n_of(n_l)`). The same word triple on every host; the float [`direction`]
/// stays beside it for the inverse path until its integer form lands.
#[must_use]
pub fn direction_q(face: Face, i: i32, j: i32, inv_n: vd_recipe::Gi) -> [vd_recipe::Gi; 3] {
    vd_recipe::bend::direction(i32::from(face.index()), i, j, inv_n)
}

/// `v / |v|`, one square root and one divide per component, in this order.
#[must_use]
pub fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
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

    /// The face byte on the wire is the address format's face number (ruling V6, Format A) — the
    /// serde index, pinned face by face, so a reorder can never re-label a cell address.
    #[test]
    fn every_face_encodes_as_its_address_byte() {
        let table = [
            (Face::PosX, 0u8),
            (Face::NegX, 1),
            (Face::PosY, 2),
            (Face::NegY, 3),
            (Face::PosZ, 4),
            (Face::NegZ, 5),
        ];
        for (face, byte) in table {
            assert_eq!(
                postcard::to_allocvec(&face).expect("encodes"),
                vec![byte],
                "{face:?}"
            );
            assert_eq!(face as u8, byte, "{face:?}: the repr and the wire agree");
            assert_eq!(postcard::from_bytes::<Face>(&[byte]), Ok(face));
        }
        assert!(
            postcard::from_bytes::<Face>(&[6]).is_err(),
            "a seventh face is refused"
        );
    }

    #[test]
    fn the_bend_is_odd_monotone_and_reaches_the_edge_exactly() {
        assert_eq!(bend(0.0), 0.0);
        assert_eq!(
            bend(1.0),
            1.0,
            "W(1) = 1: the face edge lands on the cube edge"
        );
        assert_eq!(bend(-1.0), -1.0);
        let mut last = bend(-1.0);
        let mut i = 1;
        while i <= 2000 {
            let a = -1.0 + f64::from(i) / 1000.0;
            let w = bend(a);
            assert!(w > last, "monotone at a = {a}");
            assert!((bend(-a) + w).abs() < 1e-15, "odd at a = {a}");
            last = w;
            i += 1;
        }
    }

    #[test]
    fn the_inverse_reaches_the_f64_floor_across_the_face_at_the_largest_legal_body() {
        // U-1: the residual in CELLS at N = 2^26, sampled densely, from the frozen guess and count.
        let n = f64::from(1u32 << 26);
        let mut worst_cells = 0.0_f64;
        let mut i = 0;
        while i <= 200_000 {
            let a = -1.0 + f64::from(i) / 100_000.0;
            let back = unbend(bend(a));
            let cells = (back - a).abs() * n / 2.0;
            if cells > worst_cells {
                worst_cells = cells;
            }
            i += 1;
        }
        assert!(
            worst_cells < 1e-6,
            "the inverse bend's worst residual is {worst_cells} cells at N = 2^26 (gate: < 1e-6)"
        );
    }

    #[test]
    fn the_inverse_clamps_a_tangent_that_rounding_pushed_past_the_edge() {
        assert_eq!(unbend(1.000_000_1), 1.0, "past +1 clamps to the edge");
        assert_eq!(unbend(-1.000_000_1), -1.0, "past -1 clamps to the edge");
        assert_eq!(unbend(0.0), 0.0, "the centre is a fixed point");
        assert_eq!(unbend(1.0), 1.0, "the edge is a fixed point");
    }

    #[test]
    fn every_face_basis_is_right_handed_and_names_a_distinct_normal() {
        for f in Face::ALL {
            let b = f.basis();
            // u × v == n
            let cross = [
                b.u[1] * b.v[2] - b.u[2] * b.v[1],
                b.u[2] * b.v[0] - b.u[0] * b.v[2],
                b.u[0] * b.v[1] - b.u[1] * b.v[0],
            ];
            assert_eq!(cross, b.n, "face {f:?} basis is right-handed");
            assert_eq!(Face::with_normal(b.n), f, "the normal names the face back");
            assert_eq!(Face::from_index(f.index()), Some(f), "the byte round-trips");
        }
        assert_eq!(Face::from_index(6), None, "a byte no face owns is refused");
        assert_eq!(
            Face::with_normal([0, 0, 0]),
            Face::PosX,
            "the fall-through arm (never reached by a unit axis) is total"
        );
        assert_eq!(axis_neg([1, 0, 0]), [-1, 0, 0]);
    }

    #[test]
    fn face_of_picks_the_largest_axis_and_resolves_ties_and_zeros_deterministically() {
        assert_eq!(face_of([0.9, 0.1, 0.2]), Face::PosX);
        assert_eq!(face_of([-0.9, 0.1, 0.2]), Face::NegX);
        assert_eq!(face_of([0.1, 0.9, 0.2]), Face::PosY);
        assert_eq!(face_of([0.1, -0.9, 0.2]), Face::NegY);
        assert_eq!(face_of([0.1, 0.2, 0.9]), Face::PosZ);
        assert_eq!(face_of([0.1, 0.2, -0.9]), Face::NegZ);
        // Ties: x before y before z; an edge point belongs to one face.
        assert_eq!(face_of([0.7, 0.7, 0.1]), Face::PosX);
        assert_eq!(face_of([0.1, 0.7, 0.7]), Face::PosY);
        assert_eq!(
            face_of([0.0, 0.0, 0.0]),
            Face::PosX,
            "a zero direction is still total"
        );
    }

    #[test]
    fn direction_and_face_coords_round_trip_on_every_face() {
        for f in Face::ALL {
            let mut i = 0;
            while i <= 20 {
                let a = -1.0 + f64::from(i) / 10.0;
                let b = 1.0 - f64::from(i) / 10.0;
                let d = direction(f, a, b);
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-15, "unit length on {f:?}");
                // An interior point lands on its own face. A point ON a cube edge or corner ties, and
                // the tie rule gives it to ONE neighbouring face — by design, so the face check skips
                // the edge samples; the coordinate round trip below still holds on the source face.
                if (a.abs() < 1.0) & (b.abs() < 1.0) {
                    assert_eq!(
                        face_of(d),
                        f,
                        "the direction lands on its own face ({f:?}, {a}, {b})"
                    );
                }
                let (t, s) = face_coords(f, d);
                assert!((unbend(t) - a).abs() < 1e-12, "a round-trips on {f:?}");
                assert!((unbend(s) - b).abs() < 1e-12, "b round-trips on {f:?}");
                i += 1;
            }
        }
    }

    #[test]
    fn the_cell_edge_spread_across_a_face_is_the_measured_band() {
        // U-3: the tangential cell edge under the bend, scaled so the face-centre cell is 1.000 m at
        // N cells per edge: |d(direction)/da| · (2/N) · R with R = 2N/π ⇒ |d dir/da| · 4/π.
        let mut lo = f64::MAX;
        let mut hi = 0.0_f64;
        let h = 1e-6;
        let mut i = 0;
        while i <= 400 {
            let a = -1.0 + f64::from(i) / 200.0;
            let mut j = 0;
            while j <= 400 {
                let b = -1.0 + f64::from(j) / 200.0;
                let p = direction(Face::PosX, a - h, b);
                let q = direction(Face::PosX, a + h, b);
                let d = [
                    (q[0] - p[0]) / (2.0 * h),
                    (q[1] - p[1]) / (2.0 * h),
                    (q[2] - p[2]) / (2.0 * h),
                ];
                // The cell edge is |∂dir/∂a| · (2/N) · R = |∂dir/∂a| · 4/π = |∂dir/∂a| / k₁.
                let edge = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / K1;
                lo = if edge < lo { edge } else { lo };
                hi = if edge > hi { edge } else { hi };
                j += 40;
            }
            i += 40;
        }
        let centre = {
            let p = direction(Face::PosX, -h, 0.0);
            let q = direction(Face::PosX, h, 0.0);
            let d = [
                (q[0] - p[0]) / (2.0 * h),
                (q[1] - p[1]) / (2.0 * h),
                (q[2] - p[2]) / (2.0 * h),
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / K1
        };
        assert!(
            (centre - 1.0).abs() < 1e-6,
            "the face-centre cell is 1.000 m (got {centre})"
        );
        assert!(lo > 0.70, "the narrowest cell edge is ~0.707 m (got {lo})");
        assert!(lo < 0.72, "the narrowest cell edge is ~0.707 m (got {lo})");
        assert!(hi > 0.99, "the widest cell edge is ~1.005 m (got {hi})");
        assert!(hi < 1.02, "the widest cell edge is ~1.005 m (got {hi})");
    }

    #[test]
    fn the_integer_direction_matches_the_float_bend_within_a_few_units() {
        // MEASURED (bench part 3): within 0.02 mm laterally on the home planet — under four units
        // of 2⁻⁴⁰; eight is the bar here, on every face, at the centre, the edges and a corner.
        let n_l = 9_961_472u32;
        let inv_n = vd_recipe::bend::inv_n_of(n_l);
        let one = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        for face in [
            Face::PosX,
            Face::NegX,
            Face::PosY,
            Face::NegY,
            Face::PosZ,
            Face::NegZ,
        ] {
            for (i, j) in [
                (0, 0),
                (4_980_736, 4_980_736),
                (9_961_471, 0),
                (7, 9_961_471),
            ] {
                let f = direction(
                    face,
                    crate::ladder::face_param(i, n_l),
                    crate::ladder::face_param(j, n_l),
                );
                let q = direction_q(face, i, j, inv_n);
                for c in 0..3 {
                    let units = (q[c].raw() as f64 - f[c] * one).abs();
                    assert!(units <= 8.0, "{face:?} ({i}, {j}) [{c}]: {units} units");
                }
            }
        }
        // The recipe's basis table is this crate's.
        for face in [
            Face::PosX,
            Face::NegX,
            Face::PosY,
            Face::NegY,
            Face::PosZ,
            Face::NegZ,
        ] {
            let b = face.basis();
            // ★ THE KERNEL'S OWN FORM, not the table beside it: `basis_of` is the function every
            // direction on either host reads, and `BASIS` is read by no kernel at all. Comparing
            // the table alone would leave a mistyped row in the match invisible to every gate.
            let r = vd_recipe::bend::basis_of(i32::from(face.index()));
            let wide = |a: Axis| [i32::from(a[0]), i32::from(a[1]), i32::from(a[2])];
            assert_eq!(
                (wide(b.n), wide(b.u), wide(b.v)),
                (r.n, r.u, r.v),
                "{face:?}"
            );
            assert_eq!(r, vd_recipe::bend::BASIS[face.index() as usize], "{face:?}");
        }
    }
}
