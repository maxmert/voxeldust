//! ★ THE EMULATED VIEW AND SAMPLER (slice K0c; the far-water design §5.1, `sampler.rs`).
//!
//! THE VIEW: an eye at a stated distance and incidence from a point of the sea's sphere; each
//! pixel's ray meets the sphere, and its footprint is the change of the hit's face parameters over
//! one pixel along the image's two axes (central differences), as a GPU reads a texture
//! coordinate's derivatives.
//!
//! THE SAMPLER: an anisotropic trilinear filter of the common form — `N = min(⌈P_max / P_min⌉, 16)`
//! probes spaced along the longer derivative, the level `λ = log2(P_max / N)` (the GL extension's
//! sketch) or `λ = log2(P_max / η)` with `η` the ratio itself (the Vulkan specification's sketch;
//! the specifications leave the rule to the vendor, [`LOD_FORM`]) raised to the level floor and
//! clamped to the source's levels, two levels mixed by λ's fraction, and each probe a
//! bilinear read clamped to the level's edge — with its weights in f64 or at 8 sub-texel bits. The
//! same filter reads a chunk's window, the far page, or the one global pyramid (the reference), so
//! a front line compares two SOURCES under one filter, never two filters.
//!
//! **Example.** A pilot at 15 664 km over the high lakes sees the 5 882 m lake's piece at the
//! 14 → 15 front. The pixel's footprint is 18 km, 1.1 texels of the far page's level 14: the
//! filter reads levels 14 and 15 with one probe, and the rung-14 chunk and the rung-15 chunk both
//! read the far page there, so they draw the same value.

use vd_core::glam::DVec3;
use vd_seed::bend::{Face, K1, K2, K3, face_coords, face_of};

/// The most probes a pixel takes (the 16× sampler).
pub const ANISO_MAX: f64 = 16.0;

/// ★ THE LEVEL RULE of every read this run (the named choice `skin_lod`): 0 the level of
/// `P_max / N`, 1 the level of `P_max / η`. Set once, before any read.
pub static LOD_FORM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// ★ A SOURCE the sampler reads: its finest and coarsest absolute levels, and its 8-bit texel at an
/// absolute level and GLOBAL texel index (clamped to the source's own edge by the source).
pub struct Src<'a> {
    pub lo: u8,
    pub hi: u8,
    pub fetch: &'a dyn Fn(u8, i64, i64) -> u8,
}

/// ★ THE FILTER: the value (0..=255) of `src` at `pos` — a position in GLOBAL texel units of level
/// `src.lo` — with the pixel's two derivatives `dx`, `dy` in the same units, the level floor
/// `floor` (in levels over `src.lo`), and the weights in f64 (`quant` false) or at 8 bits.
#[must_use]
pub fn sample(
    src: &Src,
    pos: (f64, f64),
    dx: (f64, f64),
    dy: (f64, f64),
    floor: f64,
    quant: bool,
) -> f64 {
    let (lx, ly) = (dx.0.hypot(dx.1), dy.0.hypot(dy.1));
    let (major, p_max, p_min) = if lx >= ly { (dx, lx, ly) } else { (dy, ly, lx) };
    let eta = (p_max / p_min.max(1e-12)).clamp(1.0, ANISO_MAX);
    let n = eta.ceil();
    let per_probe = if LOD_FORM.load(std::sync::atomic::Ordering::Relaxed) == 1 {
        p_max / eta
    } else {
        p_max / n
    };
    let levels = f64::from(src.hi - src.lo);
    let lambda = per_probe.max(1e-12).log2().max(floor).clamp(0.0, levels);
    let l0 = lambda.floor();
    let mut f = lambda - l0;
    if quant {
        f = (f * 256.0).floor() / 256.0;
    }
    let l0 = l0 as u8;
    let l1 = (l0 + 1).min(src.hi - src.lo);
    let probes = n as usize;
    let mut acc = 0.0;
    for k in 0..probes {
        let s = (k as f64 + 0.5) / n - 0.5;
        let p = (pos.0 + major.0 * s, pos.1 + major.1 * s);
        let a = bilinear(src, l0, p, quant);
        let b = if f > 0.0 {
            bilinear(src, l1, p, quant)
        } else {
            a
        };
        acc += (1.0 - f) * a + f * b;
    }
    acc / n
}

/// One bilinear tap at level `src.lo + k`; `p` in level-`src.lo` texel units.
fn bilinear(src: &Src, k: u8, p: (f64, f64), quant: bool) -> f64 {
    let scale = f64::from(1u32 << k);
    let (u, v) = (p.0 / scale - 0.5, p.1 / scale - 0.5);
    let (i0, j0) = (u.floor(), v.floor());
    let (mut fu, mut fv) = (u - i0, v - j0);
    if quant {
        fu = (fu * 256.0).floor() / 256.0;
        fv = (fv * 256.0).floor() / 256.0;
    }
    let (i0, j0) = (i0 as i64, j0 as i64);
    let l = src.lo + k;
    let t = |i: i64, j: i64| f64::from((src.fetch)(l, i, j));
    let top = (1.0 - fu) * t(i0, j0) + fu * t(i0 + 1, j0);
    let bot = (1.0 - fu) * t(i0, j0 + 1) + fu * t(i0 + 1, j0 + 1);
    (1.0 - fv) * top + fv * bot
}

/// ★ AN EYE: its place in the body's frame (metres), its axes, and the pixel's angle.
#[derive(Clone, Copy, Debug)]
pub struct Eye {
    pub pos: DVec3,
    pub fwd: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub alpha: f64,
}

impl Eye {
    /// ★ The eye that sees the sphere point `p` at distance `d`, at incidence `theta` (radians
    /// from the surface's normal), tilted toward azimuth `az`.
    #[must_use]
    pub fn looking_at(p: DVec3, d: f64, theta: f64, az: f64, alpha: f64) -> Eye {
        let n = p.normalize();
        let seed = if n.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let t0 = n.cross(seed).normalize();
        let t1 = n.cross(t0);
        let t = t0 * az.cos() + t1 * az.sin();
        let v = n * theta.cos() + t * theta.sin();
        let pos = p + v * d;
        let fwd = -v;
        // The image's x axis lies along the surface, across the tilt.
        let right = fwd.cross(n.cross(t).normalize()).normalize();
        let right = if right.is_finite() && right.length() > 0.5 {
            right
        } else {
            t0
        };
        let up = right.cross(fwd).normalize();
        Eye {
            pos,
            fwd,
            right,
            up,
            alpha,
        }
    }

    /// ★ The eye `h` over the sphere point `p` (radius `r`), looking straight down.
    #[must_use]
    pub fn over(p: DVec3, h: f64, alpha: f64) -> Eye {
        Eye::looking_at(p, h, 0.0, 0.0, alpha)
    }

    /// The ray of pixel `(x, y)`, in pixels from the image's centre.
    #[must_use]
    pub fn ray(&self, x: f64, y: f64) -> DVec3 {
        (self.fwd + self.right * (x * self.alpha).tan() + self.up * (y * self.alpha).tan())
            .normalize()
    }

    /// Where pixel `(x, y)`'s ray first meets the sphere of radius `r`.
    #[must_use]
    pub fn hit(&self, x: f64, y: f64, r: f64) -> Option<DVec3> {
        let dir = self.ray(x, y);
        let b = self.pos.dot(dir);
        let c = self.pos.length_squared() - r * r;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let s = -b - disc.sqrt();
        (s > 0.0).then(|| self.pos + dir * s)
    }

    /// ★ THE FOOTPRINT of pixel `(x, y)` on the sphere of radius `r`.
    #[must_use]
    pub fn footprint(&self, x: f64, y: f64, r: f64) -> Option<Hit> {
        let p = self.hit(x, y, r)?;
        let dirp = p.normalize();
        let arr = |v: DVec3| [v.x, v.y, v.z];
        let face = face_of(arr(dirp));
        let params = |q: DVec3| {
            let (t, s) = face_coords(face, arr(q.normalize()));
            (unbend_free(t), unbend_free(s))
        };
        let (a, b) = params(p);
        let (xp, xm) = (self.hit(x + 0.5, y, r)?, self.hit(x - 0.5, y, r)?);
        let (yp, ym) = (self.hit(x, y + 0.5, r)?, self.hit(x, y - 0.5, r)?);
        let (ax, bx) = (params(xp), params(xm));
        let (ay, by) = (params(yp), params(ym));
        Some(Hit {
            face,
            a,
            b,
            d: (p - self.pos).length(),
            dx: (ax.0 - bx.0, ax.1 - bx.1),
            dy: (ay.0 - by.0, ay.1 - by.1),
        })
    }
}

/// The face parameter whose bend is `t`, with no clamp to the face (so a derivative taken across a
/// face edge is not flattened): Newton's steps on `W(a) = k₁a + k₂a³ + k₃a⁵`.
#[must_use]
pub fn unbend_free(t: f64) -> f64 {
    let mut a = t;
    for _ in 0..8 {
        let a2 = a * a;
        let f = a * (K1 + a2 * (K2 + a2 * K3)) - t;
        let df = K1 + a2 * (3.0 * K2 + a2 * (5.0 * K3));
        a -= f / df;
    }
    a
}

/// ★ ONE PIXEL'S HIT: its face and face parameters, its distance from the eye, and the change of
/// the parameters over one pixel along the image's x and y.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub face: Face,
    pub a: f64,
    pub b: f64,
    pub d: f64,
    pub dx: (f64, f64),
    pub dy: (f64, f64),
}

impl Hit {
    /// The hit in global texel units of a level with `n_l` texels a face edge, and its two
    /// derivatives in the same units.
    #[must_use]
    pub fn in_texels(&self, n_l: f64) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let s = 0.5 * n_l;
        (
            ((self.a + 1.0) * s, (self.b + 1.0) * s),
            (self.dx.0 * s, self.dx.1 * s),
            (self.dy.0 * s, self.dy.1 * s),
        )
    }
}
