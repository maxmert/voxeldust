//! ★ THE PANELS (plan §10): for the owner's look, each stand at rungs 3, 7 and 11 under R0, N1, N2
//! and O1 side by side — a TOP-DOWN map of the drawn water over a hillshade of the drawn ground,
//! and an OBLIQUE view (a ray cast over the drawn surface). Written with the workspace's `image`
//! crate (a dev-dependency of vd-bins; no new library). Each file is captioned in its name and in a
//! PANEL line of the census ("no far shadow").
//!
//! **Example.** `plain_r07_top_R0-N1-N2-O1.png`: four 320-pixel tiles of the plain's 254 km window at
//! rung 7 — today's forty blue tiles top left, N1 top right, N2 bottom left, O1 bottom right (the
//! first round painted the pure contour where N2 stands now) — each a 16-grey cartographic hillshade (light from the north-west, 45° up) with the drawn
//! water in the game's flat blue.

use image::{Rgb, RgbImage};
use vd_bins::shore_geometry::unit_of;
use vd_core::glam::DVec3;
use vd_recipe::plan::site_direction;
use vd_seed::bend::{face_coords, face_of, unbend};

use crate::census::Census;
use crate::eval::{RungOut, Var};

/// A tile's side in pixels, and the oblique tile's height.
pub const TILE: u32 = 320;
pub const TILE_OBLIQUE_H: u32 = 213;
/// The game's flat water (status §3, 8o), sRGB bytes.
const WATER: [u8; 3] = [15, 61, 107];
/// The sky of the oblique view: plain grey; the ground past the window: a darker grey.
const SKY: [u8; 3] = [178, 186, 196];
const OUTSIDE: [u8; 3] = [60, 60, 60];

/// One tile's source: per column its drawn surface, its standing water (NaN none) and whether it
/// is drawn wet, row-major `b · n + a`.
pub struct TileSrc {
    pub n: usize,
    pub s: Vec<f32>,
    pub l: Vec<f32>,
    pub wet: Vec<bool>,
}

impl TileSrc {
    /// A rule's tile from a window's rung.
    #[must_use]
    pub fn of(out: &RungOut, v: Var) -> TileSrc {
        let vi = v.idx();
        let n = out.n;
        let mut s = vec![f32::NAN; n * n];
        let mut l = vec![f32::NAN; n * n];
        let mut wet = vec![false; n * n];
        for (k, col) in out.cols.iter().enumerate() {
            if let Some(c) = col {
                s[k] = c.drawn[vi].s;
                l[k] = c.drawn[vi].l;
                wet[k] = c.drawn[vi].wet();
            }
        }
        TileSrc { n, s, l, wet }
    }

    /// The drawn height: the water's level over a wet column, the ground elsewhere.
    fn height(&self, k: usize) -> f32 {
        if self.wet[k] { self.l[k] } else { self.s[k] }
    }

    /// The Lambert shade of each column under the cartographic light (azimuth 315°, altitude 45°,
    /// in the window's tangent frame: `+i` east, `+j` north), no vertical exaggeration.
    fn shades(&self, m_i: f64, m_j: f64) -> Vec<f32> {
        let n = self.n;
        let (az, alt) = (315f64.to_radians(), 45f64.to_radians());
        let light = (alt.cos() * az.sin(), alt.cos() * az.cos(), alt.sin());
        let mut out = vec![0.0f32; n * n];
        for b in 0..n {
            for a in 0..n {
                let k = b * n + a;
                let h = |aa: usize, bb: usize| -> f64 {
                    let v = self.s[bb * n + aa];
                    if v.is_nan() {
                        f64::from(self.s[k])
                    } else {
                        f64::from(v)
                    }
                };
                let (a0, a1) = (a.saturating_sub(1), (a + 1).min(n - 1));
                let (b0, b1) = (b.saturating_sub(1), (b + 1).min(n - 1));
                let dx = (h(a1, b) - h(a0, b)) / ((a1 - a0).max(1) as f64 * m_i);
                let dy = (h(a, b1) - h(a, b0)) / ((b1 - b0).max(1) as f64 * m_j);
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let lam = (-dx * light.0 - dy * light.1 + light.2) / len;
                out[k] = lam.max(0.0) as f32;
            }
        }
        out
    }
}

/// Sixteen grey levels of a shade in `[0, 1]`.
fn grey(shade: f64) -> [u8; 3] {
    let g = ((shade.clamp(0.0, 1.0) * 15.0).round() / 15.0 * 235.0 + 10.0) as u8;
    [g, g, g]
}

/// ★ ONE TOP-DOWN TILE: a pixel covers a box of columns; its grey is their mean shade; it is the
/// water's blue where at least half of its columns are drawn wet (a picture choice, stated).
fn top_tile(img: &mut RgbImage, ox: u32, oy: u32, src: &TileSrc, m_i: f64, m_j: f64) {
    let n = src.n;
    let shades = src.shades(m_i, m_j);
    let side = f64::from(TILE);
    for py in 0..TILE {
        for px in 0..TILE {
            let a0 = (f64::from(px) / side * n as f64) as usize;
            let a1 = ((f64::from(px + 1) / side * n as f64) as usize)
                .max(a0 + 1)
                .min(n);
            // The picture's up is +j.
            let b0 = (f64::from(TILE - 1 - py) / side * n as f64) as usize;
            let b1 = ((f64::from(TILE - py) / side * n as f64) as usize)
                .max(b0 + 1)
                .min(n);
            let (mut sum, mut cnt, mut wet) = (0.0f64, 0usize, 0usize);
            for b in b0..b1 {
                for a in a0..a1 {
                    let k = b * n + a;
                    if src.s[k].is_nan() {
                        continue;
                    }
                    sum += f64::from(shades[k]);
                    cnt += 1;
                    wet += usize::from(src.wet[k]);
                }
            }
            let px_rgb = if cnt == 0 {
                OUTSIDE
            } else if wet * 2 >= cnt {
                WATER
            } else {
                grey(sum / cnt as f64)
            };
            img.put_pixel(ox + px, oy + py, Rgb(px_rgb));
        }
    }
}

/// The window's geometry for the oblique view: its face and origin at the rung, the rung's cells
/// on a face edge, and the ladder radius.
pub struct Geo<'a> {
    pub face: vd_seed::bend::Face,
    pub i0: i32,
    pub j0: i32,
    pub n_cells: u32,
    pub radius: f64,
    pub charter: &'a vd_recipe::plan::PlanCharter,
    pub m_col: f64,
}

impl Geo<'_> {
    fn dir(&self, i: i32, j: i32) -> DVec3 {
        unit_of(site_direction(
            self.charter,
            i32::from(self.face.index()),
            i,
            j,
        ))
    }
    /// A point's continuous column coordinates in the window (column centres at whole numbers).
    fn cols_of(&self, p: DVec3) -> Option<(f64, f64)> {
        let d = [p.x, p.y, p.z];
        if face_of(d) != self.face {
            return None;
        }
        let (t, s) = face_coords(self.face, d);
        let (a, b) = (unbend(t), unbend(s));
        let nn = f64::from(self.n_cells);
        Some((
            (a + 1.0) * 0.5 * nn - 0.5 - f64::from(self.i0),
            (b + 1.0) * 0.5 * nn - 0.5 - f64::from(self.j0),
        ))
    }
}

/// ★ ONE OBLIQUE TILE: an eye over the point 0.55·D along −j from the window's middle, 0.18·D
/// over the median drawn surface, looking at the middle with a 55° horizontal field; rays march a
/// few columns at a time over the drawn surface (the water flat at its level), the curvature exact.
fn oblique_tile(
    img: &mut RgbImage,
    ox: u32,
    oy: u32,
    src: &TileSrc,
    geo: &Geo,
    m_i: f64,
    m_j: f64,
) {
    let n = src.n;
    let shades = src.shades(m_i, m_j);
    let mut hs: Vec<f64> = (0..n * n)
        .map(|k| f64::from(src.height(k)))
        .filter(|h| h.is_finite())
        .collect();
    if hs.is_empty() {
        return;
    }
    hs.sort_by(f64::total_cmp);
    let h_med = hs[hs.len() / 2];
    let h_max = hs[hs.len() - 1];
    let r = geo.radius;
    let mid = (n / 2) as i32;
    let c = geo.dir(geo.i0 + mid, geo.j0 + mid);
    let ej = (geo.dir(geo.i0 + mid, geo.j0 + mid + 1) - c).normalize();
    let d = n as f64 * geo.m_col;
    let g = (c - ej * (0.55 * d / r)).normalize();
    let eye = g * (r + h_med + 0.18 * d);
    let target = c * (r + h_med);
    let fwd = (target - eye).normalize();
    let right = fwd.cross(g).normalize();
    let up = right.cross(fwd).normalize();
    let tan_h = 27.5f64.to_radians().tan();
    let tan_v = tan_h * f64::from(TILE_OBLIQUE_H) / f64::from(TILE);
    let top = r + h_max + 1.0;
    let step = (n as f64 / 256.0).max(1.0) * geo.m_col;
    let height_at = |x: f64, y: f64| -> Option<(f64, usize)> {
        if x < 0.0 || y < 0.0 || x > (n - 1) as f64 || y > (n - 1) as f64 {
            return None;
        }
        let (a, b) = (x.floor() as usize, y.floor() as usize);
        let (a1, b1) = ((a + 1).min(n - 1), (b + 1).min(n - 1));
        let (fx, fy) = (x - a as f64, y - b as f64);
        let hh = |aa: usize, bb: usize| f64::from(src.height(bb * n + aa));
        let v = (1.0 - fx) * (1.0 - fy) * hh(a, b)
            + fx * (1.0 - fy) * hh(a1, b)
            + (1.0 - fx) * fy * hh(a, b1)
            + fx * fy * hh(a1, b1);
        let near = (y.round() as usize).min(n - 1) * n + (x.round() as usize).min(n - 1);
        v.is_finite().then_some((v, near))
    };
    for py in 0..TILE_OBLIQUE_H {
        for px in 0..TILE {
            let xn = (2.0 * (f64::from(px) + 0.5) / f64::from(TILE)) - 1.0;
            let yn = 1.0 - (2.0 * (f64::from(py) + 0.5) / f64::from(TILE_OBLIQUE_H));
            let dir = (fwd + right * (xn * tan_h) + up * (yn * tan_v)).normalize();
            // Where the ray enters the sphere of the highest drawn ground.
            let bq = eye.dot(dir);
            let cq = eye.dot(eye) - top * top;
            let disc = bq * bq - cq;
            let mut colour = SKY;
            if disc > 0.0 {
                let t0 = (-bq - disc.sqrt()).max(0.0);
                let t1 = -bq + disc.sqrt();
                let mut t = t0;
                let mut prev = t0;
                let mut seen = false;
                while t <= t1 {
                    let p = eye + dir * t;
                    match geo
                        .cols_of(p)
                        .and_then(|(x, y)| height_at(x, y).map(|h| (h, x, y)))
                    {
                        Some(((h, _), _, _)) if p.length() <= r + h => {
                            // Bisect between the last step over the ground and this one.
                            let (mut lo, mut hi) = (prev, t);
                            for _ in 0..10 {
                                let m = 0.5 * (lo + hi);
                                let pm = eye + dir * m;
                                let under = geo
                                    .cols_of(pm)
                                    .and_then(|(x, y)| height_at(x, y))
                                    .is_some_and(|(hm, _)| pm.length() <= r + hm);
                                if under {
                                    hi = m;
                                } else {
                                    lo = m;
                                }
                            }
                            let ph = eye + dir * hi;
                            colour = match geo.cols_of(ph).and_then(|(x, y)| height_at(x, y)) {
                                Some((_, k)) if src.wet[k] => WATER,
                                Some((_, k)) => grey(f64::from(shades[k])),
                                None => OUTSIDE,
                            };
                            seen = true;
                            break;
                        }
                        Some(_) => {}
                        None => {
                            colour = OUTSIDE;
                        }
                    }
                    prev = t;
                    t += step;
                }
                if !seen && colour == OUTSIDE {
                    colour = SKY;
                }
            }
            img.put_pixel(ox + px, oy + py, Rgb(colour));
        }
    }
}

/// ★ THE PANEL WRITER: the index of every file written and the bytes spent against the budget.
pub struct Panels {
    pub dir: Option<String>,
    pub budget: u64,
    pub spent: u64,
    pub files: Vec<(String, u64, String)>,
}

impl Panels {
    /// Saves an image; refused past the budget.
    fn save(&mut self, c: &mut Census, name: &str, img: &RgbImage, caption: &str) {
        let Some(dir) = self.dir.as_ref() else { return };
        let path = format!("{dir}/{name}");
        let mut bytes: Vec<u8> = Vec::new();
        let enc = image::codecs::png::PngEncoder::new_with_quality(
            &mut bytes,
            image::codecs::png::CompressionType::Best,
            image::codecs::png::FilterType::Adaptive,
        );
        if image::ImageEncoder::write_image(
            enc,
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .is_err()
        {
            c.say(format!("PANEL {name} REFUSED: the encoder failed"));
            return;
        }
        let size = bytes.len() as u64;
        if self.spent + size > self.budget {
            c.say(format!(
                "PANEL {name} REFUSED: {size} bytes would pass the budget ({} of {} spent)",
                self.spent, self.budget
            ));
            return;
        }
        let _ = std::fs::create_dir_all(dir);
        if std::fs::write(&path, &bytes).is_err() {
            c.say(format!(
                "PANEL {name} REFUSED: the file could not be written"
            ));
            return;
        }
        self.spent += size;
        c.say(format!("PANEL {name} {caption} bytes={size}"));
        self.files
            .push((name.to_string(), size, caption.to_string()));
    }

    /// ★ A STAND'S TWO COMPOSITES AT ONE RUNG: the top-down 2 × 2 and the oblique 2 × 2 of R0, N1,
    /// N2 and O1 (the second round's order: N2 beside the N1 it keeps the line of).
    #[allow(clippy::too_many_arguments, reason = "the panel's own words")]
    pub fn composites(
        &mut self,
        c: &mut Census,
        stand: &str,
        window: &str,
        out: &RungOut,
        geo: &Geo,
        m_i: f64,
        m_j: f64,
        tag: &str,
    ) {
        if self.dir.is_none() {
            return;
        }
        let order = [Var::R0, Var::N1, Var::N2, Var::O1];
        let srcs: Vec<TileSrc> = order.iter().map(|&v| TileSrc::of(out, v)).collect();
        let km = out.n as f64 * geo.m_col / 1_000.0;
        let mut top = RgbImage::new(2 * TILE, 2 * TILE);
        for (k, src) in srcs.iter().enumerate() {
            top_tile(
                &mut top,
                (k as u32 % 2) * TILE,
                (k as u32 / 2) * TILE,
                src,
                m_i,
                m_j,
            );
        }
        let caption = |view: &str| {
            format!(
                "stand={stand} rung={} view={view} tiles=[R0,N1/N2,O1] window={window}:{km:.1}km light=az315/alt45 \"no far shadow\"",
                out.rung
            )
        };
        self.save(
            c,
            &format!("{stand}_r{:02}{tag}_top_R0-N1-N2-O1.png", out.rung),
            &top,
            &caption("top"),
        );
        let mut obl = RgbImage::new(2 * TILE, 2 * TILE_OBLIQUE_H);
        for (k, src) in srcs.iter().enumerate() {
            oblique_tile(
                &mut obl,
                (k as u32 % 2) * TILE,
                (k as u32 / 2) * TILE_OBLIQUE_H,
                src,
                geo,
                m_i,
                m_j,
            );
        }
        self.save(
            c,
            &format!("{stand}_r{:02}{tag}_oblique_R0-N1-N2-O1.png", out.rung),
            &obl,
            &caption("oblique"),
        );
    }

    /// ★ A FAR TOP-DOWN ROW of tiles (the high lake's R0, R0w, R0b; the pocket's R0).
    #[allow(clippy::too_many_arguments, reason = "the panel's own words")]
    pub fn far_row(
        &mut self,
        c: &mut Census,
        name: &str,
        srcs: &[TileSrc],
        labels: &str,
        m_i: f64,
        m_j: f64,
        km: f64,
        rung: u8,
    ) {
        if self.dir.is_none() || srcs.is_empty() {
            return;
        }
        let mut img = RgbImage::new(TILE * srcs.len() as u32, TILE);
        for (k, src) in srcs.iter().enumerate() {
            top_tile(&mut img, k as u32 * TILE, 0, src, m_i, m_j);
        }
        self.save(
            c,
            name,
            &img,
            &format!(
                "rung={rung} view=top tiles=[{labels}] window=W3:{km:.1}km light=az315/alt45 \"no far shadow\""
            ),
        );
    }
}
