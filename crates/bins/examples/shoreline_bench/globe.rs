//! ★ THE COVERAGE GLOBES (slice K0c; the far-water design §5.4, the owner's look for Q1): the far
//! page of the whole planet, and pictures of the globe at W18's seven altitudes drawn two ways —
//! the skin (every pixel the share of N2's samples under its footprint, through the one filter),
//! and today's box fold (every pixel the water or the land of the drawn rung's column).
//!
//! The page is built one face at a time from each face's own sample pyramid (quarter squares up to
//! the top rung), so the whole globe is on it. The pictures are flat colour, with no light and no
//! air: the water srgb(0.06, 0.24, 0.42), the land srgb(0.55, 0.50, 0.42), mixed by the share.
//!
//! **Example.** At 52 000 km the steep coast's side of the planet fills 600 pixels. Under the skin
//! its coasts are soft edges a pixel wide and its small lakes pale tints; under the box fold every
//! pixel is blue or tan, and a lake's ragged edge shows, hides and shows again as the eye climbs
//! through the rings.

use image::{Rgb, RgbImage};
use vd_core::glam::DVec3;
use vd_seed::bend::{Face, direction};
use vd_terrain::BodyDefinition;

use crate::census::Census;
use crate::panels::Panels;
use crate::pyramid::Pyramid;
use crate::sampler::{Eye, Src, sample};
use crate::skin::{Level, R8Level, SkinPyramid, r8};

/// W18's seven altitudes (the owner's climb over the berth, and the two far stands), metres.
pub const ALTITUDES_M: [f64; 7] = [
    6_374.0e3,
    11_497.0e3,
    16_952.0e3,
    52_000.0e3,
    68_000.0e3,
    118_000.0e3,
    408_000.0e3,
];
/// The pictures' side, in pixels (the reference view's 720 rows).
pub const SIDE_PX: u32 = 720;

/// ★ THE FAR PAGE OF ONE FACE: its levels from the quarter square up, as counts and as 8-bit.
pub struct FacePage {
    pub face: Face,
    pub counts: Vec<Level>,
    pub r8: Vec<R8Level>,
}

/// ★ THE WHOLE PLANET'S FAR PAGE: each face's sample pyramid over the whole face, kept from the
/// quarter square up (the cells' classes are dropped face by face).
pub fn build_page(
    c: &mut Census,
    body: &BodyDefinition,
    nodes: &vd_bins::shore_rules::Nodes,
    ch: &vd_bins::shore_rules::NamedChoices,
    f_rung: u8,
    threads: usize,
) -> Vec<FacePage> {
    let fine = body.macro_lattice().expect("a lattice");
    let edge = fine.edge as i32;
    let mut out = Vec::new();
    for face in Face::ALL {
        let t = std::time::Instant::now();
        let (pyr, built) = Pyramid::build(
            &fine,
            nodes,
            face,
            f_rung,
            (0, 0),
            2 * edge,
            ch,
            threads,
            usize::MAX,
        );
        let base = Level {
            level: pyr.q_rung,
            i0: 0,
            j0: 0,
            n: i64::from(pyr.nq),
            wet: pyr.quarters.iter().map(|q| q.wet).collect(),
            lake: pyr.quarters.iter().map(|q| q.lake).collect(),
        };
        let top = f_rung + 11;
        let skin = SkinPyramid::from_base(face, base, top);
        let r8s: Vec<R8Level> = skin
            .levels
            .iter()
            .map(|l| R8Level {
                level: l.level,
                i0: l.i0,
                j0: l.j0,
                n: l.n,
                texels: l
                    .wet
                    .iter()
                    .map(|&w| r8(u64::from(w), l.per_texel()))
                    .collect(),
            })
            .collect();
        c.say(format!(
            "   the far page, face {face:?}: {} node cells ({} mixed) keyed in {:.1} s; levels {}..={}",
            built.cells,
            built.mixed,
            t.elapsed().as_secs_f64(),
            skin.levels.first().map_or(0, |l| l.level),
            skin.levels.last().map_or(0, |l| l.level)
        ));
        out.push(FacePage {
            face,
            counts: skin.levels,
            r8: r8s,
        });
    }
    out
}

/// The globes' flat colours: the water srgb(0.06, 0.24, 0.42), the land srgb(0.55, 0.50, 0.42).
const WATER: [f64; 3] = [0.06, 0.24, 0.42];
const LAND: [f64; 3] = [0.55, 0.50, 0.42];

/// The colour of a water share `c` (0..=1).
fn mix(c: f64) -> Rgb<u8> {
    let px = |k: usize| ((LAND[k] * (1.0 - c) + WATER[k] * c) * 255.0).round() as u8;
    Rgb([px(0), px(1), px(2)])
}

/// ★ ONE PIXEL two ways: the skin's share (the far page through the one filter) and today's box
/// fold (the drawn rung's column, water or land); `None` off the globe.
fn pixel_pair(
    body: &BodyDefinition,
    page: &[FacePage],
    eye: &Eye,
    sea_r: f64,
    px: f64,
    py: f64,
) -> Option<(f64, bool)> {
    let hit = eye.footprint(px, py, sea_r)?;
    let fp = page.iter().find(|p| p.face == hit.face)?;
    let lo = fp.r8.first().map_or(12, |l| l.level);
    let hi = fp.r8.last().map_or(lo, |p| p.level);
    let n_l = f64::from(body.ladder().cells_per_edge(lo));
    let (pos, dx, dy) = hit.in_texels(n_l);
    let fetch = |l: u8, i: i64, j: i64| -> u8 {
        fp.r8
            .iter()
            .find(|p| p.level == l)
            .map_or(0, |p| p.clamped(i, j))
    };
    let s = Src {
        lo,
        hi,
        fetch: &fetch,
    };
    let cov = sample(&s, pos, dx, dy, 0.0, false) / 255.0;
    // Today's box fold: the drawn rung's column, water or land.
    let mut l = lo;
    while l < hi && hit.d >= 1.1 * vd_client::ladder_view::switch_m(l) {
        l += 1;
    }
    let n_r = f64::from(body.ladder().cells_per_edge(l));
    let (ci, cj) = (
        ((hit.a + 1.0) * 0.5 * n_r) as i64,
        ((hit.b + 1.0) * 0.5 * n_r) as i64,
    );
    let wet = fp
        .counts
        .iter()
        .find(|lv| lv.level == l)
        .and_then(|lv| {
            lv.get(ci, cj)
                .map(|(w, _)| 2 * u64::from(w) >= lv.per_texel() && w > 0)
        })
        .unwrap_or(false);
    Some((cov, wet))
}

/// ★ THE GLOBES: for each of W18's altitudes over `at` (a direction), the skin's picture and
/// today's box fold's picture, side by side in one panel.
#[allow(clippy::too_many_arguments, reason = "the panel's own words")]
pub fn globes(
    c: &mut Census,
    panels: &mut Panels,
    body: &BodyDefinition,
    page: &[FacePage],
    sea_r: f64,
    at: DVec3,
    name: &str,
) {
    let alpha = vd_client::ladder_view::pixel_rad();
    for &h in &ALTITUDES_M {
        let eye = Eye::over(at.normalize() * sea_r, h, alpha);
        let mut img = RgbImage::new(2 * SIDE_PX + 8, SIDE_PX);
        let half = f64::from(SIDE_PX) / 2.0;
        let (mut soft, mut drawn_px) = (0usize, 0usize);
        for y in 0..SIDE_PX {
            for x in 0..SIDE_PX {
                let (px, py) = (f64::from(x) - half + 0.5, half - f64::from(y) - 0.5);
                let Some((cov, wet)) = pixel_pair(body, page, &eye, sea_r, px, py) else {
                    continue;
                };
                img.put_pixel(x, y, mix(cov));
                drawn_px += 1;
                soft += usize::from(cov > 0.02 && cov < 0.98);
                img.put_pixel(SIDE_PX + 8 + x, y, mix(if wet { 1.0 } else { 0.0 }));
            }
        }
        let file = format!("{name}_globe_{:06.0}km_skin-box.png", h / 1_000.0);
        panels.save(
            c,
            &file,
            &img,
            &format!(
                "left: the skin (the share under each pixel's footprint); right: today's box fold (the drawn rung's column); {:.0} km over the stand; {drawn_px} pixels on the globe, {soft} of them soft (2-98 % water); flat colour, no light, no air, no far shadow",
                h / 1_000.0
            ),
        );
    }
}

/// The climb strip's frame side, pixels, and its enlargement.
const STRIP_PX: u32 = 160;
const STRIP_ZOOM: u32 = 2;

/// ★ THE CLIMB STRIP (the design §5.4 item 3): the eye straight over `at` at each altitude of
/// `frames_km`, a square of pixels about the nadir; the top row the skin, the bottom row today's
/// box fold, one frame a column, left to right in the list's order.
#[allow(clippy::too_many_arguments, reason = "the panel's own words")]
pub fn climb_strip(
    c: &mut Census,
    panels: &mut Panels,
    body: &BodyDefinition,
    page: &[FacePage],
    sea_r: f64,
    at: DVec3,
    name: &str,
    frames_km: &[f64],
) {
    let alpha = vd_client::ladder_view::pixel_rad();
    let side = STRIP_PX * STRIP_ZOOM;
    let gap = 6;
    let n = frames_km.len() as u32;
    let mut img = RgbImage::from_pixel(n * (side + gap), 2 * side + gap, Rgb([255, 255, 255]));
    let half = f64::from(STRIP_PX) / 2.0;
    for (k, &h_km) in frames_km.iter().enumerate() {
        let eye = Eye::over(at.normalize() * sea_r, h_km * 1e3, alpha);
        let x0 = k as u32 * (side + gap);
        for y in 0..STRIP_PX {
            for x in 0..STRIP_PX {
                let (px, py) = (f64::from(x) - half + 0.5, half - f64::from(y) - 0.5);
                let (top, bottom) = match pixel_pair(body, page, &eye, sea_r, px, py) {
                    Some((cov, wet)) => (mix(cov), mix(if wet { 1.0 } else { 0.0 })),
                    None => (Rgb([0, 0, 0]), Rgb([0, 0, 0])),
                };
                for (dy, dx) in (0..STRIP_ZOOM).flat_map(|a| (0..STRIP_ZOOM).map(move |b| (a, b))) {
                    img.put_pixel(x0 + x * STRIP_ZOOM + dx, y * STRIP_ZOOM + dy, top);
                    img.put_pixel(
                        x0 + x * STRIP_ZOOM + dx,
                        side + gap + y * STRIP_ZOOM + dy,
                        bottom,
                    );
                }
            }
        }
    }
    let list = frames_km
        .iter()
        .map(|h| {
            let mut l = 0u8;
            while l < 18 && h * 1e3 >= 1.1 * vd_client::ladder_view::switch_m(l) {
                l += 1;
            }
            format!("{h:.0} km (rung {l})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    panels.save(
        c,
        &format!("{name}_climb_strip.png"),
        &img,
        &format!(
            "the climb over {name}: frames left to right at {list}; top row the skin, bottom row today's box fold; {STRIP_PX} pixels a side about the nadir, each pixel drawn {STRIP_ZOOM}x{STRIP_ZOOM}; flat colour, no light, no air, no far shadow"
        ),
    );
}

/// A face's centre point in a face's own texel units, as a direction (for a stand's centre).
#[must_use]
pub fn direction_of(face: Face, a: f64, b: f64) -> DVec3 {
    let d = direction(face, a, b);
    DVec3::new(d[0], d[1], d[2])
}

/// ★ THE RAW PAGE of each face at one level: one texel a pixel, the share as grey — no view, no
/// filter — so a pattern in the globes can be told apart as the data's or the drawing's.
pub fn raw_faces(c: &mut Census, panels: &mut Panels, page: &[FacePage], level: u8) {
    for fp in page {
        let Some(lv) = fp.r8.iter().find(|l| l.level == level) else {
            continue;
        };
        let n = lv.n as u32;
        let mut img = RgbImage::new(n, n);
        for j in 0..n {
            for i in 0..n {
                let v = lv.texels[(j as usize) * lv.n as usize + i as usize];
                // Row 0 at the image's bottom, so +v points up.
                img.put_pixel(i, n - 1 - j, Rgb([v, v, v]));
            }
        }
        panels.save(
            c,
            &format!("raw_face_{:?}_level{level}.png", fp.face),
            &img,
            &format!("the far page of face {:?} at level {level}: one texel a pixel, white = all water; no view, no filter", fp.face),
        );
    }
}
