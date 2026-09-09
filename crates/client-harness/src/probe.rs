//! ★ THE PICTURE INSTRUMENT (the voxel foundation, slice 8p; ruling V14 D8-7) — the pure half of
//! the stamp, the probe and the ruler that every judged picture carries from this slice on.
//!
//! **The stamp** is the measured facts written on a picture and its state file: the eye's altitude
//! over the recipe's surface, the horizon, the radius drawn, the rungs and chunk counts, the star's
//! angles, the biome, the world identity, the tick. The formulas that turn a reading into a stated
//! number live HERE, so the renderer that writes the stamp and the gate that checks it (M8-4) share
//! one expression and cannot drift apart.
//!
//! **The probe** is a second picture aligned to the first in which every pixel says WHAT drew it and
//! HOW FAR it is. It is drawn by a material that writes a code instead of a colour, and read back
//! through the same copier as the picture. This module is the CODEC: the renderer's uniform is built
//! by [`probe_byte`], the gate reads pixels through [`decode_probe`], and the two round-trip in a
//! test. The colour classifier of slice 7 (warm paint = ground) is retired by it.
//!
//! ```text
//!   R byte:  kind << 5 | rung        kind 0 = nothing, 1 = terrain, 2 = the ruler; rung 0..31
//!   G, B:    distance from the eye in CELLS of that rung, big-endian u16 (0..65 535 cells)
//!   A:       255
//! ```
//!
//! A metre-cell rung reaches 65 km; a 512 m rung reaches 33 000 km. One cell is the ladder's own
//! unit, so the same two bytes serve every rung.
//!
//! **The ruler** is a ball of known size standing on the ground where the eye's centre ray meets it
//! (placed by `vd_client::chunks::ruler_on_surface`). The gate predicts its pixel radius from the
//! stamp's stated centre and radius through the SAME projection the marker gates use, and measures it
//! on the probe: equal within one pixel, or the picture's scale is not what the stamp says.
//!
//! **Example (MEASURED 2026-09-09).** The pilot stands on the home planet; the stamp says "3.4 m over
//! the recipe, horizon 6.57 km, drawn to 0.37 km at rung 0, star 15.0° up and 120.0° off the nose,
//! grassland". The probe says the lower half of the frame is terrain at rung 0, and a ruler ball of
//! radius 1.03 m at 29.2 m covers 30.55 px of radius — its projection through the pilot camera is
//! 30.65 px.

use vd_core::glam::DVec3;

/// The probe's kind codes: what drew a pixel.
pub const PROBE_KIND_NONE: u8 = 0;
pub const PROBE_KIND_TERRAIN: u8 = 1;
pub const PROBE_KIND_RULER: u8 = 2;
/// The rung's bits in the R byte (rungs 0..31) — the kind sits above them.
pub const PROBE_RUNG_BITS: u8 = 5;
const PROBE_RUNG_MASK: u8 = (1 << PROBE_RUNG_BITS) - 1;
/// The widest distance the two bytes state, in cells.
pub const PROBE_MAX_CELLS: u16 = u16::MAX;

/// The R byte for a kind and a rung: the renderer's uniform, the gate's expectation.
#[must_use]
pub const fn probe_byte(kind: u8, rung: u8) -> u8 {
    (kind << PROBE_RUNG_BITS) | (rung & PROBE_RUNG_MASK)
}

/// One decoded probe pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbePixel {
    pub kind: u8,
    pub rung: u8,
    /// The distance from the eye, in cells of `rung`.
    pub cells: u16,
}

/// The pixel a probe's RGB bytes state.
#[must_use]
pub const fn decode_probe(rgb: [u8; 3]) -> ProbePixel {
    ProbePixel {
        kind: rgb[0] >> PROBE_RUNG_BITS,
        rung: rgb[0] & PROBE_RUNG_MASK,
        cells: ((rgb[1] as u16) << 8) | rgb[2] as u16,
    }
}

/// The RGB bytes for a pixel — the shader's arithmetic, in Rust, for the round-trip test and for
/// a fixture that paints a probe by hand.
#[must_use]
pub const fn encode_probe(p: ProbePixel) -> [u8; 3] {
    [
        probe_byte(p.kind, p.rung),
        (p.cells >> 8) as u8,
        (p.cells & 0xff) as u8,
    ]
}

/// What one kind covers in a probe: the pixel count, the centroid, and the range of rungs and
/// distances its pixels state. A blob with no pixels has no centroid and an empty range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeBlob {
    pub count: u64,
    /// The centroid in pixels from the top-left, `None` for an empty blob.
    pub centroid: Option<(f64, f64)>,
    /// The smallest and largest rung the blob's pixels state (both 0 when empty).
    pub rung_min: u8,
    pub rung_max: u8,
    /// The nearest and farthest distance in cells (0 and 0 when empty).
    pub cells_min: u16,
    pub cells_max: u16,
}

/// Read every pixel of `kind` in an RGBA8 probe of `width` columns.
#[must_use]
pub fn probe_blob(rgba: &[u8], width: usize, kind: u8) -> ProbeBlob {
    let mut count = 0u64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut rung_min = u8::MAX;
    let mut rung_max = 0u8;
    let mut cells_min = u16::MAX;
    let mut cells_max = 0u16;
    let w = width.max(1);
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let p = decode_probe([px[0], px[1], px[2]]);
        let hit = p.kind == kind;
        // Branchless accumulation: a miss adds nothing and moves no bound.
        count += u64::from(hit);
        sum_x += f64::from(u8::from(hit)) * (i % w) as f64;
        sum_y += f64::from(u8::from(hit)) * (i / w) as f64;
        rung_min = if hit { rung_min.min(p.rung) } else { rung_min };
        rung_max = if hit { rung_max.max(p.rung) } else { rung_max };
        cells_min = if hit {
            cells_min.min(p.cells)
        } else {
            cells_min
        };
        cells_max = if hit {
            cells_max.max(p.cells)
        } else {
            cells_max
        };
    }
    let empty = count == 0;
    ProbeBlob {
        count,
        centroid: (!empty).then(|| (sum_x / count as f64, sum_y / count as f64)),
        rung_min: if empty { 0 } else { rung_min },
        rung_max,
        cells_min: if empty { 0 } else { cells_min },
        cells_max,
    }
}

/// The share of pixels of `kind` in the rows `[y0, y1)` of a probe of `width` × `height`.
#[must_use]
pub fn probe_share_in_rows(
    rgba: &[u8],
    width: usize,
    height: usize,
    y0: usize,
    y1: usize,
    kind: u8,
) -> f64 {
    let y1 = y1.min(height);
    let (mut hits, mut total) = (0u64, 0u64);
    let mut y = y0;
    while y < y1 {
        let row = &rgba[y * width * 4..(y + 1) * width * 4];
        for px in row.chunks_exact(4) {
            hits += u64::from(decode_probe([px[0], px[1], px[2]]).kind == kind);
            total += 1;
        }
        y += 1;
    }
    hits as f64 / total.max(1) as f64
}

/// A blob's equivalent radius in pixels: the radius of the disc with its pixel count.
#[must_use]
pub fn equivalent_radius_px(count: u64) -> f64 {
    (count as f64 / std::f64::consts::PI).sqrt()
}

/// THE HORIZON of a smooth sphere of `radius_m` seen from `altitude_m` over it, in metres along the
/// line of sight: `√(2Rh + h²)`. A height under the surface reads as zero.
#[must_use]
pub fn horizon_m(radius_m: f64, altitude_m: f64) -> f64 {
    let h = altitude_m.max(0.0);
    (2.0 * radius_m * h + h * h).sqrt()
}

/// THE HORIZON'S DIP below level, in radians: `acos(R / (R + h))`.
#[must_use]
pub fn horizon_dip_rad(radius_m: f64, altitude_m: f64) -> f64 {
    let h = altitude_m.max(0.0);
    (radius_m / (radius_m + h)).clamp(-1.0, 1.0).acos()
}

/// THE STAR'S ANGLES at a stand: its elevation over the local level (radians, positive up) and how
/// far its azimuth stands off the nose (radians, 0 dead ahead, π behind). `star` is the direction
/// to the star, `up` the local up, `forward` the camera's nose; each is normalised here. A star
/// straight overhead has no azimuth, and reads as π/2 off the nose.
#[must_use]
pub fn star_angles(star: DVec3, up: DVec3, forward: DVec3) -> (f64, f64) {
    let s = star.normalize_or_zero();
    let u = up.normalize_or_zero();
    let f = forward.normalize_or_zero();
    let elevation = s.dot(u).clamp(-1.0, 1.0).asin();
    let s_level = (s - u * s.dot(u)).normalize_or_zero();
    let f_level = (f - u * f.dot(u)).normalize_or_zero();
    let off_nose = s_level.dot(f_level).clamp(-1.0, 1.0).acos();
    (elevation, off_nose)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_codec_round_trips_every_kind_rung_and_distance_edge() {
        let cases = [
            ProbePixel {
                kind: PROBE_KIND_NONE,
                rung: 0,
                cells: 0,
            },
            ProbePixel {
                kind: PROBE_KIND_TERRAIN,
                rung: 13,
                cells: 1,
            },
            ProbePixel {
                kind: PROBE_KIND_RULER,
                rung: 31,
                cells: PROBE_MAX_CELLS,
            },
            ProbePixel {
                kind: 7,
                rung: 5,
                cells: 0x1234,
            },
        ];
        for c in cases {
            assert_eq!(decode_probe(encode_probe(c)), c, "{c:?}");
        }
        assert_eq!(probe_byte(PROBE_KIND_TERRAIN, 3), 0b0010_0011);
        // A rung past the five bits is masked, never carried into the kind.
        assert_eq!(probe_byte(PROBE_KIND_RULER, 0xff), 0b0101_1111);
        assert_eq!(
            encode_probe(ProbePixel {
                kind: PROBE_KIND_RULER,
                rung: 2,
                cells: 0x0102
            }),
            [0b0100_0010, 1, 2]
        );
    }

    /// A 4×3 probe: row 0 nothing, row 1 terrain at rung 2 (cells 10, 20, 30, 40), row 2 the ruler
    /// at rung 2 in the middle two columns (cells 7 and 9), nothing at the edges.
    fn probe() -> Vec<u8> {
        let mut px: Vec<[u8; 3]> = Vec::new();
        for _ in 0..4 {
            px.push([0, 0, 0]);
        }
        for cells in [10u16, 20, 30, 40] {
            px.push(encode_probe(ProbePixel {
                kind: PROBE_KIND_TERRAIN,
                rung: 2,
                cells,
            }));
        }
        px.push([0, 0, 0]);
        px.push(encode_probe(ProbePixel {
            kind: PROBE_KIND_RULER,
            rung: 2,
            cells: 7,
        }));
        px.push(encode_probe(ProbePixel {
            kind: PROBE_KIND_RULER,
            rung: 2,
            cells: 9,
        }));
        px.push([0, 0, 0]);
        px.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect()
    }

    #[test]
    fn a_blob_counts_its_pixels_and_states_their_centroid_and_ranges() {
        let rgba = probe();
        let terrain = probe_blob(&rgba, 4, PROBE_KIND_TERRAIN);
        assert_eq!(
            terrain,
            ProbeBlob {
                count: 4,
                centroid: Some((1.5, 1.0)),
                rung_min: 2,
                rung_max: 2,
                cells_min: 10,
                cells_max: 40,
            }
        );
        let ruler = probe_blob(&rgba, 4, PROBE_KIND_RULER);
        assert_eq!(ruler.count, 2);
        assert_eq!(ruler.centroid, Some((1.5, 2.0)));
        assert_eq!((ruler.cells_min, ruler.cells_max), (7, 9));
        let none = probe_blob(&rgba, 4, PROBE_KIND_NONE);
        assert_eq!(none.count, 6);
        let absent = probe_blob(&rgba, 4, 5);
        assert_eq!(
            absent,
            ProbeBlob {
                count: 0,
                centroid: None,
                rung_min: 0,
                rung_max: 0,
                cells_min: 0,
                cells_max: 0,
            }
        );
        // A zero width is read as one column, never a division by zero.
        assert_eq!(probe_blob(&rgba, 0, PROBE_KIND_RULER).count, 2);
    }

    #[test]
    fn the_share_reads_a_row_band_and_clamps_it_to_the_frame() {
        let rgba = probe();
        assert!((probe_share_in_rows(&rgba, 4, 3, 1, 2, PROBE_KIND_TERRAIN) - 1.0).abs() < 1e-12);
        assert!(
            (probe_share_in_rows(&rgba, 4, 3, 0, 3, PROBE_KIND_TERRAIN) - 4.0 / 12.0).abs() < 1e-12
        );
        // The band past the frame is clamped; a band of no rows is zero, not a panic.
        assert!((probe_share_in_rows(&rgba, 4, 3, 2, 9, PROBE_KIND_RULER) - 0.5).abs() < 1e-12);
        assert_eq!(
            probe_share_in_rows(&rgba, 4, 3, 3, 3, PROBE_KIND_RULER),
            0.0
        );
        assert!((equivalent_radius_px(314) - 10.0).abs() < 0.01);
    }

    #[test]
    fn the_horizon_is_root_two_r_h_and_the_dip_its_angle() {
        // Earth-like: 6 371 km, an eye 3.4 m up sees 6.58 km; from 60 km it sees 877 km.
        let r = 6_371_000.0;
        assert!((horizon_m(r, 3.4) - 6_582.0).abs() < 1.0);
        assert!((horizon_m(r, 60_000.0) - 876_425.0).abs() < 1.0);
        assert_eq!(horizon_m(r, -5.0), 0.0);
        assert!((horizon_dip_rad(r, 60_000.0).to_degrees() - 7.83).abs() < 0.02);
        assert_eq!(horizon_dip_rad(r, 0.0), 0.0);
        assert_eq!(horizon_dip_rad(r, -1.0), 0.0);
    }

    #[test]
    fn the_star_angles_read_elevation_and_the_azimuth_off_the_nose() {
        let up = DVec3::Y;
        let nose = DVec3::NEG_Z;
        // A star 30° up, 120° round from the nose.
        let e = 30.0_f64.to_radians();
        let a = 120.0_f64.to_radians();
        let level = DVec3::new(-a.sin(), 0.0, -a.cos()); // the nose turned 120° about +Y
        let star = level * e.cos() + up * e.sin();
        let (elev, off) = star_angles(star * 5.0, up * 2.0, nose * 3.0);
        assert!((elev.to_degrees() - 30.0).abs() < 1e-9);
        assert!((off.to_degrees() - 120.0).abs() < 1e-9);
        // Dead ahead and level; straight behind; straight overhead reads π/2 off the nose.
        assert_eq!(star_angles(nose, up, nose), (0.0, 0.0));
        let (_, behind) = star_angles(-nose, up, nose);
        assert!((behind - std::f64::consts::PI).abs() < 1e-12);
        let (over, off_over) = star_angles(up, up, nose);
        assert!((over - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((off_over - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // A zero star reads level and π/2 off — degraded, never a NaN.
        let (z_e, z_o) = star_angles(DVec3::ZERO, up, nose);
        assert_eq!(z_e, 0.0);
        assert!((z_o - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }
}
