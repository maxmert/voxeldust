//! ★ THE PICTURE CENSUS (the speed step, slice T0 item 2; gate G-PICTURE-HOLES) — the pure half.
//!
//! The census finds the holes of the PICTURE CAMERA. A hole is a pixel where the probe draws
//! ground but no geometry writes a colour in the picture. The probe's own census
//! ([`crate::probe::ground_holes`]) cannot find these holes. The probe is a second camera: it
//! draws its own meshes with its own material, with no prepass and no multisampling (the
//! design's F31). So a hole that the picture camera's own depth test makes shows only in the
//! picture. Route G3 of the design can make such holes: the main pass tests its depth against
//! the depth of a second draw, the prepass.
//!
//! **The run.** The caller clears the picture to a SENTINEL colour, a colour that the ground
//! never shows ([`SENTINEL_RGB8`], pure cyan). The caller also turns off the air pass and the star
//! cloud. Then no pass writes over the clear colour where no geometry is. A pixel that keeps the
//! sentinel is a pixel that no geometry wrote. The caller states the sentinel as the picture
//! STORES it: the tonemapping pass of a camera with HDR changes the clear colour too.
//!
//! **A hole** is a pixel with three properties:
//! - the probe calls the pixel ground;
//! - the probe calls all four neighbours of the pixel ground too;
//! - the picture shows the sentinel at the pixel, within [`SENTINEL_TOLERANCE`] on each channel.
//!
//! The second property keeps the silhouette's edge out of the count. The two cameras draw their
//! own meshes, so at an edge they can disagree by one pixel. A pixel on the frame's border has
//! fewer than four neighbours, so the census never counts it.
//!
//! **The reading** ([`CensusRead`]) gives the hole count, the ground count, the 4-connected blobs
//! of holes (their count, the largest blob, the boxes of the first blobs), the first hole
//! positions, and the judged pixels and the holes by the probe's rung. Three more counts make the
//! census honest:
//! - the judged pixels: a census that judges no pixel reads zero and proves nothing;
//! - the sky pixels that show the sentinel: if no sky pixel shows it, the stated sentinel is not
//!   the stored clear colour, and a zero hole count is empty;
//! - the edge pixels that show the sentinel: the census does not count them, and it does not
//!   hide them.
//!
//! **The control** (the design's): one chunk that the prepass draws one depth step nearer than
//! the main pass must count holes. The unit tests hold its synthetic form: a ground region whose
//! picture pixels keep the sentinel counts exactly that region. The flight's form moves a whole
//! rung, and the holes by rung show that its holes stand on that rung.
//!
//! Example: the pilot stands on foot on the home planet and looks at the horizon. The census run
//! clears the picture to cyan. The prepass draws the rung-9 chunks at the horizon one depth step
//! nearer than the main pass does. The main pass's depth test refuses the pixels of those chunks,
//! so they keep the cyan. The probe draws the chunks as ground. The census reads these pixels as
//! holes of rung 9. The far hills against the cyan sky give no hole, because each pixel on their
//! silhouette has a sky neighbour in the probe.

use crate::probe::{PROBE_KIND_NONE, PROBE_KIND_TERRAIN, PROBE_RUNG_BITS, decode_probe};

/// ★ THE SENTINEL: pure cyan, as the picture stores it. The renderer clears the picture to it
/// under the census's switch (`VD_CLEAR_SENTINEL=1`), and a census gate states it back to
/// [`picture_census`]: one colour, named once. A census run turns the air pass off, and a camera
/// without the air has no HDR and no tonemapping pass, so it stores its clear colour exactly.
///
/// The basis: no pixel that geometry writes on a census stand takes it.
/// - The ground's paint is one colour, sRGB 0.55 / 0.50 / 0.42. Red is its strongest channel, and
///   a star's light (a black body at any temperature) always carries red. So the lit ground never
///   stores red at the floor while it stores green and blue at the top.
/// - The ruler is red, and the overlay draws the engine's default grey text on a dark panel.
/// - The harness's FAILURE colour, [`crate::assert::MAGENTA`] (a draw without a real material),
///   stays free. So a census run keeps its "zero magenta" check, and a failed draw on the ground
///   never reads as a hole.
///
/// Example: on the home planet's dusk stand the census run shows a cyan sky over the lit ground.
/// A chunk that the main pass refused shows as a cyan patch on the ground.
pub const SENTINEL_RGB8: [u8; 3] = [0, 255, 255];

/// How many rungs the probe can state: its rung field is [`PROBE_RUNG_BITS`] wide. The tables of
/// the reading by rung have one entry for each.
pub const PROBE_RUNGS: usize = 1 << PROBE_RUNG_BITS;

/// THE SENTINEL'S TOLERANCE, in levels on each channel. A pixel shows the sentinel when no
/// channel differs from the sentinel by more than this.
///
/// The basis is the engine's deband dither (Bevy 0.18.1, `screen_space_dither` in
/// `tonemapping_shared.wgsl`). The tonemapping pass of a camera with HDR adds the dither to every
/// pixel, and so to the clear colour too. The dither is at most half a level on the 2.2 curve.
/// The sRGB store makes that step at most 1.23 times larger, so the dither moves a stored channel
/// by less than one level. (A simulation of the shader's arithmetic over the whole range, run
/// 2026-09-29, measured a move of one level at most.) Thus each stored channel of the sentinel is
/// within one level of its true value, and two pixels that keep the sentinel are within two
/// levels of each other. So the caller can state the sentinel from any one pixel that keeps it. A
/// camera without HDR does not run the tonemapping pass, and it stores the clear colour exactly.
pub const SENTINEL_TOLERANCE: u8 = 2;

/// How many hole positions the reading keeps: the first holes in scan order (row by row, from the
/// top-left). Sixteen positions let a reader find the holes in the saved picture. The hole count
/// and the blobs stay whole.
pub const FIRST_HOLES_KEPT: usize = 16;

/// How many blob boxes the reading keeps: the first blobs in scan order (by the first pixel of
/// each blob). The blob count and the largest blob stay whole.
pub const FIRST_BLOBS_KEPT: usize = 16;

/// One frame of a census run: its picture and its probe. Each is RGBA8, row-major, 4 bytes a
/// pixel, of the one stated size.
pub struct CensusFrame<'a> {
    /// The picture camera's image.
    pub rgba: &'a [u8],
    /// The probe's image (the codec is in [`crate::probe`]).
    pub probe: &'a [u8],
    /// The width of both images, in pixels.
    pub width: usize,
    /// The height of both images, in pixels.
    pub height: usize,
}

impl CensusFrame<'_> {
    /// Whether the bytes are one picture and one probe of the stated size. A size that overflows
    /// saturates, and no slice has that length.
    fn well_formed(&self) -> bool {
        let n = self.width.saturating_mul(self.height).saturating_mul(4);
        (n > 0) & (self.rgba.len() == n) & (self.probe.len() == n)
    }
}

/// A 4-connected blob of holes: its pixel count and its bounding box. The box is inclusive, in
/// pixels from the top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoleBlob {
    /// The hole pixels of the blob: its area.
    pub pixels: u64,
    /// The leftmost column.
    pub left: usize,
    /// The rightmost column.
    pub right: usize,
    /// The top row.
    pub top: usize,
    /// The bottom row.
    pub bottom: usize,
}

/// What the census reads on one frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CensusRead {
    /// The pixels that the probe calls ground.
    pub ground: u64,
    /// The ground pixels that the census judges: off the frame's border, with all four neighbours
    /// ground in the probe.
    pub interior: u64,
    /// THE HOLES: the judged pixels that show the sentinel in the picture. G-PICTURE-HOLES reads
    /// this number (its target is zero).
    pub holes: u64,
    /// The ground pixels that show the sentinel but that the census does not judge: pixels on a
    /// silhouette's edge or on the frame's border. They are not holes. The count shows how well
    /// the picture and the probe agree at the edges.
    pub edge_sentinel: u64,
    /// The pixels that the probe calls nothing (the sky, in a census run).
    pub sky: u64,
    /// The sky pixels that show the sentinel: the sentinel's own control. With the air and the
    /// stars off, the sky keeps the sentinel. Zero here, with sky in the frame, means that the
    /// stated sentinel is not the stored clear colour.
    pub sky_sentinel: u64,
    /// The count of 4-connected blobs of holes.
    pub blobs: u64,
    /// The largest blob by pixels (on a tie, the first in scan order). `None` when no hole is.
    pub largest: Option<HoleBlob>,
    /// The first [`FIRST_BLOBS_KEPT`] blobs in scan order.
    pub first_blobs: Vec<HoleBlob>,
    /// The first [`FIRST_HOLES_KEPT`] holes in scan order, as `(column, row)`.
    pub first_holes: Vec<(usize, usize)>,
    /// The judged pixels by the probe's rung (the index is the rung): where the census can see a
    /// hole on each rung. They add up to `interior`.
    pub interior_by_rung: [u64; PROBE_RUNGS],
    /// THE HOLES by the probe's rung. They add up to `holes`. The control's holes stand on the
    /// rung that it moved, and a run's holes name the rung that made them.
    pub holes_by_rung: [u64; PROBE_RUNGS],
}

/// Whether a picture pixel's RGB shows the sentinel: no channel differs from the sentinel by more
/// than [`SENTINEL_TOLERANCE`]. The alpha takes no part.
#[must_use]
pub fn shows_sentinel(rgb: [u8; 3], sentinel: [u8; 3]) -> bool {
    (rgb[0].abs_diff(sentinel[0]) <= SENTINEL_TOLERANCE)
        & (rgb[1].abs_diff(sentinel[1]) <= SENTINEL_TOLERANCE)
        & (rgb[2].abs_diff(sentinel[2]) <= SENTINEL_TOLERANCE)
}

/// THE CENSUS of one frame against the stated `sentinel` (the RGB that the picture stores for its
/// clear colour). It returns `None` when the bytes are not one picture and one probe of the
/// stated size.
#[must_use]
pub fn picture_census(frame: &CensusFrame<'_>, sentinel: [u8; 3]) -> Option<CensusRead> {
    if !frame.well_formed() {
        return None;
    }
    let (w, h) = (frame.width, frame.height);
    // Each probe pixel's kind and rung. The rung is the field's own bits, so it always indexes the
    // tables by rung.
    let kinds: Vec<(u8, usize)> = frame
        .probe
        .chunks_exact(4)
        .map(|p| {
            let pixel = decode_probe([p[0], p[1], p[2]]);
            (pixel.kind, usize::from(pixel.rung))
        })
        .collect();
    let ground = |x: usize, y: usize| kinds[y * w + x].0 == PROBE_KIND_TERRAIN;
    let mut hole = vec![false; w * h];
    let mut read = CensusRead::default();
    let mut y = 0usize;
    while y < h {
        let mut x = 0usize;
        while x < w {
            let i = y * w + x;
            let px = &frame.rgba[i * 4..i * 4 + 3];
            let shows = shows_sentinel([px[0], px[1], px[2]], sentinel);
            let is_ground = ground(x, y);
            let (kind, rung) = kinds[i];
            let is_sky = kind == PROBE_KIND_NONE;
            // On the frame's border the clamped reads below stay in the frame, and `inside`
            // discards them.
            let inside = (x > 0) & (y > 0) & (x + 1 < w) & (y + 1 < h);
            let interior = is_ground
                & inside
                & ground(x.saturating_sub(1), y)
                & ground((x + 1).min(w - 1), y)
                & ground(x, y.saturating_sub(1))
                & ground(x, (y + 1).min(h - 1));
            let is_hole = interior & shows;
            read.ground += u64::from(is_ground);
            read.interior += u64::from(interior);
            read.holes += u64::from(is_hole);
            read.edge_sentinel += u64::from(is_ground & !interior & shows);
            read.sky += u64::from(is_sky);
            read.sky_sentinel += u64::from(is_sky & shows);
            read.interior_by_rung[rung] += u64::from(interior);
            read.holes_by_rung[rung] += u64::from(is_hole);
            hole[i] = is_hole;
            if is_hole & (read.first_holes.len() < FIRST_HOLES_KEPT) {
                read.first_holes.push((x, y));
            }
            x += 1;
        }
        y += 1;
    }
    let (blobs, largest, first_blobs) = blobs_of(hole, w);
    read.blobs = blobs;
    read.largest = largest;
    read.first_blobs = first_blobs;
    Some(read)
}

/// The 4-connected blobs of a hole mask of `w` columns, found in scan order: the count, the
/// largest blob and the first [`FIRST_BLOBS_KEPT`] blobs. The walk clears each hole that it
/// visits.
///
/// The census judges no pixel on the frame's border, so the four neighbours of each hole lie in
/// the mask, and on the hole's own row or the rows next to it. The walk therefore needs no bounds
/// test. The scan meets each blob first at its top row, so the top row of a blob is the row of its
/// first pixel.
fn blobs_of(mut hole: Vec<bool>, w: usize) -> (u64, Option<HoleBlob>, Vec<HoleBlob>) {
    let mut blobs = 0u64;
    let mut largest: Option<HoleBlob> = None;
    let mut largest_pixels = 0u64;
    let mut first: Vec<HoleBlob> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut start = 0usize;
    while start < hole.len() {
        if hole[start] {
            hole[start] = false;
            stack.push(start);
            let (x0, y0) = (start % w, start / w);
            let mut blob = HoleBlob {
                pixels: 0,
                left: x0,
                right: x0,
                top: y0,
                bottom: y0,
            };
            while let Some(i) = stack.pop() {
                let (x, y) = (i % w, i / w);
                blob.pixels += 1;
                blob.left = blob.left.min(x);
                blob.right = blob.right.max(x);
                blob.bottom = blob.bottom.max(y);
                for j in [i - 1, i + 1, i - w, i + w] {
                    if hole[j] {
                        hole[j] = false;
                        stack.push(j);
                    }
                }
            }
            blobs += 1;
            if blob.pixels > largest_pixels {
                largest_pixels = blob.pixels;
                largest = Some(blob);
            }
            if first.len() < FIRST_BLOBS_KEPT {
                first.push(blob);
            }
        }
        start += 1;
    }
    (blobs, largest, first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{PROBE_KIND_RULER, ProbePixel, encode_probe};

    /// The sentinel of these tests: the census's own, as the picture stores it.
    const SENTINEL: [u8; 3] = SENTINEL_RGB8;
    /// A ground colour: grass in the light of the star.
    const GRASS: [u8; 3] = [96, 128, 64];
    /// The rung of the scene's ground.
    const RUNG: usize = 9;
    /// The probe's codes: nothing (its clear colour), ground at rung 9, and the ruler.
    const NOTHING: [u8; 3] = [0, 0, 0];
    const GROUND: [u8; 3] = encode_probe(ProbePixel {
        kind: PROBE_KIND_TERRAIN,
        rung: RUNG as u8,
        cells: 300,
    });
    /// Ground one rung coarser, for the reading by rung.
    const GROUND_COARSER: [u8; 3] = encode_probe(ProbePixel {
        kind: PROBE_KIND_TERRAIN,
        rung: RUNG as u8 + 1,
        cells: 150,
    });

    /// A table by rung with `n` at `rung` and zero elsewhere.
    fn at_rung(rung: usize, n: u64) -> [u64; PROBE_RUNGS] {
        let mut table = [0; PROBE_RUNGS];
        table[rung] = n;
        table
    }
    const RULER: [u8; 3] = encode_probe(ProbePixel {
        kind: PROBE_KIND_RULER,
        rung: 0,
        cells: 30,
    });

    /// A census scene of `w × h`: the top `sky_rows` rows are sky (the probe draws nothing, the
    /// picture keeps the sentinel), and the rows under them are ground (the probe draws terrain,
    /// the picture shows grass).
    fn scene(w: usize, h: usize, sky_rows: usize) -> (Vec<u8>, Vec<u8>) {
        let mut picture = Vec::with_capacity(w * h * 4);
        let mut probe = Vec::with_capacity(w * h * 4);
        for i in 0..w * h {
            let (rgb, code) = if i / w < sky_rows {
                (SENTINEL, NOTHING)
            } else {
                (GRASS, GROUND)
            };
            picture.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            probe.extend_from_slice(&[code[0], code[1], code[2], 255]);
        }
        (picture, probe)
    }

    /// Write three bytes (a colour or a probe code) at pixel `(x, y)` of an RGBA8 image `w` wide.
    fn put(image: &mut [u8], w: usize, x: usize, y: usize, rgb: [u8; 3]) {
        let i = (y * w + x) * 4;
        image[i..i + 3].copy_from_slice(&rgb);
    }

    /// The census of a well-formed frame against the census's sentinel.
    fn census(picture: &[u8], probe: &[u8], w: usize, h: usize) -> CensusRead {
        let frame = CensusFrame {
            rgba: picture,
            probe,
            width: w,
            height: h,
        };
        picture_census(&frame, SENTINEL).expect("one picture and one probe of the stated size")
    }

    /// THE CONTROL, in its synthetic form. The prepass drew one chunk one depth step nearer, so
    /// the main pass refused the chunk's pixels: they keep the sentinel, while the probe draws the
    /// chunk as ground. The census counts exactly that region, as one blob, and the dither's two
    /// levels stay inside the tolerance. The same frame before the defect counts no hole.
    #[test]
    fn the_control_a_refused_chunk_counts_exactly_its_region() {
        let (w, h) = (12, 10);
        let (mut picture, probe) = scene(w, h, 3);
        let before = census(&picture, &probe, w, h);
        assert_eq!((before.holes, before.blobs, before.largest), (0, 0, None));
        // The refused chunk: columns 4..=7, rows 5..=7. Two of its pixels carry the dither.
        for y in 5..=7 {
            for x in 4..=7 {
                put(&mut picture, w, x, y, SENTINEL);
            }
        }
        put(&mut picture, w, 4, 5, [2, 253, 255]);
        put(&mut picture, w, 7, 7, [0, 255, 253]);
        let chunk = HoleBlob {
            pixels: 12,
            left: 4,
            right: 7,
            top: 5,
            bottom: 7,
        };
        assert_eq!(
            census(&picture, &probe, w, h),
            CensusRead {
                // Rows 3..=9 of 12 columns.
                ground: 84,
                // Rows 4..=8 (row 3 touches the sky, row 9 is the border) × columns 1..=10.
                interior: 50,
                holes: 12,
                edge_sentinel: 0,
                sky: 36,
                sky_sentinel: 36,
                blobs: 1,
                largest: Some(chunk),
                first_blobs: vec![chunk],
                first_holes: vec![
                    (4, 5),
                    (5, 5),
                    (6, 5),
                    (7, 5),
                    (4, 6),
                    (5, 6),
                    (6, 6),
                    (7, 6),
                    (4, 7),
                    (5, 7),
                    (6, 7),
                    (7, 7),
                ],
                interior_by_rung: at_rung(RUNG, 50),
                holes_by_rung: at_rung(RUNG, 12),
            }
        );
        // One level past the tolerance is another colour: the region less that pixel, still one
        // blob with the same box.
        put(&mut picture, w, 5, 6, [3, 255, 255]);
        let read = census(&picture, &probe, w, h);
        assert_eq!((read.holes, read.blobs), (11, 1));
        assert_eq!(
            read.largest,
            Some(HoleBlob {
                pixels: 11,
                ..chunk
            })
        );
    }

    /// THE EDGE IS NOT A HOLE. A ground pixel that keeps the sentinel is not counted when a
    /// neighbour is sky (the limb of the planet from orbit, the far hills), the ruler, or a hole
    /// of the probe itself, or when it lies on the frame's border. The census states these pixels
    /// apart. The same sentinel one row into the ground is a hole.
    #[test]
    fn a_sentinel_on_an_edge_or_the_border_is_not_a_hole() {
        let (w, h) = (12, 10);
        let (mut picture, mut probe) = scene(w, h, 3);
        // A hole of the probe at (8, 6), where the picture draws grass; the ruler at (2, 7).
        put(&mut probe, w, 8, 6, NOTHING);
        put(&mut probe, w, 2, 7, RULER);
        // The sentinel on the silhouette (a sky neighbour above), on the left, right and bottom
        // borders, over the probe's hole, and beside the ruler.
        for (x, y) in [(5, 3), (0, 6), (11, 6), (6, 9), (8, 5), (3, 7)] {
            put(&mut picture, w, x, y, SENTINEL);
        }
        let edges = census(&picture, &probe, w, h);
        assert_eq!(
            edges,
            CensusRead {
                // 84 less the probe's hole and the ruler.
                ground: 82,
                // 50 less the probe's hole and the ruler, each with its four neighbours.
                interior: 40,
                holes: 0,
                edge_sentinel: 6,
                // The probe's hole is nothing too, and the picture shows grass there.
                sky: 37,
                sky_sentinel: 36,
                interior_by_rung: at_rung(RUNG, 40),
                ..CensusRead::default()
            }
        );
        // The same sentinel one row into the ground, at (5, 4): every neighbour is ground.
        put(&mut picture, w, 5, 4, SENTINEL);
        let lone = HoleBlob {
            pixels: 1,
            left: 5,
            right: 5,
            top: 4,
            bottom: 4,
        };
        assert_eq!(
            census(&picture, &probe, w, h),
            CensusRead {
                holes: 1,
                blobs: 1,
                largest: Some(lone),
                first_blobs: vec![lone],
                first_holes: vec![(5, 4)],
                holes_by_rung: at_rung(RUNG, 1),
                ..edges
            }
        );
    }

    /// NO SENTINEL ON THE GROUND, NO HOLE. And the sentinel's own control: a stated sentinel that
    /// the picture does not store (a camera with HDR tone-mapped the clear colour) finds no sky
    /// pixel, so the zero hole count of that run is empty.
    #[test]
    fn a_picture_with_no_sentinel_on_the_ground_counts_no_hole() {
        let (w, h) = (12, 10);
        let (picture, probe) = scene(w, h, 3);
        assert_eq!(
            census(&picture, &probe, w, h),
            CensusRead {
                ground: 84,
                interior: 50,
                sky: 36,
                sky_sentinel: 36,
                interior_by_rung: at_rung(RUNG, 50),
                ..CensusRead::default()
            }
        );
        let frame = CensusFrame {
            rgba: &picture,
            probe: &probe,
            width: w,
            height: h,
        };
        let wrong = picture_census(&frame, [0, 255, 250]).expect("a well-formed frame");
        assert_eq!((wrong.sky, wrong.sky_sentinel, wrong.holes), (36, 0, 0));
    }

    /// THE BLOBS ARE 4-CONNECTED. Two holes that touch only at a corner are two blobs. A blob's
    /// box reaches left of its first pixel. The largest blob stays the first one on a tie.
    #[test]
    fn the_blobs_are_four_connected_and_the_first_largest_wins_a_tie() {
        let (w, h) = (10, 8);
        let (mut picture, probe) = scene(w, h, 0);
        let painted = [
            // A: one pixel.
            (2, 2),
            // B: one pixel at A's corner.
            (3, 3),
            // C: six pixels, whose box reaches left of its first pixel.
            (6, 2),
            (6, 3),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            // D: two pixels on the left.
            (1, 5),
            (2, 5),
            // E: six pixels, a tie with C, at D's corner.
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 6),
            (7, 6),
            (8, 6),
        ];
        for (x, y) in painted {
            put(&mut picture, w, x, y, SENTINEL);
        }
        let blob = |pixels: u64, left: usize, right: usize, top: usize, bottom: usize| HoleBlob {
            pixels,
            left,
            right,
            top,
            bottom,
        };
        let c = blob(6, 5, 8, 2, 4);
        assert_eq!(
            census(&picture, &probe, w, h),
            CensusRead {
                ground: 80,
                interior: 48,
                holes: 16,
                blobs: 5,
                largest: Some(c),
                // In scan order of their first pixels: A (2, 2), C (6, 2), B (3, 3), D (1, 5),
                // E (3, 6).
                first_blobs: vec![
                    blob(1, 2, 2, 2, 2),
                    c,
                    blob(1, 3, 3, 3, 3),
                    blob(2, 1, 2, 5, 5),
                    blob(6, 3, 8, 6, 6),
                ],
                first_holes: vec![
                    (2, 2),
                    (6, 2),
                    (3, 3),
                    (6, 3),
                    (5, 4),
                    (6, 4),
                    (7, 4),
                    (8, 4),
                    (1, 5),
                    (2, 5),
                    (3, 6),
                    (4, 6),
                    (5, 6),
                    (6, 6),
                    (7, 6),
                    (8, 6),
                ],
                interior_by_rung: at_rung(RUNG, 48),
                holes_by_rung: at_rung(RUNG, 16),
                ..CensusRead::default()
            }
        );
    }

    /// The reading keeps the first holes and the first blobs, and it counts them all.
    #[test]
    fn the_reading_keeps_the_first_holes_and_blobs_and_counts_every_one() {
        let n = FIRST_HOLES_KEPT.max(FIRST_BLOBS_KEPT) + 2;
        // Lone holes at columns 2, 4, …, 2n of row 2; the last column, 2n + 2, is the border.
        let (w, h) = (2 * n + 3, 5);
        let (mut picture, probe) = scene(w, h, 0);
        for k in 0..n {
            put(&mut picture, w, 2 + 2 * k, 2, SENTINEL);
        }
        let read = census(&picture, &probe, w, h);
        assert_eq!((read.holes, read.blobs), (n as u64, n as u64));
        assert_eq!(read.first_holes.len(), FIRST_HOLES_KEPT);
        assert_eq!(
            read.first_holes[FIRST_HOLES_KEPT - 1],
            (2 * FIRST_HOLES_KEPT, 2)
        );
        assert_eq!(read.first_blobs.len(), FIRST_BLOBS_KEPT);
        let lone = |x: usize| HoleBlob {
            pixels: 1,
            left: x,
            right: x,
            top: 2,
            bottom: 2,
        };
        assert_eq!(
            read.first_blobs[FIRST_BLOBS_KEPT - 1],
            lone(2 * FIRST_BLOBS_KEPT)
        );
        // Every blob ties at one pixel: the first stays the largest.
        assert_eq!(read.largest, Some(lone(2)));
    }

    /// A frame that is not one picture and one probe of its stated size reads `None`, never a
    /// zero; a size that overflows reads `None`, never a panic.
    #[test]
    fn a_frame_that_is_not_one_picture_and_one_probe_of_its_size_reads_none() {
        let (w, h) = (4, 3);
        let (picture, probe) = scene(w, h, 1);
        let read = |rgba: &[u8], probe: &[u8], width: usize, height: usize| {
            let frame = CensusFrame {
                rgba,
                probe,
                width,
                height,
            };
            picture_census(&frame, SENTINEL)
        };
        assert_eq!(read(&picture, &probe, 0, h), None);
        assert_eq!(read(&picture, &probe, w, 0), None);
        assert_eq!(read(&picture[..8], &probe, w, h), None);
        assert_eq!(read(&picture, &probe[..8], w, h), None);
        assert_eq!(read(&picture, &probe, usize::MAX, 2), None);
        assert_eq!(read(&picture, &probe, w, h).map(|r| r.ground), Some(8));
    }

    /// The sentinel shows within the tolerance on every channel, on both sides; one level more on
    /// any channel is another colour. The ground's colour is far from it.
    #[test]
    fn the_sentinel_shows_within_the_tolerance_on_every_channel() {
        let t = SENTINEL_TOLERANCE;
        assert!(shows_sentinel(SENTINEL, SENTINEL));
        assert!(shows_sentinel([t, 255 - t, 255 - t], SENTINEL));
        assert!(!shows_sentinel([t + 1, 255, 255], SENTINEL));
        assert!(!shows_sentinel([0, 254 - t, 255], SENTINEL));
        assert!(!shows_sentinel([0, 255, 254 - t], SENTINEL));
        let grey = [128, 128, 128];
        assert!(shows_sentinel([128 + t, 128 - t, 128 + t], grey));
        assert!(!shows_sentinel([129 + t, 128, 128], grey));
        assert!(!shows_sentinel([128, 127 - t, 128], grey));
        assert!(!shows_sentinel([128, 128, 129 + t], grey));
        assert!(!shows_sentinel(GRASS, SENTINEL));
        // The harness's failure colour is not the sentinel: a draw without a real material on the
        // ground never reads as a hole, and a census run keeps its "zero magenta" check.
        let failure = crate::assert::MAGENTA;
        assert!(!shows_sentinel(
            [failure[0], failure[1], failure[2]],
            SENTINEL
        ));
    }

    /// THE HOLES BY RUNG. The ground's left half is rung 9 and its right half rung 10. A refused
    /// patch on rung 10 counts on rung 10 alone, and the judged pixels split by rung. The flight's
    /// control moves a whole rung, and this reading shows that its holes stand on that rung.
    #[test]
    fn the_holes_and_the_judged_pixels_count_by_the_probe_s_rung() {
        let (w, h) = (12, 10);
        let (mut picture, mut probe) = scene(w, h, 3);
        for y in 3..h {
            for x in 6..w {
                put(&mut probe, w, x, y, GROUND_COARSER);
            }
        }
        // A refused patch on the coarser rung: columns 8..=9, rows 5..=6.
        for y in 5..=6 {
            for x in 8..=9 {
                put(&mut picture, w, x, y, SENTINEL);
            }
        }
        let read = census(&picture, &probe, w, h);
        // The judged pixels are rows 4..=8 by columns 1..=10, split at column 6: rung 9 holds
        // columns 1..=5 and rung 10 holds columns 6..=10. A rung boundary is not an edge, because
        // both sides are ground.
        let mut judged = at_rung(RUNG, 25);
        judged[RUNG + 1] = 25;
        assert_eq!(read.interior_by_rung, judged);
        assert_eq!(read.holes_by_rung, at_rung(RUNG + 1, 4));
        assert_eq!(read.holes, 4);
    }
}
