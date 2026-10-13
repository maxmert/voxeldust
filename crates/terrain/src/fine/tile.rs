//! ★ THE FINE TILE'S FORMS (B2; rulings B83 item 4 and B85): what the planet's shard keeps and what it ships.
//!
//! **The raw tile** is the solve's raster ([`super::l1::L1Tile`]): the tile and half its buffer, placed on the level's
//! global cells as a [`Field`] by [`raw_field`]. The shard keeps it, because a neighbour solved later finishes against it.
//!
//! **The finished tile** ([`finish`]) is the tile's own square after its cuts (B85: from its nine raw rasters by the corner
//! rule) and its five mean levels (B83 item 2: rungs 8–12) — the levels a host's [`super::read::FineBook`] holds.
//!
//! **The bytes** ([`encode_levels`], [`decode_levels`]): a form byte, the level count, the tile's side in rung-7 cells
//! as a shift, then each level's length (four bytes, little end first) and its adaptive Rice code ([`super::coder`]).
//! One form for the store and the wire, decoded alone — a client needs no body to read it — the same integers on every
//! host.
//!
//! **Example.** The belt tile (3, 23, 69) of the home planet: its raw raster kept as 640 × 640 heights; once its eight
//! neighbours are solved, its finished square of 512 × 512 and the 256² … 16² means, about 420 KB in all, stored and
//! shipped as one record.

use super::coder;
use super::finish::finished_square;
use super::l1::{BUFFER_NODES, L1Tile, TILE_NODES, TileKey};
use super::points::chart_cell;
use super::read::MEAN_LEVELS;
use super::stitch::{Field, MISSING, mean_levels};

/// The finished record's form; a change of the form is a new number.
pub const TILE_FORM: u8 = 1;
/// ★ THE LEVELS' RECORD'S FORM (ruling B86 item 3, far copies first): some of a finished tile's levels, from a finer one
/// to the coarsest — what one ship to one session carries.
pub const RANGE_FORM: u8 = 2;
/// The raw record's form.
pub const RAW_FORM: u8 = 1;
/// ★ THE FINE GENERATOR'S VERSION (B2; the way-in design §4.2 and §4.6): every change to what a fine tile holds for the
/// same artifact — the points, the graph, the law, the rounds, the raster, the cut, the means, a record's form — is a
/// new number, so a store never serves a tile an older generator made. 2 (ruling B92, 2026-10-13): the 128 m level's
/// rounds on a drainage network built once. 3 (B92 amended, the same day): the drainage built again after every second
/// round (`l1::fine_rounds`; built once, it made walls and spires).
pub const FINE_VERSION: u32 = 3;

/// ★ THE INPUT WORD a stored fine tile is keyed by: the artifact's digest and [`FINE_VERSION`] folded — the same body
/// and the same artifact give the same word on every host, and a world change (a new artifact) or a fine change gives
/// another.
#[must_use]
pub fn input_word(artifact_digest: [u64; 2]) -> u64 {
    use vd_seed::digest::{FNV_OFFSET, fnv1a_u64};
    fnv1a_u64(
        fnv1a_u64(
            fnv1a_u64(FNV_OFFSET, artifact_digest[0]),
            artifact_digest[1],
        ),
        u64::from(FINE_VERSION),
    )
}

/// ★ THE RAW RECORD (what the shard keeps of a solve, for a neighbour finished later): the form, the node's cells as a
/// shift, and the raster's adaptive Rice code. A cell no face holds (the cube corner's diagonal region) is coded as the
/// cell before it and restored from the chart on reading — the coder carries no missing word. `None` where the raster
/// cannot be coded.
#[must_use]
pub fn encode_raw(field: &Field, shift: u32) -> Option<Vec<u8>> {
    let mut last = 0i32;
    let filled: Vec<i32> = field
        .v
        .iter()
        .map(|&q| {
            if q != MISSING {
                last = q;
            }
            last
        })
        .collect();
    let mut out = vec![RAW_FORM, shift as u8];
    out.extend(coder::encode(&filled, field.w, field.h)?);
    Some(out)
}

/// The raw field of tile `key` back from its record, `n7` rung 7's cells along a face's edge. `None` where the form is
/// not this one's or the bytes end early.
#[must_use]
pub fn decode_raw(bytes: &[u8], key: TileKey, n7: i32) -> Option<Field> {
    let (&form, rest) = bytes.split_first()?;
    let (&shift, code) = rest.split_first()?;
    if (form != RAW_FORM) | (shift > 16) {
        return None;
    }
    let shift = u32::from(shift);
    let side = ((TILE_NODES + BUFFER_NODES) << shift) as usize;
    let mut v = coder::decode(code, side, side)?;
    let (i0, j0) = (
        (key.tx * TILE_NODES - 1) << shift,
        (key.ty * TILE_NODES - 1) << shift,
    );
    for (b, row) in v.chunks_mut(side).enumerate() {
        for (a, q) in row.iter_mut().enumerate() {
            if chart_cell(key.face, i0 + a as i32, j0 + b as i32, n7).is_none() {
                *q = MISSING;
            }
        }
    }
    Some(Field {
        gx0: i64::from(i0),
        gy0: i64::from(j0),
        w: side,
        h: side,
        v,
    })
}

/// The raw solve's raster as a field on the level's global cells: the tile and one node beyond each edge, `shift` the
/// level's cells to a node as a shift.
#[must_use]
pub fn raw_field(key: TileKey, tile: L1Tile, shift: u32) -> Field {
    let node = 1i64 << shift;
    Field {
        gx0: (i64::from(key.tx) * i64::from(TILE_NODES) - 1) * node,
        gy0: (i64::from(key.ty) * i64::from(TILE_NODES) - 1) * node,
        w: tile.side,
        h: tile.side,
        v: tile.quarters,
    }
}

/// ★ THE FINISHED TILE in slot 4 of `block` from its nine raw fields (row by row, NW … SE; each its tile's key and raw
/// field in its own face's chart; `None` past a cube corner), `n7` rung 7's cells along a face's edge: the square and its
/// mean levels, level `k` at index `k` ([`super::finish::finished_square`]: any tile of the cube, a face's edge and a
/// cube corner too). The cut's band is a node less the blend, so no blend reads past a raw field that reaches one node
/// beyond its tile. `None` where a cut cannot be joined or a level not made.
#[must_use]
pub fn finish(block: [Option<(TileKey, &Field)>; 9], shift: u32, n7: i32) -> Option<Vec<Vec<i32>>> {
    let square = finished_square(block, shift, n7)?;
    let means = mean_levels(&square, square.gx0, square.gy0, square.w, MEAN_LEVELS)?;
    let mut levels = Vec::with_capacity(means.len() + 1);
    levels.push(square.v);
    levels.extend(means);
    Some(levels)
}

/// The levels of a tile `2^ts` cells wide as one record: the form, the count, the side's shift, then each level's
/// length and code. `None` where a level cannot be coded (a step past the coder's plain word) or does not have the size
/// its side gives it.
#[must_use]
pub fn encode_levels(levels: &[Vec<i32>], ts: u32) -> Option<Vec<u8>> {
    let side = 1usize << ts;
    let mut out = vec![TILE_FORM, levels.len() as u8, ts as u8];
    for (k, level) in levels.iter().enumerate() {
        let s = side >> k;
        if level.len() != s * s {
            return None;
        }
        let code = coder::encode(level, s, s)?;
        out.extend_from_slice(&(code.len() as u32).to_le_bytes());
        out.extend_from_slice(&code);
    }
    Some(out)
}

/// The tile's side shift and its levels back from a record. `None` where the form or the count is not this one's, the
/// side cannot carry the levels, or the bytes end early.
#[must_use]
pub fn decode_levels(bytes: &[u8]) -> Option<(u32, Vec<Vec<i32>>)> {
    let (&form, rest) = bytes.split_first()?;
    let (&count, rest) = rest.split_first()?;
    let (&ts, rest) = rest.split_first()?;
    if (form != TILE_FORM)
        | (u32::from(count) != MEAN_LEVELS + 1)
        | (u32::from(ts) < MEAN_LEVELS)
        | (ts > 16)
    {
        return None;
    }
    Some((
        u32::from(ts),
        decode_codes(rest, 1usize << ts, 0, usize::from(count))?,
    ))
}

/// The levels `lo .. lo + count` of a tile `side` cells wide, each a length and a code, from `rest`.
fn decode_codes(mut rest: &[u8], side: usize, lo: usize, count: usize) -> Option<Vec<Vec<i32>>> {
    let mut levels = Vec::with_capacity(count);
    for k in lo..lo + count {
        let (len, tail) = rest.split_first_chunk::<4>()?;
        let len = u32::from_le_bytes(*len) as usize;
        let code = tail.get(..len)?;
        let s = side >> k;
        levels.push(coder::decode(code, s, s)?);
        rest = &tail[len..];
    }
    Some(levels)
}

/// ★ THE LEVELS `lo ..` THE COARSEST OF A FINISHED RECORD, as one record of [`RANGE_FORM`] (ruling B86 item 3): the form,
/// the first level, the count, the side's shift, then each level's length and code cut from the finished record as they
/// stand — nothing coded again. `None` where `record` is no finished record or `lo` is past its coarsest level.
///
/// **Example.** A tile 600 km from a pilot on the home belt is read only at rungs 10–12: the shard ships its levels 3–5,
/// about 6 KB, where the whole record is about 420 KB.
#[must_use]
pub fn range_record(record: &[u8], lo: u32, hi: u32) -> Option<Vec<u8>> {
    let (&form, rest) = record.split_first()?;
    let (&count, rest) = rest.split_first()?;
    let (&ts, mut rest) = rest.split_first()?;
    if (form != TILE_FORM) | (lo >= hi) | (hi > u32::from(count)) {
        return None;
    }
    let mut out = vec![RANGE_FORM, lo as u8, (hi - lo) as u8, ts];
    for k in 0..hi {
        let (len, tail) = rest.split_first_chunk::<4>()?;
        let n = u32::from_le_bytes(*len) as usize;
        let code = tail.get(..n)?;
        if k >= lo {
            out.extend_from_slice(len);
            out.extend_from_slice(code);
        }
        rest = &tail[n..];
    }
    Some(out)
}

/// ★ A LEVELS' RECORD'S LABEL — its first level and its count — read off its first bytes alone (a host that relays a
/// record undecoded, the gateway, keys its cache by it). `None` where the bytes are no levels' record.
#[must_use]
pub fn range_label(bytes: &[u8]) -> Option<(u8, u8)> {
    match bytes {
        [form, lo, count, ..] if *form == RANGE_FORM => Some((*lo, *count)),
        _ => None,
    }
}

/// A levels' record back: the side's shift, the first level, and the levels from it. `None` where the form is not this
/// one's, the levels pass the coarsest, the side cannot carry them, or the bytes end early.
#[must_use]
pub fn decode_range(bytes: &[u8]) -> Option<(u32, u32, Vec<Vec<i32>>)> {
    let [form, lo, count, ts, rest @ ..] = bytes else {
        return None;
    };
    let (lo, count, ts) = (u32::from(*lo), u32::from(*count), u32::from(*ts));
    if (*form != RANGE_FORM)
        | (count == 0)
        | (lo + count > MEAN_LEVELS + 1)
        | !(MEAN_LEVELS..=16).contains(&ts)
    {
        return None;
    }
    Some((
        ts,
        lo,
        decode_codes(rest, 1usize << ts, lo as usize, count as usize)?,
    ))
}

#[cfg(test)]
mod tests {
    //! A test may divide (ruling F7's rule is about the shipped path): a fixture places its cells with a quotient and a
    //! remainder.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a fixture's placing; never a kernel's path"
    )]
    use super::*;
    use vd_seed::bend::Face;

    #[test]
    fn a_record_carries_every_level_back_exactly_and_refuses_what_is_not_one() {
        // A 32-cell tile's square and its five means over a slope with a ridge.
        let square: Vec<i32> = (0..32 * 32)
            .map(|c| 3 * (c & 31) - 2 * (c >> 5) + if (c & 7) == 3 { 40 } else { 0 })
            .collect();
        let f = Field {
            gx0: 0,
            gy0: 0,
            w: 32,
            h: 32,
            v: square.clone(),
        };
        let mut levels = vec![square];
        levels.extend(mean_levels(&f, 0, 0, 32, MEAN_LEVELS).expect("the means"));
        let bytes = encode_levels(&levels, 5).expect("codes");
        assert_eq!(decode_levels(&bytes), Some((5, levels.clone())));
        // Another form, another count, a side too small or too large, bytes that end early, a level of the wrong size.
        let altered = |at: usize, v: u8| {
            let mut b = bytes.clone();
            b[at] = v;
            decode_levels(&b)
        };
        assert_eq!(altered(0, TILE_FORM + 1), None);
        assert_eq!(altered(1, 2), None);
        assert_eq!(altered(2, 4), None);
        assert_eq!(altered(2, 17), None);
        assert_eq!(decode_levels(&bytes[..bytes.len() - 1]), None);
        for cut in 0..4 {
            assert_eq!(decode_levels(&bytes[..cut]), None);
        }
        assert_eq!(decode_levels(&bytes[..5]), None);
        // ★ THE LEVELS' RECORDS (B86 item 3): any range cut from the finished record decodes to those levels exactly; a
        // range past the coarsest, an empty one, or a record that is not a finished one is refused.
        for (lo, hi) in [(3u32, 6u32), (0, 2), (5, 6), (0, 6)] {
            let cut = range_record(&bytes, lo, hi).expect("a range");
            assert_eq!(
                decode_range(&cut),
                Some((5, lo, levels[lo as usize..hi as usize].to_vec()))
            );
        }
        assert_eq!(range_record(&bytes, 4, 7), None);
        assert_eq!(range_record(&bytes, 3, 3), None);
        assert_eq!(range_record(&bytes[..2], 0, 1), None);
        assert_eq!(range_record(&bytes[..bytes.len() - 1], 0, 6), None);
        let mut other = bytes.clone();
        other[0] = RANGE_FORM;
        assert_eq!(range_record(&other, 0, 1), None);
        let cut = range_record(&bytes, 3, 6).expect("a range");
        let refused = |at: usize, v: u8| {
            let mut c = cut.clone();
            c[at] = v;
            decode_range(&c)
        };
        assert_eq!(refused(0, TILE_FORM), None);
        assert_eq!(refused(2, 0), None);
        assert_eq!(refused(2, 4), None);
        assert_eq!(refused(3, 4), None);
        assert_eq!(refused(3, 17), None);
        assert_eq!(decode_range(&cut[..3]), None);
        assert_eq!(decode_range(&cut[..cut.len() - 1]), None);
        levels[3].pop();
        assert_eq!(encode_levels(&levels, 5), None);
        // A step too large for the coder's plain word.
        assert_eq!(encode_levels(&[vec![0, 1 << 24, 0, 0]], 1), None);
    }

    #[test]
    fn a_raw_record_at_a_cube_corner_comes_back_with_its_empty_corner_and_the_input_word_follows_the_artifact()
     {
        // Shift 2 (a node of four cells): the tile (0, 0) of a face 64 cells wide, its raster 40 wide from cell −4; the
        // cells past both edges at once (the cube corner's diagonal region) hold no height.
        let key = TileKey {
            face: Face::PosX,
            tx: 0,
            ty: 0,
        };
        let n7 = 64;
        let v: Vec<i32> = (0..40 * 40)
            .map(|c| {
                let (a, b) = (c % 40 - 4, c / 40 - 4);
                if chart_cell(key.face, a, b, n7).is_none() {
                    MISSING
                } else {
                    3 * a - 5 * b
                }
            })
            .collect();
        assert_eq!(v.iter().filter(|&&q| q == MISSING).count(), 16);
        let field = Field {
            gx0: -4,
            gy0: -4,
            w: 40,
            h: 40,
            v,
        };
        let record = encode_raw(&field, 2).expect("codes");
        assert_eq!(decode_raw(&record, key, n7), Some(field));
        // Another form, a shift past the ladder, bytes that end early.
        let mut other = record.clone();
        other[0] = RAW_FORM + 1;
        assert_eq!(decode_raw(&other, key, n7), None);
        other[0] = RAW_FORM;
        other[1] = 17;
        assert_eq!(decode_raw(&other, key, n7), None);
        assert_eq!(decode_raw(&record[..1], key, n7), None);
        assert_eq!(decode_raw(&record[..record.len() / 2], key, n7), None);
        // A step past the coder's plain word.
        let cliff = Field {
            gx0: -4,
            gy0: -4,
            w: 2,
            h: 1,
            v: vec![0, 1 << 24],
        };
        assert_eq!(encode_raw(&cliff, 2), None);
        // The input word: the same artifact the same word, another artifact another.
        assert_eq!(input_word([1, 2]), input_word([1, 2]));
        assert_ne!(input_word([1, 2]), input_word([1, 3]));
    }

    #[test]
    fn a_raw_tile_stands_one_node_beyond_its_tile_and_nine_equal_ones_finish_into_their_own_ground()
    {
        // Shift 2: a node of four cells, a tile of 32, its raster 40 wide.
        let ground = |x: i64, y: i64| (5 * x + 7 * y) as i32;
        let raw = |tx: i32, ty: i32| {
            let key = TileKey {
                face: Face::PosX,
                tx,
                ty,
            };
            let (x0, y0) = ((i64::from(tx) * 8 - 1) * 4, (i64::from(ty) * 8 - 1) * 4);
            let quarters = (0..40 * 40)
                .map(|c| ground(x0 + (c % 40), y0 + (c / 40)))
                .collect();
            raw_field(
                key,
                L1Tile {
                    side: 40,
                    quarters,
                    points: 0,
                    passes: Vec::new(),
                },
                2,
            )
        };
        let grid: Vec<Field> = (0..3)
            .flat_map(|ty| (0..3).map(move |tx| (tx, ty)))
            .map(|(tx, ty)| raw(tx, ty))
            .collect();
        assert_eq!((grid[4].gx0, grid[4].gy0, grid[4].w), (28, 28, 40));
        let slot = |k: usize| {
            Some((
                TileKey {
                    face: Face::PosX,
                    tx: (k % 3) as i32,
                    ty: (k / 3) as i32,
                },
                &grid[k],
            ))
        };
        // A band of a node less the blend is under nought here: no cut can join, no tile.
        assert_eq!(finish(std::array::from_fn(slot), 2, 96), None);
        // Shift 4 (a 16-cell node, a 128-cell tile, its raster 160 wide): the tile's own ground and its means.
        let raw16 = |tx: i32, ty: i32| {
            let key = TileKey {
                face: Face::PosX,
                tx,
                ty,
            };
            let (x0, y0) = ((i64::from(tx) * 8 - 1) * 16, (i64::from(ty) * 8 - 1) * 16);
            let quarters = (0..160 * 160)
                .map(|c| ground(x0 + (c % 160), y0 + (c / 160)))
                .collect();
            raw_field(
                key,
                L1Tile {
                    side: 160,
                    quarters,
                    points: 0,
                    passes: Vec::new(),
                },
                4,
            )
        };
        let grid: Vec<Field> = (0..3)
            .flat_map(|ty| (0..3).map(move |tx| (tx, ty)))
            .map(|(tx, ty)| raw16(tx, ty))
            .collect();
        let slot = |k: usize| {
            Some((
                TileKey {
                    face: Face::PosX,
                    tx: (k % 3) as i32,
                    ty: (k / 3) as i32,
                },
                &grid[k],
            ))
        };
        let levels = finish(std::array::from_fn(slot), 4, 384).expect("finishes");
        assert_eq!(levels.len(), MEAN_LEVELS as usize + 1);
        assert_eq!(levels[0][0], ground(128, 128));
        assert_eq!(levels[0][127 * 128 + 127], ground(255, 255));
        assert_eq!(levels[5].len(), 16);
    }
}
