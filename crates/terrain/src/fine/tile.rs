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
use super::l1::TileKey;
use super::level::{FineLevel, LevelGeom};
use super::points::chart_cell;
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
/// round (`l1::fine_rounds`; built once, it made walls and spires). 4 (slice W2, rulings B105, B114): every standing
/// water's points under its level are outlets, each node's base level its own water, the rivers carved, the water map in
/// the records ([`WATER_TILE_FORM`]). 5 (slice W1 item 4, ruling B94 item 1): an 8 km river leaves a tile at its bed,
/// under the lower row by the channel law's depth (`l1::exits_at_bed`). 6 (ruling B115, form (a′)): a lake stands at its
/// 128 m spill — each tile solved with the fine level of every lake in its domain (`l1::solve_l1_levels`), every river exit
/// an outlet at any height, a dry end's lowest point, the rise's floor. 7 (the same ruling): a lowered lake's dry floor
/// drains into its lake (its base level the lake's fine level), never a dry end's playa. 8 (ruling B116 part 1): the
/// coarse copies' heights by the majority class (`class_mean_levels`). 9 (the lake floor design's bench, not yet ruled
/// into the game): only a dry hollow's pit keeps a dry end's playa; its other nodes drain to it (`l1::Waters::dry_pit`).
/// 10 (ruling B115, the lakes together; the owner's window, a 595 m wall of water): a lake's rounds name the other lakes of
/// each tile's domain at their settled levels, and a final solve whose raster shows a lake passing under its level reopens
/// it (the planet's shard, `vd_bins::fine_source`), so no final raster holds a lake over its pass.
pub const FINE_VERSION: u32 = 10;

/// ★ THE 16 m LEVEL'S OWN VERSION (ruling B95): every change to what a 16 m tile holds for the same 128 m ground is a
/// new number; it folds into the 16 m level's input word only ([`input_word_of`]), so a 128 m tile stays valid across it.
/// 2 (ruling B97, 2026-10-14): the 16 m law is the 128 m level's own law under tan 35° (the creep, the shared water and
/// the pinned channels made walls and needles on steep ground). 3 (slice W2): the 16 m channels carved in the law's raster
/// and the 16 m water map from the 128 m water. 4 (ruling B115): the 16 m water spreads only inside the 128 m water's
/// reach (`water::raster_water_within`).
pub const L2_VERSION: u32 = 4;

/// ★ THE INPUT WORD of a stored tile of `level`: the 128 m level's is [`input_word`]; the 16 m level folds its own
/// version and number in after it (a 16 m tile reads the finished 128 m ground, so it changes with it).
#[must_use]
pub fn input_word_of(level: FineLevel, artifact_digest: [u64; 2]) -> u64 {
    use vd_seed::digest::fnv1a_u64;
    match level {
        FineLevel::L1 => input_word(artifact_digest),
        FineLevel::L2 => fnv1a_u64(
            input_word(artifact_digest),
            (u64::from(level.id()) << 32) | u64::from(L2_VERSION),
        ),
    }
}

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

/// The raw field of tile `key` of the level `geom` back from its record. `None` where the form is not this one's or the
/// bytes end early.
#[must_use]
pub fn decode_raw(bytes: &[u8], key: TileKey, geom: LevelGeom) -> Option<Field> {
    let (&form, rest) = bytes.split_first()?;
    let (&shift, code) = rest.split_first()?;
    if (form != RAW_FORM) | (shift > 16) {
        return None;
    }
    if u32::from(shift) != geom.shift {
        return None;
    }
    let side = (geom.side() + 2 * geom.margin) as usize;
    let mut v = coder::decode(code, side, side)?;
    let (i0, j0) = (
        (i64::from(key.tx) * geom.side() - geom.margin) as i32,
        (i64::from(key.ty) * geom.side() - geom.margin) as i32,
    );
    for (b, row) in v.chunks_mut(side).enumerate() {
        for (a, q) in row.iter_mut().enumerate() {
            if chart_cell(key.face, i0 + a as i32, j0 + b as i32, geom.n).is_none() {
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

/// A raw solve's raster (`side × side` heights, the tile and its margin beyond each edge) as a field on the level's
/// global cells.
#[must_use]
pub fn raw_field(key: TileKey, geom: LevelGeom, side: usize, quarters: Vec<i32>) -> Field {
    Field {
        gx0: i64::from(key.tx) * geom.side() - geom.margin,
        gy0: i64::from(key.ty) * geom.side() - geom.margin,
        w: side,
        h: side,
        v: quarters,
    }
}

/// ★ THE FINISHED TILE in slot 4 of `block` from its nine raw fields (row by row, NW … SE; each its tile's key and raw
/// field in its own face's chart; `None` past a cube corner), at the level `geom`: the square and its mean levels, level
/// `k` at index `k` ([`super::finish::finished_square`]: any tile of the cube, a face's edge and a cube corner too). The
/// cut's band is the margin less the blend, so no blend reads past a raw field. `None` where a cut cannot be joined or a
/// level not made.
#[must_use]
pub fn finish(block: [Option<(TileKey, &Field)>; 9], geom: LevelGeom) -> Option<Vec<Vec<i32>>> {
    let square = finished_square(block, geom)?;
    let means = mean_levels(
        &square,
        square.gx0,
        square.gy0,
        square.w,
        geom.level.mean_levels(),
    )?;
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

/// The tile's side shift and its levels back from a record of `level`. `None` where the form or the count is not this
/// one's, the side cannot carry the levels, or the bytes end early.
#[must_use]
pub fn decode_levels(bytes: &[u8], level: FineLevel) -> Option<(u32, Vec<Vec<i32>>)> {
    let (&form, rest) = bytes.split_first()?;
    let (&count, rest) = rest.split_first()?;
    let (&ts, rest) = rest.split_first()?;
    let means = level.mean_levels();
    if (form != TILE_FORM) | (u32::from(count) != means + 1) | (u32::from(ts) < means) | (ts > 16) {
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
        [form, lo, count, ..] if (*form == RANGE_FORM) | (*form == WATER_RANGE_FORM) => {
            Some((*lo, *count))
        }
        _ => None,
    }
}

/// A levels' record of `level` back: the side's shift, the first level, and the levels from it. `None` where the form is
/// not this one's, the levels pass the coarsest, the side cannot carry them, or the bytes end early.
#[must_use]
pub fn decode_range(bytes: &[u8], level: FineLevel) -> Option<(u32, u32, Vec<Vec<i32>>)> {
    let [form, lo, count, ts, rest @ ..] = bytes else {
        return None;
    };
    let (lo, count, ts) = (u32::from(*lo), u32::from(*count), u32::from(*ts));
    if (*form != RANGE_FORM)
        | (count == 0)
        | (lo + count > level.mean_levels() + 1)
        | !(level.mean_levels()..=16).contains(&ts)
    {
        return None;
    }
    Some((
        ts,
        lo,
        decode_codes(rest, 1usize << ts, lo as usize, count as usize)?,
    ))
}

/// ★ THE WATER RECORDS' FORMS (slice W2; the one-volume revision §3.4 and §3.6; ruling B105 Q2): a finished record of
/// [`WATER_TILE_FORM`] carries each level as THREE planes — the heights, the water's level (quarter metres over the sea)
/// and the wet share (0 ..= [`WET_FULL`]) — a levels' record of [`WATER_RANGE_FORM`] some of them, and a raw record of
/// [`WATER_RAW_FORM`] its raster's three. A level holds meaning only where the share is full; the coder carries no
/// missing word, so a dry cell's level is coded as the level before it.
pub const WATER_TILE_FORM: u8 = 3;
pub const WATER_RANGE_FORM: u8 = 4;
pub const WATER_RAW_FORM: u8 = 2;
/// A cell wholly under water (a standing water or a carved river); a smaller share is a narrower river's tint.
pub const WET_FULL: i32 = 255;

/// The water's planes of a raster from its water map and its narrower rivers' shares: the level where a water holds the
/// cell ([`MISSING`] elsewhere) and the share — full there, the narrower rivers' share elsewhere.
#[must_use]
pub fn water_planes(water: &[i32], river_share: &[u8]) -> (Vec<i32>, Vec<i32>) {
    let wet = water
        .iter()
        .zip(river_share)
        .map(|(&w, &s)| if w == MISSING { i32::from(s) } else { WET_FULL })
        .collect();
    (water.to_vec(), wet)
}

/// A plane with each missing value taken from the one before it (the coder's form; the share says what is dry).
fn filled(v: &[i32]) -> Vec<i32> {
    let mut last = 0i32;
    v.iter()
        .map(|&q| {
            if q != MISSING {
                last = q;
            }
            last
        })
        .collect()
}

/// Codes after each other, each with its length (four bytes, little end first).
fn push_code(out: &mut Vec<u8>, plane: &[i32], w: usize, h: usize) -> Option<()> {
    let code = coder::encode(plane, w, h)?;
    out.extend_from_slice(&(code.len() as u32).to_le_bytes());
    out.extend_from_slice(&code);
    Some(())
}

/// One length-prefixed code of a `w × h` plane off the front of `rest`.
fn pop_code(rest: &mut &[u8], w: usize, h: usize) -> Option<Vec<i32>> {
    let (len, tail) = rest.split_first_chunk::<4>()?;
    let len = u32::from_le_bytes(*len) as usize;
    let code = tail.get(..len)?;
    *rest = &tail[len..];
    coder::decode(code, w, h)
}

/// The level plane where the share is full, [`MISSING`] elsewhere.
fn level_where_full(level: &mut [i32], wet: &[i32]) {
    for (l, &s) in level.iter_mut().zip(wet) {
        if s != WET_FULL {
            *l = MISSING;
        }
    }
}

/// ★ THE RAW RECORD WITH ITS WATER: [`encode_raw`]'s heights and the raster's water level and share, each a
/// length-prefixed code. `None` where a plane cannot be coded.
#[must_use]
pub fn encode_raw_water(field: &Field, level: &[i32], wet: &[i32], shift: u32) -> Option<Vec<u8>> {
    let mut out = vec![WATER_RAW_FORM, shift as u8];
    push_code(&mut out, &filled(&field.v), field.w, field.h)?;
    push_code(&mut out, &filled(level), field.w, field.h)?;
    push_code(&mut out, wet, field.w, field.h)?;
    Some(out)
}

/// A raw record with its water back: the heights (a cell no face holds missing), the level (missing where the share is
/// not full) and the share, each a field on the level's global cells. `None` as [`decode_raw`].
#[must_use]
pub fn decode_raw_water(bytes: &[u8], key: TileKey, geom: LevelGeom) -> Option<[Field; 3]> {
    let [form, shift, rest @ ..] = bytes else {
        return None;
    };
    let mut rest = rest;
    if (*form != WATER_RAW_FORM) | (u32::from(*shift) != geom.shift) {
        return None;
    }
    let side = (geom.side() + 2 * geom.margin) as usize;
    let mut heights = pop_code(&mut rest, side, side)?;
    let mut level = pop_code(&mut rest, side, side)?;
    let wet = pop_code(&mut rest, side, side)?;
    let (i0, j0) = (
        (i64::from(key.tx) * geom.side() - geom.margin) as i32,
        (i64::from(key.ty) * geom.side() - geom.margin) as i32,
    );
    for (b, row) in heights.chunks_mut(side).enumerate() {
        for (a, q) in row.iter_mut().enumerate() {
            if chart_cell(key.face, i0 + a as i32, j0 + b as i32, geom.n).is_none() {
                *q = MISSING;
            }
        }
    }
    level_where_full(&mut level, &wet);
    let field = |v: Vec<i32>| Field {
        gx0: i64::from(i0),
        gy0: i64::from(j0),
        w: side,
        h: side,
        v,
    };
    Some([field(heights), field(level), field(wet)])
}

/// A finished tile's levels with their water: level `k` at index `k` of each plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterLevels {
    pub heights: Vec<Vec<i32>>,
    pub level: Vec<Vec<i32>>,
    pub wet: Vec<Vec<i32>>,
}

/// ★ THE WATER'S MEAN LEVELS of a finished square `side` cells wide: level `k` the box of `2^k × 2^k` cells — its share
/// the cells' mean share (rounded half up), its level the highest level among its full cells ([`MISSING`] where none):
/// the far map counts the water the near view shows (§3.6), and a far pixel takes the water that holds its cells.
#[must_use]
pub fn water_mean_levels(
    level: &[i32],
    wet: &[i32],
    side: usize,
    levels: u32,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let (mut lv, mut wt) = (Vec::new(), Vec::new());
    for k in 1..=levels {
        let n = side >> k;
        let (mut l, mut w) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
        for by in 0..n {
            for bx in 0..n {
                let (mut sum, mut top) = (0i64, MISSING);
                for y in 0..1usize << k {
                    for x in 0..1usize << k {
                        let c = ((by << k) + y) * side + (bx << k) + x;
                        sum += i64::from(wet[c]);
                        if wet[c] == WET_FULL {
                            top = top.max(level[c]);
                        }
                    }
                }
                w.push(((sum + (1i64 << (2 * k - 1))) >> (2 * k)) as i32);
                l.push(top);
            }
        }
        lv.push(l);
        wt.push(w);
    }
    (lv, wt)
}

/// ★ THE COARSE COPIES' HEIGHTS BY THE MAJORITY CLASS (ruling B116 part 1; the one-volume revision §3.6): level `k` the
/// box of `2^k × 2^k` cells of the finished square `side` cells wide — where at least half its cells stand under water
/// (`wet` full, a `level` named), the mean ground of those wet cells (the floor under the water); elsewhere the mean ground
/// of its other cells (the land). So a coarse copy's shore falls where half the finest cells are wet, and the water's depth
/// over the floor stays true for its colour (ruling B109). Rounded half up. `None` where a cell is missing or the side does
/// not halve so often. MEASURED before (2026-10-19, the owner: *"The rung change should be seamless also for the water"*):
/// the plain mean mixed a shore cell's floor and land, so a cell half under water stood over the water and drew dry, and
/// the shore jumped at each rung change.
///
/// **Example.** A 1 km cell on the basin lake's shore holds 40 cells of floor at −2 800 m and 24 of land at −2 300 m: it
/// stands at −2 800 m, under the water at −2 721 m, and draws wet; with 24 of floor and 40 of land it stands at −2 300 m and
/// draws dry.
#[must_use]
#[allow(
    clippy::integer_division,
    reason = "the finish runs on the planet's shard's CPU alone (the client reads the record): an exact integer quotient, \
              one answer on every host"
)]
pub fn class_mean_levels(
    ground: &[i32],
    level: &[i32],
    wet: &[i32],
    side: usize,
    levels: u32,
) -> Option<Vec<Vec<i32>>> {
    if (side >> levels) << levels != side {
        return None;
    }
    let mut out = Vec::with_capacity(levels as usize);
    for k in 1..=levels {
        let n = side >> k;
        let mut plane = Vec::with_capacity(n * n);
        for by in 0..n {
            for bx in 0..n {
                let (mut wet_sum, mut wet_n, mut dry_sum, mut dry_n) = (0i64, 0i64, 0i64, 0i64);
                for y in 0..1usize << k {
                    for x in 0..1usize << k {
                        let c = ((by << k) + y) * side + (bx << k) + x;
                        if ground[c] == MISSING {
                            return None;
                        }
                        if wet[c] == WET_FULL && level[c] != MISSING {
                            wet_sum += i64::from(ground[c]);
                            wet_n += 1;
                        } else {
                            dry_sum += i64::from(ground[c]);
                            dry_n += 1;
                        }
                    }
                }
                let (sum, count) = if 2 * wet_n >= wet_n + dry_n {
                    (wet_sum, wet_n)
                } else {
                    (dry_sum, dry_n)
                };
                plane.push((2 * sum + count).div_euclid(2 * count) as i32);
            }
        }
        out.push(plane);
    }
    Some(out)
}

/// ★ THE FINISHED TILE WITH ITS WATER (slice W2): [`finish`]'s levels, and the water's — each finished cell takes the
/// level and share of the raster its height came from in the cut ([`super::finish::finished_square_planes`]) and stays
/// wet only where the finished ground stands under that level (a blend that lifted the ground over the water dries it);
/// then the water's mean levels ([`water_mean_levels`]). `levels` and `wets` are the nine rasters' water planes beside
/// `block`'s heights. `None` as [`finish`].
#[must_use]
pub fn finish_water(
    block: [Option<(TileKey, &Field)>; 9],
    levels: [Option<&Field>; 9],
    wets: [Option<&Field>; 9],
    geom: LevelGeom,
) -> Option<WaterLevels> {
    let (square, planes) = super::finish::finished_square_planes(block, &[levels, wets], geom)?;
    let [mut level, mut wet]: [Vec<i32>; 2] = planes
        .into_iter()
        .map(|f| f.v)
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    for c in 0..square.v.len() {
        if wet[c] == WET_FULL && !(level[c] != MISSING && square.v[c] < level[c]) {
            wet[c] = 0;
        }
    }
    level_where_full(&mut level, &wet);
    let n_means = geom.level.mean_levels();
    // ★ The coarse copies stand by the majority class (ruling B116 part 1).
    let means = class_mean_levels(&square.v, &level, &wet, square.w, n_means)?;
    let (lv, wt) = water_mean_levels(&level, &wet, square.w, n_means);
    let mut out = WaterLevels {
        heights: vec![square.v],
        level: vec![level],
        wet: vec![wet],
    };
    out.heights.extend(means);
    out.level.extend(lv);
    out.wet.extend(wt);
    Some(out)
}

/// ★ THE FINISHED RECORD WITH ITS WATER, a tile `2^ts` cells wide: the form, the count, the side's shift, then each
/// level's three planes (heights, level, share), each a length and a code. `None` as [`encode_levels`].
#[must_use]
pub fn encode_water_levels(levels: &WaterLevels, ts: u32) -> Option<Vec<u8>> {
    let side = 1usize << ts;
    let count = levels.heights.len();
    if (levels.level.len() != count) | (levels.wet.len() != count) {
        return None;
    }
    let mut out = vec![WATER_TILE_FORM, count as u8, ts as u8];
    for k in 0..count {
        let s = side >> k;
        if (levels.heights[k].len() != s * s)
            | (levels.level[k].len() != s * s)
            | (levels.wet[k].len() != s * s)
        {
            return None;
        }
        push_code(&mut out, &levels.heights[k], s, s)?;
        push_code(&mut out, &filled(&levels.level[k]), s, s)?;
        push_code(&mut out, &levels.wet[k], s, s)?;
    }
    Some(out)
}

/// The levels `lo .. lo + count` of a water record's planes from `rest` (a tile `side` cells wide).
fn decode_water_codes(
    mut rest: &[u8],
    side: usize,
    lo: usize,
    count: usize,
) -> Option<WaterLevels> {
    let mut out = WaterLevels {
        heights: Vec::with_capacity(count),
        level: Vec::with_capacity(count),
        wet: Vec::with_capacity(count),
    };
    for k in lo..lo + count {
        let s = side >> k;
        out.heights.push(pop_code(&mut rest, s, s)?);
        let mut level = pop_code(&mut rest, s, s)?;
        let wet = pop_code(&mut rest, s, s)?;
        level_where_full(&mut level, &wet);
        out.level.push(level);
        out.wet.push(wet);
    }
    Some(out)
}

/// A finished record with its water back: the side's shift and the levels. `None` as [`decode_levels`].
#[must_use]
pub fn decode_water_levels(bytes: &[u8], level: FineLevel) -> Option<(u32, WaterLevels)> {
    let [form, count, ts, rest @ ..] = bytes else {
        return None;
    };
    let means = level.mean_levels();
    if (*form != WATER_TILE_FORM)
        | (u32::from(*count) != means + 1)
        | (u32::from(*ts) < means)
        | (*ts > 16)
    {
        return None;
    }
    Some((
        u32::from(*ts),
        decode_water_codes(rest, 1usize << ts, 0, usize::from(*count))?,
    ))
}

/// ★ THE LEVELS `lo .. hi` OF A FINISHED WATER RECORD as one record of [`WATER_RANGE_FORM`] (ruling B86 item 3, far
/// copies first): the form, the first level, the count, the side's shift, then each level's three length-prefixed codes
/// cut from the record as they stand. `None` as [`range_record`].
#[must_use]
pub fn water_range_record(record: &[u8], lo: u32, hi: u32) -> Option<Vec<u8>> {
    let [form, count, ts, rest @ ..] = record else {
        return None;
    };
    let mut rest = rest;
    if (*form != WATER_TILE_FORM) | (lo >= hi) | (hi > u32::from(*count)) {
        return None;
    }
    let mut out = vec![WATER_RANGE_FORM, lo as u8, (hi - lo) as u8, *ts];
    for k in 0..hi {
        for _ in 0..3 {
            let (len, tail) = rest.split_first_chunk::<4>()?;
            let n = u32::from_le_bytes(*len) as usize;
            let code = tail.get(..n)?;
            if k >= lo {
                out.extend_from_slice(len);
                out.extend_from_slice(code);
            }
            rest = &tail[n..];
        }
    }
    Some(out)
}

/// A levels' water record of `level` back: the side's shift, the first level, and the levels from it. `None` as
/// [`decode_range`].
#[must_use]
pub fn decode_water_range(bytes: &[u8], level: FineLevel) -> Option<(u32, u32, WaterLevels)> {
    let [form, lo, count, ts, rest @ ..] = bytes else {
        return None;
    };
    let (lo, count, ts) = (u32::from(*lo), u32::from(*count), u32::from(*ts));
    if (*form != WATER_RANGE_FORM)
        | (count == 0)
        | (lo + count > level.mean_levels() + 1)
        | (ts < level.mean_levels())
        | (ts > 16)
    {
        return None;
    }
    Some((
        ts,
        lo,
        decode_water_codes(rest, 1usize << ts, lo as usize, count as usize)?,
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

    const L1: FineLevel = FineLevel::L1;

    /// ★ THE MAJORITY CLASS (ruling B116 part 1): a 2 × 2 box three-quarters wet stands at its wet cells' mean floor, one
    /// a quarter wet at its dry cells' mean ground, one half wet at its wet cells' (a tie goes to the water); negative
    /// heights round half up.
    #[test]
    fn a_coarse_copy_stands_by_its_majority_class() {
        let ground = [10, 20, 30, 40];
        let lake = 50;
        let full = |n: usize| -> (Vec<i32>, Vec<i32>) {
            (
                (0..4).map(|c| if c < n { lake } else { MISSING }).collect(),
                (0..4).map(|c| if c < n { WET_FULL } else { 0 }).collect(),
            )
        };
        let (l3, w3) = full(3);
        assert_eq!(
            class_mean_levels(&ground, &l3, &w3, 2, 1),
            Some(vec![vec![20]])
        );
        let (l1, w1) = full(1);
        assert_eq!(
            class_mean_levels(&ground, &l1, &w1, 2, 1),
            Some(vec![vec![30]])
        );
        let (l2, w2) = full(2);
        assert_eq!(
            class_mean_levels(&ground, &l2, &w2, 2, 1),
            Some(vec![vec![15]])
        );
        let low = [-11, -12, -13, -14];
        assert_eq!(
            class_mean_levels(&low, &l3, &w3, 2, 1),
            Some(vec![vec![-12]])
        );
        assert_eq!(class_mean_levels(&ground, &l3, &w3, 3, 1), None);
    }

    /// ★ THE WATER RECORD (slice W2): a 32-cell tile with a lake (level 100 over a floor at 40) and a narrower river's
    /// tint comes back exactly — its heights, its level only where the share is full, its share — the coarser levels
    /// take the mean share and the highest full level, a range record carries the levels it names, and the old forms
    /// are refused.
    #[test]
    fn a_water_record_carries_its_planes_back_and_its_means_count_the_water() {
        let side = 32usize;
        let heights: Vec<i32> = (0..side * side)
            .map(|c| {
                if c % side < 8 {
                    40
                } else {
                    200 + (c / side) as i32
                }
            })
            .collect();
        let water: Vec<i32> = (0..side * side)
            .map(|c| if c % side < 8 { 100 } else { MISSING })
            .collect();
        let river: Vec<u8> = (0..side * side)
            .map(|c| if c % side == 20 { 64 } else { 0 })
            .collect();
        let (level, wet) = water_planes(&water, &river);
        assert_eq!((level[0], wet[0], wet[20], wet[21]), (100, WET_FULL, 64, 0));
        let f = Field {
            gx0: 0,
            gy0: 0,
            w: side,
            h: side,
            v: heights.clone(),
        };
        let n = L1.mean_levels();
        let mut levels = WaterLevels {
            heights: vec![heights],
            level: vec![level],
            wet: vec![wet],
        };
        levels
            .heights
            .extend(mean_levels(&f, 0, 0, side, n).expect("the means"));
        let (lv, wt) = water_mean_levels(&levels.level[0], &levels.wet[0], side, n);
        // The first mean level: a box of four, all lake (share full, level 100) or all dry land.
        assert_eq!(
            (lv[0][0], wt[0][0], lv[0][8], wt[0][8]),
            (100, WET_FULL, MISSING, 0)
        );
        // The coarsest: one cell, a quarter lake and a river column (64 over 32 cells).
        assert_eq!(lv[n as usize - 1][0], 100);
        // (8 columns full and one at 64, 32 rows each, over 1 024 cells, rounded half up) = 65.75 → 66.
        assert_eq!(wt[n as usize - 1][0], ((255 * 8 + 64) * 32 + 512) >> 10);
        levels.level.extend(lv);
        levels.wet.extend(wt);
        let bytes = encode_water_levels(&levels, 5).expect("codes");
        assert_eq!(decode_water_levels(&bytes, L1), Some((5, levels.clone())));
        assert_eq!(
            decode_levels(&bytes, L1),
            None,
            "the old form's reader refuses it"
        );
        let range = water_range_record(&bytes, 3, n + 1).expect("a range");
        let (ts, lo, part) = decode_water_range(&range, L1).expect("decodes");
        assert_eq!((ts, lo), (5, 3));
        assert_eq!(part.heights[..], levels.heights[3..]);
        assert_eq!(part.level[..], levels.level[3..]);
        assert_eq!(part.wet[..], levels.wet[3..]);
        assert_eq!(water_range_record(&bytes, 4, 4), None);
        assert_eq!(decode_water_range(&bytes, L1), None);
    }

    /// A small body's 128 m level: a node of `1 << shift` cells, an eight-node tile, the raw raster a node past it.
    fn l1_geom(shift: u32, n: i32) -> LevelGeom {
        LevelGeom {
            level: L1,
            shift,
            ts: shift + 3,
            margin: 1 << shift,
            n,
        }
    }

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
        levels.extend(mean_levels(&f, 0, 0, 32, L1.mean_levels()).expect("the means"));
        let bytes = encode_levels(&levels, 5).expect("codes");
        assert_eq!(decode_levels(&bytes, L1), Some((5, levels.clone())));
        // Another form, another count, a side too small or too large, bytes that end early, a level of the wrong size.
        let altered = |at: usize, v: u8| {
            let mut b = bytes.clone();
            b[at] = v;
            decode_levels(&b, L1)
        };
        assert_eq!(altered(0, TILE_FORM + 1), None);
        assert_eq!(altered(1, 2), None);
        assert_eq!(altered(2, 4), None);
        assert_eq!(altered(2, 17), None);
        assert_eq!(decode_levels(&bytes[..bytes.len() - 1], L1), None);
        for cut in 0..4 {
            assert_eq!(decode_levels(&bytes[..cut], L1), None);
        }
        assert_eq!(decode_levels(&bytes[..5], L1), None);
        // ★ THE LEVELS' RECORDS (B86 item 3): any range cut from the finished record decodes to those levels exactly; a
        // range past the coarsest, an empty one, or a record that is not a finished one is refused.
        for (lo, hi) in [(3u32, 6u32), (0, 2), (5, 6), (0, 6)] {
            let cut = range_record(&bytes, lo, hi).expect("a range");
            assert_eq!(
                decode_range(&cut, L1),
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
            decode_range(&c, L1)
        };
        assert_eq!(refused(0, TILE_FORM), None);
        assert_eq!(refused(2, 0), None);
        assert_eq!(refused(2, 4), None);
        assert_eq!(refused(3, 4), None);
        assert_eq!(refused(3, 17), None);
        assert_eq!(decode_range(&cut[..3], L1), None);
        assert_eq!(decode_range(&cut[..cut.len() - 1], L1), None);
        // ★ A RECORD SAYS ITS LEVEL BY WHO READS IT (ruling B95): the 128 m record is refused as a 16 m one, and a 16 m
        // record (two means) comes back as a 16 m one and is refused as a 128 m one.
        assert_eq!(decode_levels(&bytes, FineLevel::L2), None);
        assert_eq!(decode_range(&cut, FineLevel::L2), None);
        let l2 = encode_levels(&levels[..3], 5).expect("codes");
        assert_eq!(
            decode_levels(&l2, FineLevel::L2),
            Some((5, levels[..3].to_vec()))
        );
        assert_eq!(decode_levels(&l2, L1), None);
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
        let geom = l1_geom(2, n7);
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
        assert_eq!(decode_raw(&record, key, geom), Some(field));
        // Another form, a shift past the ladder, bytes that end early.
        let mut other = record.clone();
        other[0] = RAW_FORM + 1;
        assert_eq!(decode_raw(&other, key, geom), None);
        other[0] = RAW_FORM;
        other[1] = 17;
        assert_eq!(decode_raw(&other, key, geom), None);
        assert_eq!(decode_raw(&record[..1], key, geom), None);
        assert_eq!(decode_raw(&record[..record.len() / 2], key, geom), None);
        // A step past the coder's plain word.
        let cliff = Field {
            gx0: -4,
            gy0: -4,
            w: 2,
            h: 1,
            v: vec![0, 1 << 24],
        };
        assert_eq!(encode_raw(&cliff, 2), None);
        // A record of another level's cells (another shift) is not this level's.
        assert_eq!(decode_raw(&record, key, l1_geom(3, n7)), None);
        // The input word: the same artifact the same word, another artifact another; the 128 m level's word is the
        // plain one, the 16 m level's its own.
        assert_eq!(input_word([1, 2]), input_word([1, 2]));
        assert_ne!(input_word([1, 2]), input_word([1, 3]));
        assert_eq!(input_word_of(L1, [1, 2]), input_word([1, 2]));
        assert_ne!(input_word_of(FineLevel::L2, [1, 2]), input_word([1, 2]));
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
            raw_field(key, l1_geom(2, 96), 40, quarters)
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
        assert_eq!(finish(std::array::from_fn(slot), l1_geom(2, 96)), None);
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
            raw_field(key, l1_geom(4, 384), 160, quarters)
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
        let levels = finish(std::array::from_fn(slot), l1_geom(4, 384)).expect("finishes");
        assert_eq!(levels.len(), L1.mean_levels() as usize + 1);
        assert_eq!(levels[0][0], ground(128, 128));
        assert_eq!(levels[0][127 * 128 + 127], ground(255, 255));
        assert_eq!(levels[5].len(), 16);
    }

    /// ★ THE 16 m LEVEL FINISHES BY THE SAME CODE (ruling B95): a node of 128 cells is the whole tile, its raw raster an
    /// eighth of a node (16 cells) past it; nine equal tiles finish into their own ground with the level's two means.
    #[test]
    fn a_16m_tile_is_one_node_and_nine_equal_ones_finish_into_their_own_ground() {
        let geom = LevelGeom::from_shift(FineLevel::L2, 7, 3).expect("a level");
        assert_eq!(
            (geom.side(), geom.margin, geom.band(), geom.n),
            (128, 16, 8, 384)
        );
        let ground = |x: i64, y: i64| (3 * x - 2 * y) as i32;
        let grid: Vec<Field> = (0..9)
            .map(|k| {
                let key = TileKey {
                    face: Face::PosX,
                    tx: k % 3,
                    ty: k / 3,
                };
                let (x0, y0) = (i64::from(key.tx) * 128 - 16, i64::from(key.ty) * 128 - 16);
                let quarters = (0..160 * 160)
                    .map(|c| ground(x0 + (c % 160), y0 + (c / 160)))
                    .collect();
                raw_field(key, geom, 160, quarters)
            })
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
        let levels = finish(std::array::from_fn(slot), geom).expect("finishes");
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0][0], ground(128, 128));
        assert_eq!(levels[2].len(), 32 * 32);
        // A node too small for a band over zero reads but does not finish; one under a margin's cell has no level.
        let small = LevelGeom::from_shift(FineLevel::L2, 6, 3).expect("a level");
        assert_eq!(small.band(), 0);
        let fields: Vec<Field> = (0..9)
            .map(|k| {
                let key = TileKey {
                    face: Face::PosX,
                    tx: k % 3,
                    ty: k / 3,
                };
                raw_field(key, small, 80, vec![0; 80 * 80])
            })
            .collect();
        let slot = |k: usize| {
            Some((
                TileKey {
                    face: Face::PosX,
                    tx: (k % 3) as i32,
                    ty: (k / 3) as i32,
                },
                &fields[k],
            ))
        };
        assert_eq!(finish(std::array::from_fn(slot), small), None);
        assert_eq!(LevelGeom::from_shift(FineLevel::L2, 2, 3), None);
        assert_eq!(LevelGeom::from_side(FineLevel::L1, 2, 3), None);
    }
}
