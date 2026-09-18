//! ★ THE CHUNK DIGEST and the golden self-check — the gate's unit of comparison. A digest folds every
//! cell's substance byte and gap byte in packing order, twice, from two offsets, into 128 bits. The
//! golden set (`tests/terrain_pin.rs`) pins the home planet's chunks at every rung; the self-check
//! folds eight of them at boot into the world tag's MEASURED half.
//!
//! **Example.** The client evaluates the eight self-check chunks at login and states the fold. The
//! gateway, which evaluated the same eight at boot, compares. A chip whose arithmetic drifted by one
//! ulp on one cell states a different number and is refused before it draws a hill.

use crate::body::{BodyDefinition, RADIUS_RECIP_BITS};
use crate::chunk::{ChunkKey, generate};
use crate::units::{LENGTH_BITS, STEPS_PER_M, greater, lesser, metres_of_q28};
use vd_recipe::Gi;
use vd_recipe::noise::{NOISE_BITS, NOISE_ONE};
use vd_seed::bend::Face;
use vd_seed::digest::{FNV_OFFSET, fnv1a};

/// A chunk's 128-bit digest: two FNV-1a folds over the same bytes from two offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkDigest(pub [u64; 2]);

/// The second fold's offset: the first offset with every bit flipped.
const SECOND_OFFSET: u64 = !FNV_OFFSET;

/// The digest of one chunk; `None` for a key outside the ladder.
#[must_use]
pub fn chunk_digest(body: &BodyDefinition, key: ChunkKey) -> Option<ChunkDigest> {
    Some(digest_of(&generate(body, key)?))
}

/// The digest of a generated chunk: its key, then its cells in packing order.
#[must_use]
pub fn digest_of(chunk: &crate::chunk::ChunkLattice) -> ChunkDigest {
    let key = chunk.key;
    let mut a = fnv1a(FNV_OFFSET, &[key.face as u8, key.rung]);
    let mut b = fnv1a(SECOND_OFFSET, &[key.face as u8, key.rung]);
    for coord in [key.x, key.y, key.z] {
        a = fnv1a(a, &coord.to_le_bytes());
        b = fnv1a(b, &coord.to_le_bytes());
    }
    for cell in &chunk.cells {
        let bytes = [cell.stratum.code(), cell.gap as u8];
        a = fnv1a(a, &bytes);
        b = fnv1a(b, &bytes);
    }
    ChunkDigest([a, b])
}

/// The digest of an extracted chunk (slice 6): its key, its vertex count, every vertex's three
/// integers and every triangle's three indices, little-endian, in order. Pins the triangle list —
/// a quad list cannot go red on the diagonal that matters.
#[must_use]
pub fn mesh_digest(mesh: &crate::extract::ChunkMesh) -> ChunkDigest {
    let key = mesh.key;
    let mut a = fnv1a(FNV_OFFSET, &[key.face as u8, key.rung]);
    let mut b = fnv1a(SECOND_OFFSET, &[key.face as u8, key.rung]);
    for coord in [key.x, key.y, key.z] {
        a = fnv1a(a, &coord.to_le_bytes());
        b = fnv1a(b, &coord.to_le_bytes());
    }
    let count = (mesh.vertices.len() as u32).to_le_bytes();
    a = fnv1a(a, &count);
    b = fnv1a(b, &count);
    for v in &mesh.vertices {
        for x in v {
            a = fnv1a(a, &x.to_le_bytes());
            b = fnv1a(b, &x.to_le_bytes());
        }
    }
    for t in &mesh.triangles {
        for i in t {
            a = fnv1a(a, &i.to_le_bytes());
            b = fnv1a(b, &i.to_le_bytes());
        }
    }
    ChunkDigest([a, b])
}

/// The eight self-check keys: one chunk per face at rung 0, near the surface, plus two at the top
/// rung. Stated as `(face, rung, x, y)`; the radial index is the surface chunk of that column.
pub const GOLDEN_SELF_CHECK_KEYS: [(Face, u8, i32, i32); 8] = [
    (Face::PosX, 0, 300, 700),
    (Face::NegX, 0, 1_181, 77),
    (Face::PosY, 0, 5, 5),
    (Face::NegY, 0, 2_000, 2_000),
    (Face::PosZ, 0, 999, 1),
    (Face::NegZ, 0, 40, 4_000),
    (Face::PosX, 255, 3, 3),
    (Face::NegY, 255, 1, 2),
];

/// The heights that sample a column: a grid of this many points along each edge (the corners, the
/// centre and the quarter points), so the ground between two samples is a quarter chunk apart and
/// the column bound below is sixteen times smaller than with the corners alone.
pub const COLUMN_SAMPLES_PER_EDGE: i32 = 5;
/// The GAPS between those samples, as a shift: four gaps, so the grid's spacing is a shift and not a
/// divide (ruling F7). Asserted against the sample count, so the two can never drift apart.
pub const SAMPLE_GAPS_LOG2: u32 = 2;
const _: () = assert!(1 << SAMPLE_GAPS_LOG2 == COLUMN_SAMPLES_PER_EDGE - 1);

/// THE COLUMN BOUND: how far the surface inside one chunk column can stand from the heights that
/// sample it, in metres — the recipe's own statement about itself, stated as a bound with a
/// margin and MEASURED below (`the_column_bound_holds_against_a_dense_sample`). One octave varies
/// between two samples `s` apart by at most its amplitude times `(π·s/λ)²/2` for a wave of period
/// `λ`; the surface is sampled on a grid, so the two axes' terms add (a bump between four samples
/// stands off both), which doubles it; and `λ` here is the octave's noise LATTICE CELL on the
/// surface (`radius / frequency`, the period of `noise3`'s lattice), which is half a wave's period
/// at most, so the term is four times a sinusoid's. Capped at the amplitude, summed over the
/// rung's live octaves, with `s` a quarter chunk. The span reads this instead of a whole chunk of
/// margin each way: MEASURED on the first ladder (slice 8 step 2), the margin made every column
/// three chunks tall, two of them empty, and the rung-0 disc cost 2 014 chunks for about 620
/// columns; with five samples the bound at the coarse rungs kept every far column "visible" over
/// the horizon, and the grid cut it.
///
/// ★ ON INTEGERS (ruling F7). The sampling ratio `s = π·edge/lattice` is `π · edge · frequency /
/// radius`: the frequency is the charter's own pair, the edge is an exact whole number of gap steps
/// (62 × cell × 128 ÷ 4 = 1 984 × cell), the divide by the radius is ONE multiply by the charter's
/// reciprocal, and π is [`PI`], `round(π · 2²⁸)`. The ratio is carried at [`BOUND_BITS`] fraction bits
/// so its SQUARE lands at the noise's own 28 with no shift; the amplitude then reads its share and the
/// sum is gap steps at [`LENGTH_BITS`].
///
/// ★ **A RIDGED OCTAVE IS BOUNDED BY ITS FIRST-ORDER TERM** (slice 8a stage 2;
/// `slice_8a_design.md` §2.2). The curvature argument above needs a SECOND derivative, and a ridged
/// octave has a KINK at `n = 0` — that is what a crest IS — so its second derivative is unbounded
/// there and the argument does not hold. Its FIRST derivative is bounded, and it is exactly twice
/// the smooth octave's (the transform's slope is `∓2` where the noise's is `±1`), so the term
/// becomes `min(1, 2·π·edge·f/R) × amplitude`. That is LARGER than the curvature term wherever the
/// curvature term is honest, and it is honest where the curvature term is not. Without it a surface
/// chunk's column span can be short and a chunk is missed — a hole, which is a defect.
///
/// **A worked number.** The home planet's 781 m octave at rung 0: `s = 0.0625`, so the smooth term
/// is `s² = 0.0039` of its 25.1 m amplitude (98 mm) and the ridged term is `2s = 0.125` of it
/// (3.14 m) — thirty-two times as much, which is `2/s`.
///
/// The choice is an ARITHMETIC MASK on the octave's own `kind` word, the same select the kernel
/// uses, so the two can never answer different questions about one octave.
///
/// ★ **A FINE OCTAVE CARRIES THE ROUGHNESS FACTOR'S OWN VARIATION** (slice 8a stage 3;
/// `slice_8a_design.md` §1.2 and §2.3). The amplitudes below are read at the factor's CEILING
/// (`m = 1`), which bounds the fine octaves' own VALUE — but the factor itself moves across a
/// chunk, and `m₁·F₁ − m₂·F₂ = m₁(F₁ − F₂) + F₂(m₁ − m₂)`. The first half is the term already
/// summed; the second is `Σ_fine a × |Δm|`, and it is added to every fine octave's share:
///
/// ```text
///    |Δm| ≤ (1 − m_min) · max|fade′| · |Δr| = (1 − m_min) · (15/16) · min(1, s_rough²)
/// ```
///
/// `max|fade′| = 30·t²(t−1)²` at `t = ½`, which is `15/8`, and `Δr` is HALF the field's own change
/// because the reading is `(n + 1)/2` — so the two make `15/16`. `s_rough` is the same sampling
/// ratio the octaves use, on the roughness field's own frequency. The field is CONTINENTAL (20 km
/// to 120 km), so at rung 0 the term is millimetres; at a coarse rung, where a chunk is kilometres
/// wide, it is real, and without it a surface chunk's span could be short — a hole, which is a
/// defect.
#[must_use]
pub fn column_bound(body: &BodyDefinition, rung: u8) -> Gi {
    // The sample grid's own spacing in gap steps: a chunk's width over the grid's gaps. The gap count
    // is a power of two ([`SAMPLE_GAPS_LOG2`]), so the division is a shift and the answer is exact
    // (62 × 128 ÷ 4 = 1 984 steps a metre-rung cell).
    let edge_steps = Gi::new(
        (crate::chunk::CHUNK_EDGE as i64 * i64::from(vd_seed::ladder::cell_m(rung)) * STEPS_PER_M)
            >> SAMPLE_GAPS_LOG2,
    );
    // ★ THE ROUGHNESS FACTOR'S OWN VARIATION over the same sample spacing, at the noise's bits: the
    // field's second-order share, capped at one, times `(1 − m_min) · 15/16`. A FINE octave adds it;
    // a coarse one does not, because the factor never touches a coarse octave.
    let rough = body.roughness();
    let drift = (curvature_share(body, edge_steps, &rough.octave)
        .mul_shr(NOISE_ONE - rough.m_min, NOISE_BITS)
        * Gi::new(FADE_SLOPE_NUM))
        >> FADE_SLOPE_LOG2;
    let first_fine = rough.first_fine.raw() as usize;
    let mut bound = Gi::ZERO;
    let live = body.octaves_at(rung);
    let mut k = 0usize;
    while k < live.len() {
        let o = &live[k];
        let frequency = (o.frequency_int << NOISE_BITS) + o.frequency_frac;
        // frequency × edge, then ÷ radius, then × π — each at BOUND_BITS.
        let product = frequency.mul_shr(edge_steps, NOISE_BITS - BOUND_BITS);
        let ratio = product.mul_shr(body.radius_recip, RADIUS_RECIP_BITS);
        let s = ratio.mul_shr(PI, NOISE_BITS);
        // Half of a wave's second-order term, times two axes: the two cancel, so the square stands.
        // `s` carries BOUND_BITS, so its square carries the noise's own bits with no shift.
        let curve = s * s;
        // A RIDGED octave's first-order term, `2s`, at the noise's bits: one more shift left than
        // the one that takes BOUND_BITS to NOISE_BITS.
        let first = s << (BOUND_BITS + 1);
        // The select, on the mask and never on a branch.
        let term = curve + (o.kind & (first - curve));
        // The factor's own drift, on a FINE octave only.
        let term = term + if k < first_fine { Gi::ZERO } else { drift };
        // min(1, term), branchless: one minus the positive part of (1 − term).
        let share = NOISE_ONE - greater(NOISE_ONE - term, Gi::ZERO);
        bound += (o.amplitude * share) >> NOISE_BITS;
        k += 1;
    }
    // The amplitudes carry AMP_BITS below a gap step; the bound leaves at the length format's bits.
    let bound = bound << (LENGTH_BITS - vd_recipe::height::AMP_BITS);
    // ★ THE CAP-ROCK BENCH AMPLIFIES IT (slice 8a stage 4). Everything above bounds how far the
    // surface can stand from the samples that measure it — a DIFFERENCE — and the terrace multiplies
    // every difference under it by at most `1 + (7/9)·strength` (`vd_terrain::body::TERRACE_LIP_NUM`,
    // re-derived for the quintic and LARGER than `04`'s cubic constant). One multiply at the end,
    // because the terrace is a function of the height alone and reads nothing about the column.
    // Without it a surface chunk's span can be short and a chunk is missed — a hole, which is a
    // defect.
    //
    // ★ THE BENCH ADDS THE LESSER OF ITS TWO HONEST BOUNDS. The terrace amplifies a difference by
    // `Lip − 1` of it; it also never carries ANY column further than its own pull bound, so a
    // difference of two columns cannot gain more than twice that. Both hold, and the cheaper one is
    // the one the span pays for: near the ground a chunk is narrow and the amplification is small,
    // while at a coarse rung a chunk is kilometres wide and twice the pull is the far smaller number.
    // `greater` against zero is the fence's own branchless lesser.
    let amplified = bound.mul_shr(body.lip(rung) - NOISE_ONE, NOISE_BITS);
    let capped = body.pull_bound(rung) << 1;
    let added = amplified - greater(amplified - capped, Gi::ZERO);
    bound + added + (body.terrace_slack() << 1)
}

/// ONE OCTAVE'S SECOND-ORDER SHARE over a sample spacing, capped at one, at [`NOISE_BITS`]: the
/// curvature argument [`column_bound`] states, on one octave's own frequency. The roughness field's
/// own drift reads it, and so could any other slow field a later slice adds.
fn curvature_share(body: &BodyDefinition, edge_steps: Gi, o: &vd_recipe::height::Octave) -> Gi {
    let frequency = (o.frequency_int << NOISE_BITS) + o.frequency_frac;
    let product = frequency.mul_shr(edge_steps, NOISE_BITS - BOUND_BITS);
    let ratio = product.mul_shr(body.radius_recip, RADIUS_RECIP_BITS);
    let s = ratio.mul_shr(PI, NOISE_BITS);
    let curve = s * s;
    NOISE_ONE - greater(NOISE_ONE - curve, Gi::ZERO)
}

/// `max|fade′| ÷ 2 = 15/16` as a numerator and a shift: the quintic `t³(t(6t − 15) + 10)` has slope
/// `30·t²(t − 1)²`, whose largest value is `15/8` at `t = ½`, and the roughness reading is HALF the
/// field, so the two make fifteen sixteenths. A numerator and a shift, never a divide (ruling F7).
const FADE_SLOPE_NUM: i64 = 15;
const FADE_SLOPE_LOG2: u32 = 4;

/// The sampling ratio's fraction bits: half the noise's, so the ratio's SQUARE lands at the noise's
/// own without a shift.
pub const BOUND_BITS: u32 = NOISE_BITS >> 1;
const _: () = assert!(2 * BOUND_BITS == NOISE_BITS);

/// `round(π · 2²⁸)` — the one transcendental constant the recipe holds, as a word. Derived from the
/// standard library's own `PI` and pinned by this module's test.
pub const PI: Gi = Gi::new(843_314_857);

/// THE COLUMN BOUND in metres, for a host outside the recipe.
#[must_use]
pub fn column_bound_m(body: &BodyDefinition, rung: u8) -> f64 {
    metres_of_q28(column_bound(body, rung))
}

/// What one chunk column holds along the radial: the chunk span of its surface and the surface's
/// highest point, as the five heights and the column bound state them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnSpan {
    /// The lowest and the highest chunk index that hold the surface, clamped to the band.
    pub lo: i32,
    pub hi: i32,
    /// The lowest and the highest SAMPLED radius, in metres (the grid's own extrema, no bound). The
    /// recipe decides them as words and states them here at the seam, because the ladder view that
    /// reads them is outside the recipe.
    pub sampled_low_m: f64,
    pub sampled_high_m: f64,
    /// The surface's highest radius inside the column, in metres, AT RUNG 0: the highest sample
    /// plus the column bound plus the dropped octaves' bound (a coarse rung's field lies under the
    /// true peak by up to the amplitudes it dropped — the refuter's finding: a ridge 14 km up read
    /// 2 km at the top rung and its whole subtree was culled behind the horizon). What a ladder
    /// view tests against the sightline's drop past the horizon.
    pub peak_m: f64,
}

/// THE SURFACE SPAN of chunk column `(x, y)` of `face` at a rung: the lowest and the highest chunk
/// index along the radial that hold the surface at the column's sample grid, widened by the column
/// bound (above) each way, clamped to the band. Twenty-five heights, not a column pass — a client
/// asks this per wanted column once and keeps it. A cave mouth under the surface chunk is what
/// slice 8's residency band is for.
#[must_use]
pub fn surface_chunk_span(
    body: &BodyDefinition,
    face: Face,
    rung: u8,
    x: i32,
    y: i32,
) -> (i32, i32) {
    let span = surface_column(body, face, rung, x, y);
    (span.lo, span.hi)
}

/// The span AND the peak of a column (see [`surface_chunk_span`]).
#[must_use]
pub fn surface_column(body: &BodyDefinition, face: Face, rung: u8, x: i32, y: i32) -> ColumnSpan {
    let edge = crate::chunk::CHUNK_EDGE as i32;
    let key = ChunkKey {
        face,
        rung,
        x,
        y,
        z: 0,
    };
    let floor = i64::from(body.ladder.floor_m);
    let top = top_chunk_z(body, rung);
    let bound = column_bound(body, rung);
    let dropped = body.dropped_bound(rung);
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    let mut low = Gi::ZERO;
    let mut high = Gi::ZERO;
    let step = (edge - 1) >> SAMPLE_GAPS_LOG2;
    let mut i = 0;
    while i < COLUMN_SAMPLES_PER_EDGE {
        let mut j = 0;
        while j < COLUMN_SAMPLES_PER_EDGE {
            let site = crate::lattice::site_of(body, key, i * step, j * step);
            let dir = crate::lattice::site_dir(body, key, site);
            let h = crate::height::height(body, dir, rung);
            // One cell of margin at the bottom: the extractor gives an edge to the chunk that owns
            // its LOWER cell, so a crossing of a chunk's bottom boundary edge is drawn by the chunk
            // BELOW it; a surface whose low bound lands in a chunk's first cell may cross exactly
            // there. (The top boundary edge is the chunk's own: no margin above.)
            // The cell and the chunk the surface lands in. The cell's width is a power of two metres
            // (a shift); a CHUNK is 62 cells, which no shift divides — and this is a CPU-ONLY
            // bookkeeping step (the client's wanted set, never a GPU kernel: the G1 cell field reads
            // `z` as given), integer-exact on every host, so the one `/` stands, named.
            #[allow(
                clippy::integer_division,
                reason = "CPU-only: a chunk is 62 cells, no power of two; never on a GPU kernel's path"
            )]
            let z_lo = (((metres_floor(h - bound) - floor) >> rung) - 1) / i64::from(edge);
            #[allow(
                clippy::integer_division,
                reason = "CPU-only: a chunk is 62 cells, no power of two; never on a GPU kernel's path"
            )]
            let z_hi = ((metres_floor(h + bound) - floor) >> rung) / i64::from(edge);
            let (z_lo, z_hi) = (z_lo as i32, z_hi as i32);
            lo = lo.min(z_lo);
            hi = hi.max(z_hi);
            // The extrema, from the FIRST sample's own surface, never from a sentinel. (MEASURED
            // before this: the low was taken as `low − max(low − h, 0)` from the largest float,
            // which every subtraction absorbed — a column's floor read 6.9 km, and nothing had read
            // it yet.)
            if (i == 0) & (j == 0) {
                low = h;
                high = h;
            } else {
                high = greater(high, h);
                low = lesser(low, h);
            }
            j += 1;
        }
        i += 1;
    }
    ColumnSpan {
        lo: lo.clamp(0, top),
        hi: hi.clamp(0, top),
        sampled_low_m: metres_of_q28(low),
        sampled_high_m: metres_of_q28(high),
        peak_m: metres_of_q28(high + bound + dropped),
    }
}

/// A length in gap steps at [`LENGTH_BITS`], floored to whole METRES: one arithmetic shift, which
/// floors on both sides of zero.
#[must_use]
fn metres_floor(length: Gi) -> i64 {
    (length >> (LENGTH_BITS + STEPS_PER_M.trailing_zeros())).raw()
}

/// The highest chunk index along the radial of a rung's band: the last chunk a column can hold.
#[must_use]
#[allow(
    clippy::integer_division,
    reason = "CPU-only: a chunk is 62 cells, no power of two; never on a GPU kernel's path"
)]
pub fn top_chunk_z(body: &BodyDefinition, rung: u8) -> i32 {
    (body.ladder.cells_in_band(rung) as i32 - 1) / crate::chunk::CHUNK_EDGE as i32
}

/// The chunk index along the radial that holds the SURFACE at the centre column of chunk `(x, y)`
/// of `face` at a rung: the chunk the extractor has work in.
#[must_use]
pub fn surface_chunk_z(body: &BodyDefinition, face: Face, rung: u8, x: i32, y: i32) -> i32 {
    let edge = crate::chunk::CHUNK_EDGE as i32;
    let key = ChunkKey {
        face,
        rung,
        x,
        y,
        z: 0,
    };
    let site = crate::lattice::site_of(body, key, edge >> 1, edge >> 1);
    let dir = crate::lattice::site_dir(body, key, site);
    let h = crate::height::height(body, dir, rung);
    // The cell index is the rung's shift; the chunk index is the one `/` by 62 this module keeps.
    let k = (metres_floor(h) - i64::from(body.ladder.floor_m)) >> rung;
    #[allow(
        clippy::integer_division,
        reason = "CPU-only: a chunk is 62 cells, no power of two; never on a GPU kernel's path"
    )]
    let z = k / i64::from(edge);
    z as i32
}

/// The key a self-check entry names on this body: rung 255 means the body's top rung, and the
/// face coordinates are wrapped into the rung's face so every body, however small, has eight.
#[must_use]
pub fn self_check_key(body: &BodyDefinition, entry: (Face, u8, i32, i32)) -> ChunkKey {
    let rung = if entry.1 == 255 {
        body.ladder.rungs - 1
    } else {
        entry.1
    };
    // CPU-ONLY, and never on a kernel's path: the self-check's eight keys are WRAPPED into whatever
    // face the body has, so a three-kilometre rock has eight chunks to fold as well. A chunk is 62
    // cells, which no shift divides.
    #[allow(
        clippy::integer_division,
        reason = "CPU-only: a chunk is 62 cells, no power of two; the self-check's key wrap"
    )]
    let chunks_per_edge =
        (body.ladder.cells_per_edge(rung) as i32 / crate::chunk::CHUNK_EDGE as i32).max(1);
    #[allow(
        clippy::modulo_arithmetic,
        reason = "CPU-only: the self-check's key wrap into a small body's face"
    )]
    let (x, y) = (entry.2 % chunks_per_edge, entry.3 % chunks_per_edge);
    ChunkKey {
        face: entry.0,
        rung,
        x,
        y,
        z: surface_chunk_z(body, entry.0, rung, x, y),
    }
}

/// The boot self-check: the fold of the eight self-check chunks' CELL digests and, since slice 6,
/// their MESH digests (the extractor, the halo, the seams and the corner are part of the shape a
/// client must compute identically — SL10 clauses 3 and 5), the world tag's MEASURED half.
/// `None` when a self-check key names no chunk of this body — a REFUSAL, never a fold of zeros
/// (the decode-to-Default ban, in the one place whose job is to refuse).
#[must_use]
pub fn golden_self_check(body: &BodyDefinition) -> Option<u64> {
    let mut acc = FNV_OFFSET;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(body, entry);
        let d = chunk_digest(body, key)?.0;
        acc = fnv1a(acc, &d[0].to_le_bytes());
        acc = fnv1a(acc, &d[1].to_le_bytes());
        let samples = crate::lattice::sample_box(body, key)?;
        let m = mesh_digest(&crate::extract::extract(&samples)).0;
        acc = fnv1a(acc, &m[0].to_le_bytes());
        acc = fnv1a(acc, &m[1].to_le_bytes());
    }
    Some(acc)
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
    use crate::home::home_planet;

    /// ★ A RIDGED OCTAVE WIDENS THE COLUMN BOUND (slice 8a stage 2; `slice_8a_design.md` §2.2).
    ///
    /// A ridged octave has a KINK at `n = 0` — that is what a crest IS — so the curvature argument
    /// the smooth bound rests on does not hold for it, and the bound must fall back on the FIRST
    /// derivative, which is twice the noise's. The test states the whole bound from first principles
    /// in real numbers — `s = π · edge / λ`, `min(1, s²)` for a smooth octave and `min(1, 2s)` for a
    /// ridged one, times the amplitude — and compares it with the shipped word.
    ///
    /// RED before the stage: every octave took the curvature term, so the home planet's bound at
    /// rung 0 stood at the smooth number and a chunk whose crest left the column's span was missed —
    /// a hole, which is a defect.
    #[test]
    fn a_ridged_octave_widens_the_column_bound() {
        let m = home_planet();
        // THE HOME PLANET HAS CRESTS: the band in metres selects five octaves, 12.5 km down to
        // 781 m — read from the body's own charter, never from the draw.
        let live = m.octaves_at(0);
        let radius_m = m.radius_m();
        let mut ridged_count = 0;
        for o in live {
            let lambda = radius_m / crate::body::octave_frequency(o);
            let in_band = (lambda >= crate::body::RIDGE_LO_M as f64)
                & (lambda <= crate::body::RIDGE_HI_M as f64);
            assert_eq!(
                o.kind == vd_recipe::height::OCTAVE_RIDGED,
                in_band,
                "octave of {lambda} m"
            );
            ridged_count += i32::from(in_band);
        }
        assert_eq!(
            ridged_count, 5,
            "octaves 5..=9 are crests on the home planet"
        );

        // THE SAME BODY WITH EVERY OCTAVE SMOOTH: the ONLY difference is the kind word, so what
        // separates the two bounds is the transform and nothing else.
        let mut plain = m;
        let mut o = 0;
        while o < plain.octaves.len() {
            plain.octaves[o].kind = vd_recipe::height::OCTAVE_SMOOTH;
            o += 1;
        }

        // The bound stated in real numbers, at a stated rung: the sum, and HOW MANY of its terms
        // stood at the ceiling. The count is read below, because a ceiling no octave ever reaches
        // is a line nobody walks.
        let want_m = |body: &BodyDefinition, rung: u8| -> (f64, usize) {
            let edge_m = f64::from(crate::chunk::CHUNK_EDGE as u32)
                * f64::from(vd_seed::ladder::cell_m(rung))
                / f64::from(1u32 << SAMPLE_GAPS_LOG2);
            let mut sum = 0.0;
            let mut at_ceiling = 0usize;
            for o in body.octaves_at(rung) {
                let lambda = radius_m / crate::body::octave_frequency(o);
                let s = std::f64::consts::PI * edge_m / lambda;
                let term = if o.kind == vd_recipe::height::OCTAVE_RIDGED {
                    2.0 * s
                } else {
                    s * s
                };
                // The cap is WRITTEN AS A COMPARISON: the fence bans `f64::min` (it is documented
                // non-deterministic on +0.0 against −0.0), even in a test.
                let capped = if term > 1.0 {
                    at_ceiling += 1;
                    1.0
                } else {
                    term
                };
                sum += crate::body::octave_amplitude_m(o) * capped;
            }
            // ★ THE CAP-ROCK BENCH AMPLIFIES THE WHOLE SUM (slice 8a stage 4): the column bound is a
            // bound on a DIFFERENCE, and the terrace multiplies every difference under it by its own
            // rung's Lipschitz word, then stands its own rounding slack on top — twice, because a
            // difference reads two columns.
            let lip = crate::units::share_of_q28(body.lip(rung));
            let slack = 2.0 * metres_of_q28(body.terrace_slack());
            (sum * lip + slack, at_ceiling)
        };

        // Rung 0, where the five crests are live and none of them is capped: the ridged bound is
        // STRICTLY wider, and both stand where the real-number statement puts them.
        let ridged_m = column_bound_m(&m, 0);
        let smooth_m = column_bound_m(&plain, 0);
        assert!(
            ridged_m > smooth_m,
            "the crests widen the bound: {ridged_m} m against {smooth_m} m"
        );
        // The shipped word stands AT OR UNDER the real-number statement and within one part in
        // fifty of it. Under, because every step of the integer path truncates toward zero; within
        // one part in fifty, because the sampling ratio is carried at [`BOUND_BITS`] = 14 fraction
        // bits and the coarsest crest's own ratio is only 64 of those units. A bound that drifted
        // far from the statement would say the two are no longer the same quantity.
        let (want_ridged, ridged_at_ceiling) = want_m(&m, 0);
        let (want_smooth, smooth_at_ceiling) = want_m(&plain, 0);
        // The doc comment's claim, stated as a measurement: at rung 0 a chunk is narrower than every
        // live octave, so no term stands at its ceiling and the bound is the whole sum.
        assert_eq!(ridged_at_ceiling, 0, "a crest caps at rung 0");
        assert_eq!(smooth_at_ceiling, 0, "a smooth octave caps at rung 0");
        assert!(
            ridged_m <= want_ridged,
            "the ridged bound {ridged_m} m over its statement {want_ridged} m"
        );
        assert!(
            ridged_m > want_ridged * 0.97,
            "the ridged bound {ridged_m} m against {want_ridged} m"
        );
        assert!(
            smooth_m <= want_smooth,
            "the smooth bound {smooth_m} m over its statement {want_smooth} m"
        );
        assert!(
            smooth_m > want_smooth * 0.98,
            "the smooth bound {smooth_m} m against {want_smooth} m"
        );

        // ★ THE WORKED NUMBER the doc comment states: the 781 m octave at rung 0 alone. Its sample
        // spacing is a quarter chunk — 15.5 m at rung 0 — so `s = π · 15.5 / 781.25 = 0.0623`, the
        // smooth term is `s² = 0.0039` of its amplitude and the ridged term is `2s = 0.1247` of it.
        // The ratio is `2/s`, which is thirty-two.
        let finest_crest = live
            .iter()
            .rfind(|o| o.kind == vd_recipe::height::OCTAVE_RIDGED)
            .expect("a crest");
        let lambda = radius_m / crate::body::octave_frequency(finest_crest);
        assert!((lambda - 781.25).abs() < 0.5, "{lambda} m");
        let edge_m = f64::from(crate::chunk::CHUNK_EDGE as u32) / 4.0;
        let ratio = 2.0 / (std::f64::consts::PI * edge_m / lambda);
        assert!((ratio - 32.1).abs() < 0.2, "the first order buys {ratio}×");

        // The bound still NESTS and still stands under the live amplitudes at every rung, which is
        // the promise the span reads it for.
        //
        // ★ AND THE CEILING IS REAL GROUND, NOT A LINE NOBODY WALKS. At a COARSE rung the sample
        // spacing is wider than the octave's own wavelength — at the home planet's top rung a chunk
        // is four thousand kilometres and the longest wave is four hundred — so `s` passes one and
        // the term stands at its ceiling. The loop counts every capped term over every rung, and the
        // count below states that the coarse rungs meet them.
        let mut at_ceiling = 0usize;
        let mut rung = 0u8;
        while rung < m.ladder.rungs {
            at_ceiling += want_m(&m, rung).1;
            assert!(
                column_bound(&m, rung) <= m.relief_bound(rung),
                "rung {rung}: the bound leaves the amplitudes"
            );
            assert!(
                column_bound(&m, rung) >= column_bound(&plain, rung),
                "rung {rung}"
            );
            rung += 1;
        }
        assert!(
            at_ceiling > 0,
            "no octave of any rung stands at the bound's ceiling: the cap is never taken"
        );
    }

    /// The span holds the centre's surface chunk, widened by the column bound, stays inside the
    /// band at the top rung (where one chunk is the whole band), and widens on a column whose
    /// corners sit in other chunks than its centre.
    #[test]
    fn the_surface_span_holds_the_centres_chunk_within_the_bound_and_stays_in_the_band() {
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let (lo, hi) = surface_chunk_span(&m, Face::PosX, 0, 300, 700);
        assert!(lo <= z, "{lo} {z}");
        assert!(hi >= z, "{hi} {z}");
        assert!(lo >= 0);
        // The bound: metres at rung 0 (the finest octaves whole, the coarse ones a hair), under the
        // sum of every live amplitude, and larger at a coarser rung where a chunk spans more ground
        // — while the coarser rung's own dropped octaves leave it, so it stays under the total.
        let b0 = column_bound(&m, 0);
        let total0 = m.relief_bound(0);
        assert!(b0 > Gi::ZERO);
        assert!(b0 < total0, "{b0:?} vs {total0:?}");
        let b9 = column_bound(&m, 9);
        assert!(b9 > b0, "{b9:?} vs {b0:?}");
        let total9 = m.relief_bound(9);
        assert!(b9 <= total9, "{b9:?} vs {total9:?}");
        assert_eq!(column_bound_m(&m, 0), metres_of_q28(b0));
        // The peak stands at or above every sample plus the bound: at least the centre's height.
        let span = surface_column(&m, Face::PosX, 0, 300, 700);
        assert_eq!((span.lo, span.hi), (lo, hi));
        let key = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 300,
            y: 700,
            z: 0,
        };
        let edge = crate::chunk::CHUNK_EDGE as i32;
        let site = crate::lattice::site_of(&m, key, edge / 2, edge / 2);
        let dir = crate::lattice::site_dir(&m, key, site);
        let centre_h = metres_of_q28(crate::height::height(&m, dir, 0));
        assert!(
            span.peak_m >= centre_h + metres_of_q28(b0),
            "{span:?} vs {centre_h}"
        );
        assert!(span.sampled_low_m <= centre_h);
        assert!(span.sampled_high_m >= centre_h);
        // At a coarse rung the peak carries the dropped octaves' bound too: it stands at least the
        // rung-0 relief bound over the sampled high.
        let coarse = surface_column(&m, Face::PosX, 9, 3, 5);
        assert!(
            coarse.peak_m >= coarse.sampled_high_m + metres_of_q28(m.dropped_bound(9)),
            "{coarse:?}"
        );
        // A column of one chunk exists at rung 0 now that the margin is the bound, not a chunk.
        //
        // ★ AND AT RUNG 0 IT NO LONGER DOES — THE CAP-ROCK BENCH'S PRICE, MEASURED (slice 8a stage
        // 4). The terrace amplifies every variation under it by 1.4358, so the rung-0 column bound
        // stands at 32.98 m where it stood at 22.97 m, and TWICE it — 65.96 m — is over a 62-cell
        // chunk. No rung-0 column can fit in one chunk any more: over 400 columns of face `+X` the
        // spans read 0 of one chunk, 338 of two and 62 of three or more. The claim therefore moves
        // to a rung the bench does not reach, where it still holds.
        let mut single = 0;
        for x in 300..340 {
            let (lo, hi) = surface_chunk_span(&m, Face::PosX, 6, x, 700);
            single += i32::from(lo == hi);
        }
        assert!(
            single > 0,
            "a one-chunk column at a rung the bench leaves alone"
        );
        let mut wide = 0;
        for x in 300..340 {
            let (lo, hi) = surface_chunk_span(&m, Face::PosX, 0, x, 700);
            wide += i32::from(hi > lo);
        }
        assert_eq!(wide, 40, "every rung-0 column spans two chunks or more");
        // The top rung: the band is one chunk, so the span is (0, 0) whatever the heights.
        let top = m.ladder().rungs - 1;
        assert_eq!(surface_chunk_span(&m, Face::PosX, top, 3, 3), (0, 0));
        // Every column of the golden set: the span holds the golden surface chunk.
        for entry in crate::GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&m, entry);
            let (lo, hi) = surface_chunk_span(&m, key.face, key.rung, key.x, key.y);
            assert!(lo <= key.z, "{key:?}: {lo}");
            assert!(hi >= key.z, "{key:?}: {hi}");
        }
        // A column whose corners disagree with its centre widens the span: measured over the
        // columns near the golden +X chunk, at least one span is two chunks or more.
        let mut widest = 0;
        for x in 300..340 {
            let (lo, hi) = surface_chunk_span(&m, Face::PosX, 0, x, 700);
            widest = widest.max(hi - lo);
        }
        assert!(widest >= 1, "{widest}");
    }

    /// The one transcendental constant the recipe holds is the standard library's own π, rounded once
    /// to the noise's fraction bits — stated here so a mistyped digit is a red test, not a moon.
    #[test]
    fn the_recipes_pi_is_the_standard_librarys_pi_at_the_noises_bits() {
        let one = f64::from(1u32 << NOISE_BITS);
        assert_eq!(PI.raw(), (std::f64::consts::PI * one).round() as i64);
        assert_eq!(BOUND_BITS, 14);
    }

    /// THE BOUND, MEASURED: over columns near the golden +X chunk at rung 0 and at rung 9, a dense
    /// sample of the column (every cell on a 16 × 16 grid) never leaves the sampled extrema widened
    /// by the column bound — the bound is a bound, not an argument (the refuter's finding).
    #[test]
    fn the_column_bound_holds_against_a_dense_sample() {
        let m = home_planet();
        let edge = crate::chunk::CHUNK_EDGE as i32;
        for (rung, xs) in [(0u8, 300..316), (9u8, 3..7)] {
            let bound = column_bound_m(&m, rung);
            for x in xs {
                let span = surface_column(&m, Face::PosX, rung, x, 700 >> rung);
                let key = ChunkKey {
                    face: Face::PosX,
                    rung,
                    x,
                    y: 700 >> rung,
                    z: 0,
                };
                let mut a = 0;
                let mut dense_low = f64::MAX;
                let mut dense_high = f64::MIN;
                while a < edge {
                    let mut b = 0;
                    while b < edge {
                        let site = crate::lattice::site_of(&m, key, a, b);
                        let dir = crate::lattice::site_dir(&m, key, site);
                        let h = metres_of_q28(crate::height::height(&m, dir, rung));
                        assert!(
                            h >= span.sampled_low_m - bound,
                            "rung {rung} column {x} cell ({a}, {b}): {h} under {span:?} - {bound}"
                        );
                        assert!(
                            h <= span.sampled_high_m + bound,
                            "rung {rung} column {x} cell ({a}, {b}): {h} over {span:?} + {bound}"
                        );
                        if h < dense_low {
                            dense_low = h;
                        }
                        if h > dense_high {
                            dense_high = h;
                        }
                        b += 4;
                    }
                    a += 4;
                }
                // The sampled extrema are heights of the column, so each lies within the bound
                // of the dense extremum on its side — never a sentinel, never absorbed.
                assert!(
                    (span.sampled_low_m >= dense_low - bound)
                        & (span.sampled_low_m <= dense_low + bound),
                    "rung {rung} column {x}: sampled low {:?} against the dense low {dense_low} ± {bound}",
                    span.sampled_low_m
                );
                assert!(
                    (span.sampled_high_m >= dense_high - bound)
                        & (span.sampled_high_m <= dense_high + bound),
                    "rung {rung} column {x}: sampled high {:?} against the dense high {dense_high} ± {bound}",
                    span.sampled_high_m
                );
                assert!(span.sampled_low_m <= span.sampled_high_m);
            }
        }
    }

    #[test]
    fn a_digest_is_the_chunks_bytes_and_moves_when_one_cell_or_the_key_moves() {
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let key = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 300,
            y: 700,
            z,
        };
        let d = chunk_digest(&m, key).expect("in the ladder");
        assert_eq!(
            chunk_digest(&m, key),
            Some(d),
            "the same bytes, the same digest"
        );
        let beside = ChunkKey { x: 301, ..key };
        assert_ne!(chunk_digest(&m, beside), Some(d));
        assert_ne!(d.0[0], d.0[1], "two folds, two offsets");
        assert_eq!(
            chunk_digest(&m, ChunkKey { rung: 99, ..key }),
            None,
            "outside the ladder is refused"
        );
        // One cell changed: the digest moves.
        let mut chunk = generate(&m, key).expect("in the ladder");
        chunk.cells[7].gap = chunk.cells[7].gap.wrapping_add(1);
        assert_ne!(digest_of(&chunk), d);
        // A second body (the home planet's seed plus one): every digest differs.
        let other = BodyDefinition::from_seed(
            m.seed + 1,
            f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS),
            crate::home::home_facts(),
        )
        .expect("on the ladder");
        assert_ne!(chunk_digest(&other, key), Some(d));
    }

    #[test]
    fn the_self_check_has_eight_keys_inside_every_body_and_folds_to_one_number() {
        let m = home_planet();
        for entry in GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&m, entry);
            assert!(key.rung < m.ladder.rungs);
            assert!(
                chunk_digest(&m, key).is_some(),
                "{entry:?} names a chunk of the home planet"
            );
        }
        let keys: std::collections::BTreeSet<ChunkKey> = GOLDEN_SELF_CHECK_KEYS
            .iter()
            .map(|e| self_check_key(&m, *e))
            .collect();
        assert_eq!(keys.len(), 8, "eight distinct chunks");
        let check = golden_self_check(&m).expect("the home planet self-checks");
        assert_eq!(golden_self_check(&m), Some(check));
        // A tiny unnamed test body: the keys wrap into it, and its check differs.
        let rock = BodyDefinition::from_seed(5, 3_000.0, crate::body::ROCK_3KM_FACTS)
            .expect("a test body of 3 km");
        for entry in GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&rock, entry);
            assert!(
                chunk_digest(&rock, key).is_some(),
                "{entry:?} wraps into a tiny body"
            );
        }
        assert_ne!(golden_self_check(&rock), Some(check));
        let z0 = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let z3 = surface_chunk_z(&m, Face::PosX, 3, 37, 87);
        assert!(
            (z0 / 8 - z3).abs() <= 1,
            "the surface chunk scales with the rung: {z0} vs {z3}"
        );
    }
}
