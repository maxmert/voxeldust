//! ★ THE FINE TILES' CODER (B1 (f); the way-in design revision 4 §4.4; ruling B83 item 4): THE ADAPTIVE RICE CODE of
//! JPEG-LS (Weinberger, Seroussi & Sapiro 2000), our own code. Each sample (a height in quarter metres) is predicted by
//! LOCO-I's MEDIAN rule from its decoded left `a`, upper `b` and upper-left `c` neighbours — `min(a, b)` when
//! `c ≥ max(a, b)`, `max(a, b)` when `c ≤ min(a, b)`, else `a + b − c` (the first row from the left, the first column
//! from above, the first sample from nought). The residual is zigzagged and written as its quotient by `2^k` in ones
//! ended by a zero and its `k` low bits; `k` is the least with `n·2^k ≥ A`, `A` the running sum of the residuals' size
//! and `n` their count, both halved every 64 samples. A quotient of [`ESCAPE`] or more writes the residual as a plain
//! [`RAW_BITS`]-bit word instead, so no sample costs more than 4 bytes. A tile decodes alone; integers only and no
//! division, the same bits on every host.
//!
//! MEASURED (B0 step 4, every level decoded back exactly): the belt's 128 m near tile 468 KB (1.17 bytes a sample),
//! within 4 % of its residuals' entropy.
//!
//! **Example.** A 640 × 640 near tile of the home planet's belt leaves the planet's shard as about 468 KB in 15 parts of
//! 32 KB; a client decodes it alone into the same 409 600 quarter metres.

/// A quotient at or past this many ones writes the residual as a plain [`RAW_BITS`]-bit word.
pub const ESCAPE: u64 = 8;
/// The plain word's bits: a zigzagged residual under 2^24 (a 2 000 km step in quarter metres).
pub const RAW_BITS: u32 = 24;

/// LOCO-I's median prediction of the sample at column `x`, row `y` of a raster `w` wide, from the samples before it.
#[must_use]
pub fn predict(q: &[i32], w: usize, x: usize, y: usize) -> i64 {
    let k = y * w + x;
    match (x, y) {
        (0, 0) => 0,
        (_, 0) => i64::from(q[k - 1]),
        (0, _) => i64::from(q[k - w]),
        _ => {
            let (a, b, c) = (i64::from(q[k - 1]), i64::from(q[k - w]), i64::from(q[k - w - 1]));
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if c >= hi {
                lo
            } else if c <= lo {
                hi
            } else {
                a + b - c
            }
        }
    }
}

/// The Rice parameter from the running statistics: the least `k` with `n·2^k ≥ a`.
fn rice_k(a: u64, n: u64) -> u32 {
    let mut k = 0u32;
    while (n << k) < a {
        k += 1;
    }
    k
}

/// The statistics' step after a residual `z`.
fn rice_update(a: &mut u64, n: &mut u64, z: u64) {
    *a += z;
    *n += 1;
    if *n == 64 {
        *a >>= 1;
        *n >>= 1;
    }
}

/// The bits written most significant first.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    count: u32,
}

impl Bits {
    fn put(&mut self, v: u64, bits: u32) {
        for i in (0..bits).rev() {
            self.acc = (self.acc << 1) | ((v >> i) & 1);
            self.count += 1;
            if self.count == 8 {
                self.out.push(self.acc as u8);
                self.acc = 0;
                self.count = 0;
            }
        }
    }
}

/// ★ THE ENCODER: `q`, a raster `w` wide and `h` high, row by row, as bytes. A residual whose zigzag passes
/// [`RAW_BITS`] cannot be written: `None` (a height step of 2 000 km is no ground).
#[must_use]
pub fn encode(q: &[i32], w: usize, h: usize) -> Option<Vec<u8>> {
    let mut bits = Bits { out: Vec::new(), acc: 0, count: 0 };
    let (mut a, mut n) = (4u64, 1u64);
    for y in 0..h {
        for x in 0..w {
            let r = i64::from(*q.get(y * w + x)?) - predict(q, w, x, y);
            let z = ((r << 1) ^ (r >> 63)) as u64;
            if z >> RAW_BITS != 0 {
                return None;
            }
            let kk = rice_k(a, n);
            let quot = z >> kk;
            if quot >= ESCAPE {
                bits.put((1u64 << ESCAPE) - 1, ESCAPE as u32);
                bits.put(z, RAW_BITS);
            } else {
                bits.put(((1u64 << quot) - 1) << 1, quot as u32 + 1);
                bits.put(z & ((1u64 << kk) - 1), kk);
            }
            rice_update(&mut a, &mut n, z);
        }
    }
    bits.put(0, 7);
    Some(bits.out)
}

/// ★ THE DECODER: a raster `w` wide and `h` high from `bytes`; `None` where the bytes end early.
#[must_use]
pub fn decode(bytes: &[u8], w: usize, h: usize) -> Option<Vec<i32>> {
    let mut q = vec![0i32; w * h];
    let mut bit = 0usize;
    let mut get = |count: u32| -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..count {
            let byte = *bytes.get(bit >> 3)?;
            v = (v << 1) | u64::from((byte >> (7 - (bit & 7))) & 1);
            bit += 1;
        }
        Some(v)
    };
    let (mut a, mut n) = (4u64, 1u64);
    for y in 0..h {
        for x in 0..w {
            let kk = rice_k(a, n);
            let mut quot = 0u64;
            while quot < ESCAPE && get(1)? == 1 {
                quot += 1;
            }
            let z = if quot == ESCAPE { get(RAW_BITS)? } else { (quot << kk) | get(kk)? };
            let r = ((z >> 1) as i64) ^ -((z & 1) as i64);
            q[y * w + x] = (predict(&q, w, x, y) + r) as i32;
            rice_update(&mut a, &mut n, z);
        }
    }
    Some(q)
}

#[cfg(test)]
mod tests {
    //! A test may divide (ruling F7's rule is about the shipped path): a fixture's pattern picks its cells with a
    //! remainder.
    #![allow(clippy::modulo_arithmetic, reason = "a fixture's pattern; never a kernel's path")]
    use super::*;

    #[test]
    fn a_raster_decodes_back_exactly_and_the_escape_carries_a_cliff() {
        // A slope with ridges, and a 1 km cliff (an escaped residual).
        let (w, h) = (40, 30);
        let mut q = Vec::with_capacity(w * h);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let ridge = if (x + y) % 7 == 0 { 120 } else { 0 };
                let cliff = if x >= 25 { 4_000 } else { 0 };
                q.push(3 * x - 2 * y + ridge + cliff);
            }
        }
        let bytes = encode(&q, w, h).expect("encodes");
        assert_eq!(decode(&bytes, w, h).expect("decodes"), q);
        assert!(bytes.len() < q.len() * 2, "{} bytes", bytes.len());
    }

    #[test]
    fn the_prediction_reads_its_three_cases_and_the_edges() {
        // c over both: the lesser; c under both: the greater; between: a + b − c.
        assert_eq!(predict(&[9, 5, 4, 0], 2, 1, 1), 4);
        assert_eq!(predict(&[1, 5, 4, 0], 2, 1, 1), 5);
        assert_eq!(predict(&[4, 5, 3, 0], 2, 1, 1), 4);
        assert_eq!(predict(&[7, 0, 0, 0], 2, 0, 0), 0);
        assert_eq!(predict(&[7, 0, 0, 0], 2, 1, 0), 7);
        assert_eq!(predict(&[7, 0, 0, 0], 2, 0, 1), 7);
    }

    #[test]
    fn a_step_too_large_for_the_plain_word_a_short_raster_and_short_bytes_refuse() {
        assert_eq!(encode(&[0, 1 << 24], 2, 1), None);
        assert_eq!(encode(&[0, 1, 2], 2, 2), None);
        let bytes = encode(&[5, 6, 7, 8], 2, 2).expect("encodes");
        assert_eq!(decode(&bytes[..0], 2, 2), None);
        assert_eq!(decode(&bytes, 2, 2), Some(vec![5, 6, 7, 8]));
    }

    #[test]
    fn the_statistics_halve_every_64_samples() {
        let q: Vec<i32> = (0..200).map(|i| (i * 37) % 101).collect();
        let bytes = encode(&q, 20, 10).expect("encodes");
        assert_eq!(decode(&bytes, 20, 10).expect("decodes"), q);
    }
}
