//! ★ THE SEATING RULE (ruling V10 S6-7): where a sub-metre block sits inside its cell on a slope,
//! in eighths of a cell above the cell's floor — the one rule both hosts drop a lamp post by.
//!
//! The seat is the height where the gap crosses zero along the cell's own radial, from the cell's
//! gap byte and its neighbour's above or below, by the same division the extractor uses, snapped to
//! the sub-metre grid's step (1/8 cell). In exact integer arithmetic:
//!
//! - the cell's centre is ROCK and the cell above is air: the crossing is `t = |g| / (|g| + g_up)`
//!   above the centre, the seat is `½ + t` cells; a seat past the cell's top means the ground is
//!   above this cell — the placement is REFUSED ("solid ground, mine first");
//! - the cell's centre is rock and so is the one above: refused;
//! - the cell's centre is AIR and the cell below is rock: the crossing is `t = |g_down| /
//!   (|g_down| + g)` above the centre below, the seat is `t − ½` cells, and a seat under the floor
//!   is the floor;
//! - air over air: the floor.
//!
//! **Example.** A player sets a 25 cm lamp post on a 30° slope. The cell's gap is `−0.13` cell
//! (`−17`) and the cell above reads `+0.87` (`+111`): `t = 17 / 128`, the seat is `0.5 + 0.133 =
//! 0.633` cells, snapped to `5/8`. Her client draws the post at 5/8; the planet's shard collides it
//! at 5/8. Nobody stored a height.

use crate::extract::is_rock;
use crate::lattice::SampleBox;

/// Seat steps per cell.
pub const SEAT_STEPS: i32 = 8;

/// The seat of a sub-metre block on the chunk's own cell `(a, b, c)`, in eighths of a cell above
/// the cell's floor (`0..=8`); `None` refuses (the cell is buried, or the address is not one of
/// the chunk's own `62³` cells).
#[must_use]
pub fn seat_eighths(samples: &SampleBox, cell: [u8; 3]) -> Option<u8> {
    let (a, b, c) = (i32::from(cell[0]), i32::from(cell[1]), i32::from(cell[2]));
    // Only the chunk's own cells have a seat; a cell past them names nothing (never a panic).
    let own = 0..=(crate::chunk::CHUNK_EDGE as i32 - 1);
    if !(own.contains(&a) & own.contains(&b) & own.contains(&c)) {
        return None;
    }
    let g = i32::from(samples.cell(a, b, c).gap);
    if is_rock(g as i8) {
        let up = i32::from(samples.cell(a, b, c + 1).gap);
        if is_rock(up as i8) {
            return None;
        }
        // seat · 8 = 4 + 8·|g| / (|g| + up), rounded half up.
        let den = -g + up;
        let eighths = 4 + (2 * SEAT_STEPS * (-g) + den) / (2 * den);
        if eighths > SEAT_STEPS {
            return None;
        }
        return Some(eighths as u8);
    }
    let down = i32::from(samples.cell(a, b, c - 1).gap);
    if is_rock(down as i8) {
        // seat · 8 = 8·|down| / (|down| + g) − 4, rounded half up, never under the floor.
        let den = -down + g;
        let eighths = (2 * SEAT_STEPS * (-down) + den) / (2 * den) - 4;
        return Some(eighths.max(0) as u8);
    }
    Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::tests::synthetic;

    #[test]
    fn a_seat_is_the_zero_crossing_along_the_radial_in_eighths_and_a_buried_cell_refuses() {
        // The owner's example: gap −17 at c = 5, +111 at c = 6 → 5/8.
        let sb = synthetic(|_, _, c| match c {
            c if c < 5 => -128,
            5 => -17,
            6 => 111,
            _ => 127,
        });
        assert_eq!(seat_eighths(&sb, [3, 3, 5]), Some(5));
        // The cell above the crossing: air over rock, the crossing at t = 17/128 above the centre
        // BELOW → seat = 0.133 − 0.5 < 0 → the floor.
        assert_eq!(seat_eighths(&sb, [3, 3, 6]), Some(0));
        // Air over air: the floor. Rock under rock: refused.
        assert_eq!(seat_eighths(&sb, [3, 3, 9]), Some(0));
        assert_eq!(seat_eighths(&sb, [3, 3, 2]), None);
        // A crossing just above the cell's top: rock −120 at c = 5, air +8 at c = 6 → t = 120/128,
        // seat = 0.5 + 0.9375 = 1.4375 cells > 1 → refused (the ground is in the cell above).
        let sb2 = synthetic(|_, _, c| match c {
            c if c < 5 => -128,
            5 => -120,
            6 => 8,
            _ => 127,
        });
        assert_eq!(seat_eighths(&sb2, [0, 0, 5]), None);
        // And from that cell above: air +8 over rock −120: t = 120/128 → seat = 0.9375 − 0.5 =
        // 0.4375 → 3.5 eighths → rounds half up to 4.
        assert_eq!(seat_eighths(&sb2, [0, 0, 6]), Some(4));
        // A crossing exactly at the cell's top from below: rock −64 at c = 5, air +64 at c = 6 →
        // seat = 1.0 → 8/8, allowed.
        let sb3 = synthetic(|_, _, c| match c {
            c if c < 5 => -128,
            5 => -64,
            6 => 64,
            _ => 127,
        });
        assert_eq!(seat_eighths(&sb3, [0, 0, 5]), Some(8));
        assert_eq!(seat_eighths(&sb3, [0, 0, 6]), Some(0));
        // An address past the chunk's own cells is refused, never read.
        assert_eq!(seat_eighths(&sb3, [0, 0, 62]), None);
        assert_eq!(seat_eighths(&sb3, [62, 0, 5]), None);
        assert_eq!(seat_eighths(&sb3, [0, 200, 5]), None);
    }
}
