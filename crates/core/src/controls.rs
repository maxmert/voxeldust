//! THROWAWAY — THE TEMPORARY CONTROL SEAM'S KEY BINDINGS (owner, 2026-09-02).
//!
//! Until a hull is built from blocks and a SEAT names the pilot, the keyboard flies the realm the
//! player stands in. The wire carries movement axes and an action MASK whose bits mean nothing on the
//! client; the binding of a bit to a meaning is the server's (the future Functional Panel). These are
//! the two bits the interim pilot uses, named once so the window that sets them and the shard that
//! reads them cannot drift. Both go when a seat and a functional block replace them.
//!
//! Example: the pilot holds `E`. The window sets action bit [`PILOT_PITCH_UP_INDEX`] on the wire. The
//! ship realm's shard reads that bit, raises the nose of the hull at its rated turn, and states the
//! turn to the star system, which is the only realm that writes where the hull points.

/// The action INDEX (see `action_bit`) that lowers the nose: the `Q` key in the dev window.
pub const PILOT_PITCH_DOWN_INDEX: u32 = 0;
/// The action INDEX that raises the nose: the `E` key in the dev window.
pub const PILOT_PITCH_UP_INDEX: u32 = 1;

/// The mask bit for an index — the one place the transform lives on the server side (the client's
/// twin is `vd_devproto::action_bit`, which also refuses an index past the `u32` channel).
#[must_use]
pub const fn mask_of(index: u32) -> u32 {
    1u32 << index
}

#[cfg(test)]
mod tests {
    use super::{PILOT_PITCH_DOWN_INDEX, PILOT_PITCH_UP_INDEX, mask_of};

    #[test]
    fn the_two_pilot_bits_are_distinct_and_are_the_indices_shifted() {
        assert_eq!(mask_of(PILOT_PITCH_DOWN_INDEX), 0b01);
        assert_eq!(mask_of(PILOT_PITCH_UP_INDEX), 0b10);
        assert_ne!(PILOT_PITCH_DOWN_INDEX, PILOT_PITCH_UP_INDEX);
    }
}
