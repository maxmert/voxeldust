//! Client operational parameters — ONE reviewed struct, no inline literals
//! (CLAUDE.md no-magic-numbers). The interpolation buffer lives here, on the
//! client, because only the client interpolates; it mirrors the spec's
//! `interp_buffer_ms` (connection_plane.md §latency budget).

/// Interpolation + render-cursor tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClientInterpTuning {
    /// How far behind the freshest delivered snapshot the render cursor sits, in
    /// milliseconds. 100-150 ms absorbs 20 Hz snapshot jitter and datagram reorder
    /// so there is ALWAYS a newer snapshot to interpolate toward — never
    /// extrapolation (the no-prediction mandate).
    pub interp_buffer_ms: f64,
    /// The universe-tick rate (ticks per second). Snapshots stamp `universe_tick`;
    /// between them the render cursor advances at this rate in tick units.
    pub tick_hz: f64,
}

impl ClientInterpTuning {
    /// The P1.5 default: a 120 ms buffer at a 20 Hz universe tick.
    pub const DEFAULT: ClientInterpTuning = ClientInterpTuning {
        // The contract constant: the shard's interest lead reads the same number.
        interp_buffer_ms: vd_wire::channels::INTERP_BUFFER_MS,
        tick_hz: 20.0,
    };

    /// The buffer expressed in universe ticks (what the render cursor sits behind
    /// the freshest delivered tick).
    #[must_use]
    pub fn buffer_ticks(self) -> f64 {
        self.interp_buffer_ms / 1000.0 * self.tick_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_buffer_is_a_few_ticks() {
        let t = ClientInterpTuning::DEFAULT;
        assert_eq!(t.interp_buffer_ms, 120.0);
        assert_eq!(t.tick_hz, 20.0);
        // 120 ms * 20 Hz / 1000 = 2.4 ticks behind the freshest snapshot.
        assert_eq!(t.buffer_ticks(), 2.4);
    }
}
