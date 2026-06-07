//! The `EffectFree` marker — the compile-time half of HR1's coupling rule
//! (`docs/design/sealed_shards.md` §2).
//!
//! A *coupling* is a continuous, latest-wins, loss-tolerant simulation cross-feed
//! (ship thrust → hull host; atmosphere sample → drag). Its payload must be provably
//! effect-free: a force/sample/input VALUE the sink applies each tick — never a
//! transfer trigger, never authority-gating discrete state (the old dock-clamp-on-a-
//! lossy-datagram bug class is made UNCOMPILABLE: a discrete authority event cannot
//! implement `EffectFree`, so it cannot ride a coupling and must be a `Transfer`).
//!
//! The trait is SEALED: implementations live here, added under the same review
//! discipline as `InterShardFlow` arms. `CouplingPort` itself lands at P8 with its
//! first real consumer (`ShipThrustPort`) — freezing it earlier would freeze the
//! wrong shape.

mod sealed {
    /// Sealing token: only this module can name it, so only this crate can admit
    /// new `EffectFree` payloads.
    pub trait Sealed {}
}

/// Marker: a type carrying NO discrete state that gates a saga/authority change and
/// NO durable side effect. Latest-wins delivery must be semantically lossless.
pub trait EffectFree: sealed::Sealed {}

/// The proof-of-shape payload (P0): continuous force/torque values are the canonical
/// effect-free class. Real ports (ShipOutputs, AtmosphereSample) land at P8 here.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContinuousSample {
    pub values: [f64; 3],
}

impl sealed::Sealed for ContinuousSample {}
impl EffectFree for ContinuousSample {}

/// Compile-time guard usable in generic bounds and tests: only `EffectFree` payloads
/// pass through. (A negative trybuild-style test is unnecessary: not implementing a
/// sealed trait is unrepresentable, not merely unlikely.)
#[must_use]
pub fn assert_effect_free<T: EffectFree>(payload: T) -> T {
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_samples_are_effect_free_and_roundtrip() {
        let sample = assert_effect_free(ContinuousSample {
            values: [1.0, -2.0, 3.5],
        });
        let bytes = postcard::to_allocvec(&sample).expect("encode");
        assert_eq!(
            postcard::from_bytes::<ContinuousSample>(&bytes).expect("decode"),
            sample
        );
        assert_eq!(ContinuousSample::default().values, [0.0; 3]);
    }
}
