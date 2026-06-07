//! Identity and correlation primitives.
//!
//! The identity triple (integration resolution): `AccountId` is the durable principal,
//! `SessionId` the connection-session principal, `EntityId` the simulated thing — a
//! session may own several entities, and the directory (not the credential) maps
//! between them. `TransferId` is THE one correlation id (== the old designs'
//! `TxnId`/`correlation_id`/`TransferCorrelationId`), carried on every message, span,
//! and idempotency key touching a transfer.
//!
//! Nothing here is ever derived from wall-clock time (R7: the old session token was a
//! SipHash of `SystemTime::now()` — forgeable, and the enabler of three bug classes).

use serde::{Deserialize, Serialize};

use crate::entity_kind::EntityKind;

/// Identifies a process-level node (shard, gateway, orchestrator, scripted client).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// A node-local simulation tick counter (monotonic; NOT globally synchronized —
/// every cross-shard message carries the sender's `source_tick`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct TickId(pub u64);

impl TickId {
    /// The next tick. Plain saturating increment: a tick counter never wraps in practice
    /// (u64 at 20 Hz outlives the universe), but saturation keeps the type total.
    #[must_use]
    pub fn next(self) -> TickId {
        TickId(self.0.saturating_add(1))
    }
}

impl core::fmt::Display for TickId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tick-{}", self.0)
    }
}

/// Transport-assigned, per-sender monotonic message sequence number.
///
/// Used for at-least-once delivery bookkeeping and to correlate
/// `Inbound::NodeUnreachable { undelivered }` back to a send: sends on one
/// transport are FIFO, so the k-th accepted `send` carries `MsgId(k)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MsgId(pub u64);

impl core::fmt::Display for MsgId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "msg-{}", self.0)
    }
}

/// The durable account principal (random u128, non-PII; minted at account creation).
/// The old system evicted players by NAME — names here are display-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AccountId(pub u128);

/// The connection-session principal: the cross-shard identity a gateway attests for a
/// connected client. 128-bit random — never time-derived (R7), never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionId(pub u128);

/// THE transfer correlation id: minted by the orchestrator at saga creation, carried
/// on every message/span/frame touching the transfer; `(TransferId, step_id)` is the
/// universal side-effect idempotency key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransferId(pub u128);

impl core::fmt::Display for TransferId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "xfer-{:032x}", self.0)
    }
}

/// Identifies one universe epoch (genesis). Every persisted record carries it;
/// a mismatch on recovery means the record belongs to a wiped/rolled universe and is
/// discarded fail-safe rather than resumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EpochId(pub u64);

/// The analytic universe clock value (glossary: `universe_tick`). Owned by the
/// orchestrator's DurableUniverseClock (write-ahead ceiling, monotonic clamp); all
/// Category-A celestial math is a closed form `f(seed, universe_tick)`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct UniverseTick(pub u64);

/// The globally unique, stable identity of a simulated thing (player avatar, ship,
/// debris chunk, rocket). Packed `{kind: u8, mint_shard: u32, seq: u64, rand: u24}` —
/// wait-free to mint shard-locally, never reused, never time-derived.
///
/// Layout (most-significant first): `kind:8 | mint_shard:32 | seq:64 | rand:24` = 128.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u128);

impl EntityId {
    /// Pack the components. `seq` is the minting shard's monotonic entity counter;
    /// `rand` is 24 bits of seed-derived (NOT wall-clock) entropy guarding against a
    /// recovered shard re-minting after losing its counter tail.
    #[must_use]
    pub fn pack(kind: EntityKind, mint_shard: u32, seq: u64, rand24: u32) -> EntityId {
        let kind_bits = u128::from(kind as u8) << 120;
        let shard_bits = u128::from(mint_shard) << 88;
        let seq_bits = u128::from(seq) << 24;
        let rand_bits = u128::from(rand24 & 0x00FF_FFFF);
        EntityId(kind_bits | shard_bits | seq_bits | rand_bits)
    }

    /// The entity-kind tag (drives the transfer registry, HR2).
    #[must_use]
    pub fn kind_tag(self) -> u8 {
        (self.0 >> 120) as u8
    }

    /// The shard that minted this entity.
    #[must_use]
    pub fn mint_shard(self) -> u32 {
        (self.0 >> 88) as u32
    }

    /// The minting shard's sequence number.
    #[must_use]
    pub fn seq(self) -> u64 {
        (self.0 >> 24) as u64
    }

    /// The 24-bit entropy tail.
    #[must_use]
    pub fn rand24(self) -> u32 {
        (self.0 as u32) & 0x00FF_FFFF
    }
}

impl core::fmt::Display for EntityId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ent-{:02x}.{:08x}.{:x}.{:06x}",
            self.kind_tag(),
            self.mint_shard(),
            self.seq(),
            self.rand24()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_next_increments() {
        assert_eq!(TickId(0).next(), TickId(1));
        assert_eq!(TickId(41).next(), TickId(42));
    }

    #[test]
    fn tick_next_saturates_at_max() {
        assert_eq!(TickId(u64::MAX).next(), TickId(u64::MAX));
    }

    #[test]
    fn display_formats() {
        assert_eq!(NodeId(7).to_string(), "node-7");
        assert_eq!(TickId(3).to_string(), "tick-3");
        assert_eq!(MsgId(9).to_string(), "msg-9");
    }

    #[test]
    fn ids_roundtrip_postcard() {
        let n = NodeId(123);
        let bytes = postcard::to_allocvec(&n).expect("encode");
        let back: NodeId = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(n, back);
    }

    #[test]
    fn ids_are_ordered() {
        assert!(NodeId(1) < NodeId(2));
        assert!(MsgId(1) < MsgId(2));
        assert!(TickId::default() < TickId(1));
        assert!(AccountId(1) < AccountId(2));
        assert!(SessionId(1) < SessionId(2));
        assert!(TransferId(1) < TransferId(2));
        assert!(EpochId(1) < EpochId(2));
        assert!(UniverseTick::default() < UniverseTick(1));
    }

    #[test]
    fn entity_id_packs_and_unpacks_every_component() {
        let id = EntityId::pack(
            EntityKind::Debris,
            0xDEAD_BEEF,
            0x0123_4567_89AB_CDEF,
            0xFAB123,
        );
        assert_eq!(id.kind_tag(), EntityKind::Debris as u8);
        assert_eq!(id.mint_shard(), 0xDEAD_BEEF);
        assert_eq!(id.seq(), 0x0123_4567_89AB_CDEF);
        assert_eq!(id.rand24(), 0xFAB123);
        assert_eq!(
            crate::entity_kind::EntityKind::from_tag(id.kind_tag()),
            Ok(EntityKind::Debris)
        );
    }

    #[test]
    fn entity_id_rand_is_masked_to_24_bits() {
        let id = EntityId::pack(EntityKind::Player, 1, 2, 0xFFFF_FFFF);
        assert_eq!(id.rand24(), 0x00FF_FFFF, "upper rand bits must not leak");
        assert_eq!(id.seq(), 2, "overflow must not corrupt seq");
    }

    #[test]
    fn entity_and_transfer_display_formats() {
        let id = EntityId::pack(EntityKind::Ship, 0xAB, 0x10, 0x00CAFE);
        assert_eq!(id.to_string(), "ent-01.000000ab.10.00cafe");
        assert_eq!(
            TransferId(0xFF).to_string(),
            "xfer-000000000000000000000000000000ff"
        );
    }

    #[test]
    fn new_ids_roundtrip_postcard() {
        let ids = (
            AccountId(7),
            SessionId(8),
            TransferId(9),
            EpochId(10),
            UniverseTick(11),
            EntityId::pack(EntityKind::Rocket, 1, 2, 3),
        );
        let bytes = postcard::to_allocvec(&ids).expect("encode");
        let back: (
            AccountId,
            SessionId,
            TransferId,
            EpochId,
            UniverseTick,
            EntityId,
        ) = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, ids);
    }

    proptest::proptest! {
        /// Pack/unpack is a bijection over the full component domains.
        #[test]
        fn entity_id_pack_roundtrips(
            shard in proptest::prelude::any::<u32>(),
            seq in proptest::prelude::any::<u64>(),
            rand in 0u32..0x0100_0000,
        ) {
            let id = EntityId::pack(EntityKind::Projectile, shard, seq, rand);
            proptest::prop_assert_eq!(id.mint_shard(), shard);
            proptest::prop_assert_eq!(id.seq(), seq);
            proptest::prop_assert_eq!(id.rand24(), rand);
            proptest::prop_assert_eq!(id.kind_tag(), EntityKind::Projectile as u8);
        }
    }
}
