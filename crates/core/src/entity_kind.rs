//! The generic-transfer kind registry (HR2): the ONLY per-kind surface in the whole
//! transfer machinery (`docs/design/generic_transfer.md` §A0).
//!
//! A new entity kind that reuses an existing `(DurabilityClass, ContinuityModel,
//! GhostPolicy)` triple is four lines: enum variant, `KindDef` const, one trait impl
//! (the ECS-coupled `TransferableKind` trait lives in `vd-sim` — this crate stays free
//! of bevy_ecs), `register!`. A kind needing a NEW triple is first-class engineering
//! and a crash-matrix multiplication — deliberately scheduled, never casual.
//!
//! ## ⚠️ INTERIM — only the STATIC half (this file) exists; the `TransferableKind` TRAIT is owed (DEFERRED D-31)
//! `EntityKind` + `KindDef` (below) are real and tested. The behavioral half — the per-kind
//! `serialize`/`spawn`/`precondition`/`rebind_refs` trait + the `register!` macro + the
//! `kind_blob_evolution` gate — is NOT built yet: the transfer envelope carries an opaque
//! `state: Vec<u8>` (a Player's pose crosses as a typed field, not through the trait). It is
//! deliberately deferred to its first real consumer (serialize/spawn with the TLV blob at 1d.6;
//! `rebind_refs` with the first compound kind at P6/P8) because the methods have nothing to do
//! until per-kind state + child/frame refs exist — the retrofit is additive at the `state` field.
//! See **DEFERRED.md D-31**. Flips green when a NON-Player kind crosses end-to-end via the trait.
//!
//! Day-one behavioral surface is capped to TWO triples (PLAN.md P0 delta):
//! `{Durable, Frozen, Always}` (Player) and
//! `{Transient, BallisticReadvance, NeverTransferOnly}` (Debris).
//! Guided/RealmAnchored/InBand land at P6/P10 where their crash-matrix cost is budgeted.

use serde::{Deserialize, Serialize};

/// Entity kind tag, embedded in `EntityId` bits 120..128.
///
/// Reserved bands: 0..10 Durable kinds, 10..20 Transient kinds (matches the design's
/// numbering; gaps are deliberate growth room).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    Player = 0,
    Ship = 1,
    NamedConstruction = 2,
    Debris = 10,
    DroppedBlock = 11,
    Projectile = 12,
    Rocket = 13,
}

impl EntityKind {
    /// All registered kinds, in tag order. The registry tables below are total over
    /// this set (compiler-enforced by exhaustive matches).
    pub const ALL: [EntityKind; 7] = [
        EntityKind::Player,
        EntityKind::Ship,
        EntityKind::NamedConstruction,
        EntityKind::Debris,
        EntityKind::DroppedBlock,
        EntityKind::Projectile,
        EntityKind::Rocket,
    ];

    /// Recover a kind from an `EntityId`'s tag byte. Unknown tags are an error —
    /// never a default (the decode-to-Default ban, HR2).
    pub fn from_tag(tag: u8) -> Result<EntityKind, UnknownKindTag> {
        match tag {
            0 => Ok(EntityKind::Player),
            1 => Ok(EntityKind::Ship),
            2 => Ok(EntityKind::NamedConstruction),
            10 => Ok(EntityKind::Debris),
            11 => Ok(EntityKind::DroppedBlock),
            12 => Ok(EntityKind::Projectile),
            13 => Ok(EntityKind::Rocket),
            other => Err(UnknownKindTag(other)),
        }
    }

    /// This kind's transfer policy. Static data, total over all kinds.
    #[must_use]
    pub fn def(self) -> &'static KindDef {
        match self {
            EntityKind::Player => &PLAYER_DEF,
            EntityKind::Ship => &SHIP_DEF,
            EntityKind::NamedConstruction => &NAMED_CONSTRUCTION_DEF,
            EntityKind::Debris => &DEBRIS_DEF,
            EntityKind::DroppedBlock => &DROPPED_BLOCK_DEF,
            EntityKind::Projectile => &PROJECTILE_DEF,
            EntityKind::Rocket => &ROCKET_DEF,
        }
    }
}

/// The continuity model of the kind an `EntityId` encodes (D-7b: the per-continuity advance seam —
/// dispatch on the KIND, never a shard kind, HR3). An `EntityId` whose tag this binary does not know
/// re-advances as `Frozen` (stamp-only, no invented motion) — the conservative default for the
/// decode-to-Default-banned (HR2) unknown-kind case, never a panic on a corrupt id.
#[must_use]
pub fn continuity_of(entity: crate::EntityId) -> ContinuityModel {
    EntityKind::from_tag(entity.kind_tag()).map_or(ContinuityModel::Frozen, |k| k.def().continuity)
}

/// An `EntityId` carried a kind tag this binary does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown entity-kind tag {0}")]
pub struct UnknownKindTag(pub u8);

/// Durable kinds get the full machinery (per-entity OwnerRecord, saga WAL, full crash
/// matrix, zero loss). Transient kinds share the SAME machinery with policy-driven
/// persistence (held-set authority + batched TransientGo go-token, declared loss
/// budgets). One FSM, one commit point — the tier is a parameter, not a fork (HR3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurabilityClass {
    Durable,
    Transient,
}

/// Per-kind ghost policy and what NO-VANISH means for it
/// (`docs/design/generic_transfer.md` §A4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GhostPolicy {
    /// Continuous kinematic-collider ghost across the whole overlap band.
    Always,
    /// Ghosted only inside the overlap band.
    InBand,
    /// Never ghosted; crosses via the transient batch, appears next tick.
    /// NO-VANISH relaxes to TRANSIENT-CONSERVATION.
    NeverTransferOnly,
}

/// How an entity's motion survives a transfer (`generic_transfer.md` §A3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuityModel {
    /// Source freezes, computes the dest pose, dest resumes (players, ships).
    Frozen,
    /// Closed-form Category-A ballistic re-advance from `source_tick` to dest's tick.
    BallisticReadvance,
    /// BallisticReadvance + guidance target resolved via the four-way branch
    /// (live ghost / fence-stale -> honest ballistic / extrapolated / unresolvable).
    Guided,
    /// In flight = transient ballistic; at rest = durable realm-block conversion via
    /// the two-step `LandBlock` ack keyed by `entity_id`.
    RealmAnchored,
}

/// Declared, asserted loss tolerance. Durable kinds are exactly `ZERO`; transient
/// kinds declare how many in-flight losses per scenario the TRANSIENT-CONSERVATION
/// invariant tolerates (loss beyond budget is a test failure; duplication always is).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LossBudget(pub u16);

impl LossBudget {
    pub const ZERO: LossBudget = LossBudget(0);
}

/// Schema id of a kind's TLV state blob (`crate::tlv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaId(pub u16);

/// The per-kind transfer policy record: the registry's single point of variation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindDef {
    pub class: DurabilityClass,
    pub ghost_policy: GhostPolicy,
    pub continuity: ContinuityModel,
    pub loss_budget: LossBudget,
    pub blob_schema: SchemaId,
    /// Hard cap on the kind's serialized TLV blob (sizing the transfer envelopes).
    pub max_state_bytes: u16,
}

impl KindDef {
    /// Registry coherence rules, checked by tests over `EntityKind::ALL`:
    /// Durable implies zero loss budget; a Frozen continuity implies Durable
    /// (only authoritative-state kinds freeze); Transient kinds must declare
    /// a ghost policy that never requires AUTHORITY-UNIQUE ghosting (`Always`
    /// is reserved for Durable kinds whose ghosts the oracle tracks).
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        let durable_lossless =
            self.class != DurabilityClass::Durable || self.loss_budget == LossBudget::ZERO;
        let frozen_is_durable =
            self.continuity != ContinuityModel::Frozen || self.class == DurabilityClass::Durable;
        let transient_ghosting =
            self.class != DurabilityClass::Transient || self.ghost_policy != GhostPolicy::Always;
        durable_lossless && frozen_is_durable && transient_ghosting
    }
}

// --- The day-one registry -------------------------------------------------------
// Two active triples (P0 cap); Ship/NamedConstruction reuse the Player triple,
// DroppedBlock/Projectile/Rocket are REGISTERED (ids mintable) but their
// RealmAnchored/Guided continuity activates at P6/P10 per the roadmap.

pub static PLAYER_DEF: KindDef = KindDef {
    class: DurabilityClass::Durable,
    ghost_policy: GhostPolicy::Always,
    continuity: ContinuityModel::Frozen,
    loss_budget: LossBudget::ZERO,
    blob_schema: SchemaId(1),
    max_state_bytes: 4096,
};

pub static SHIP_DEF: KindDef = KindDef {
    class: DurabilityClass::Durable,
    ghost_policy: GhostPolicy::Always,
    continuity: ContinuityModel::Frozen,
    loss_budget: LossBudget::ZERO,
    blob_schema: SchemaId(2),
    max_state_bytes: 8192,
};

pub static NAMED_CONSTRUCTION_DEF: KindDef = KindDef {
    class: DurabilityClass::Durable,
    ghost_policy: GhostPolicy::Always,
    continuity: ContinuityModel::Frozen,
    loss_budget: LossBudget::ZERO,
    blob_schema: SchemaId(3),
    max_state_bytes: 8192,
};

pub static DEBRIS_DEF: KindDef = KindDef {
    class: DurabilityClass::Transient,
    ghost_policy: GhostPolicy::NeverTransferOnly,
    continuity: ContinuityModel::BallisticReadvance,
    loss_budget: LossBudget(4),
    blob_schema: SchemaId(10),
    max_state_bytes: 128,
};

pub static DROPPED_BLOCK_DEF: KindDef = KindDef {
    class: DurabilityClass::Transient,
    ghost_policy: GhostPolicy::NeverTransferOnly,
    continuity: ContinuityModel::RealmAnchored,
    loss_budget: LossBudget(2),
    blob_schema: SchemaId(11),
    max_state_bytes: 64,
};

pub static PROJECTILE_DEF: KindDef = KindDef {
    class: DurabilityClass::Transient,
    ghost_policy: GhostPolicy::NeverTransferOnly,
    continuity: ContinuityModel::BallisticReadvance,
    loss_budget: LossBudget(8),
    blob_schema: SchemaId(12),
    max_state_bytes: 64,
};

pub static ROCKET_DEF: KindDef = KindDef {
    class: DurabilityClass::Transient,
    ghost_policy: GhostPolicy::InBand,
    continuity: ContinuityModel::Guided,
    loss_budget: LossBudget(2),
    blob_schema: SchemaId(13),
    max_state_bytes: 256,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_kind_is_coherent() {
        let incoherent: Vec<EntityKind> = EntityKind::ALL
            .into_iter()
            .filter(|kind| !kind.def().is_coherent())
            .collect();
        assert_eq!(incoherent, Vec::<EntityKind>::new());
    }

    #[test]
    fn from_tag_roundtrips_every_kind_and_rejects_unknown() {
        for kind in EntityKind::ALL {
            assert_eq!(EntityKind::from_tag(kind as u8), Ok(kind));
        }
        assert_eq!(EntityKind::from_tag(200), Err(UnknownKindTag(200)));
        assert_eq!(
            EntityKind::from_tag(200).expect_err("unknown").to_string(),
            "unknown entity-kind tag 200"
        );
    }

    #[test]
    fn continuity_of_reads_the_kind_and_defaults_unknown_to_frozen() {
        // D-7b: a Debris id re-advances ballistically (its kind's continuity); a corrupt id whose tag
        // this binary does not know re-advances as Frozen (stamp-only) — never a panic.
        let debris = crate::EntityId::pack(EntityKind::Debris, 1, 7, 0);
        assert_eq!(continuity_of(debris), ContinuityModel::BallisticReadvance);
        let player = crate::EntityId::pack(EntityKind::Player, 1, 7, 0);
        assert_eq!(continuity_of(player), ContinuityModel::Frozen);
        // An EntityId with an unknown tag byte (99) → the safe Frozen default.
        let unknown_tag = crate::EntityId(99u128 << 120);
        assert_eq!(unknown_tag.kind_tag(), 99);
        assert_eq!(continuity_of(unknown_tag), ContinuityModel::Frozen);
    }

    #[test]
    fn the_registry_cannot_drift_from_the_enum() {
        // ALL and from_tag are hand-maintained; HR2 hinges on tag decode, so a variant
        // missing from either is a silently-unspawnable entity CLASS (audit FG-3).
        //
        // Tripwire 1 — adding an enum variant breaks THIS exhaustive match at compile
        // time, directing the author here: register it in ALL + from_tag + this list.
        for kind in EntityKind::ALL {
            match kind {
                EntityKind::Player
                | EntityKind::Ship
                | EntityKind::NamedConstruction
                | EntityKind::Debris
                | EntityKind::DroppedBlock
                | EntityKind::Projectile
                | EntityKind::Rocket => {}
            }
        }
        // Tripwire 2 — sweep the FULL tag space: every accepted tag must round-trip to
        // itself AND appear in ALL, and the accepted count must equal ALL's length —
        // so from_tag and ALL can never disagree in either direction.
        let mut accepted = 0usize;
        for tag in 0..=u8::MAX {
            if let Ok(kind) = EntityKind::from_tag(tag) {
                accepted += 1;
                assert_eq!(kind as u8, tag, "tag {tag} decodes to a kind with that tag");
                assert!(
                    EntityKind::ALL.contains(&kind),
                    "{kind:?} decodable but missing from ALL"
                );
            }
        }
        assert_eq!(
            accepted,
            EntityKind::ALL.len(),
            "from_tag accepts exactly the registered set"
        );
    }

    #[test]
    fn durable_kinds_have_zero_loss_budget() {
        for kind in EntityKind::ALL {
            let def = kind.def();
            if def.class == DurabilityClass::Durable {
                assert_eq!(def.loss_budget, LossBudget::ZERO, "{kind:?}");
            }
        }
    }

    #[test]
    fn blob_schemas_are_unique_across_kinds() {
        let mut schemas: Vec<u16> = EntityKind::ALL
            .iter()
            .map(|k| k.def().blob_schema.0)
            .collect();
        schemas.sort_unstable();
        let before = schemas.len();
        schemas.dedup();
        assert_eq!(before, schemas.len(), "duplicate SchemaId in the registry");
    }

    #[test]
    fn incoherent_defs_are_detected() {
        // Durable with a loss budget: forbidden.
        let lossy_durable = KindDef {
            loss_budget: LossBudget(1),
            ..PLAYER_DEF
        };
        assert!(!lossy_durable.is_coherent());
        // Frozen continuity on a Transient: forbidden.
        let frozen_transient = KindDef {
            continuity: ContinuityModel::Frozen,
            ..DEBRIS_DEF
        };
        assert!(!frozen_transient.is_coherent());
        // Always-ghosted Transient: forbidden (oracle can't track it).
        let ghosted_transient = KindDef {
            ghost_policy: GhostPolicy::Always,
            ..DEBRIS_DEF
        };
        assert!(!ghosted_transient.is_coherent());
    }

    #[test]
    fn kind_defs_serde_roundtrip() {
        let bytes = postcard::to_allocvec(&PLAYER_DEF).expect("encode");
        let back: KindDef = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, PLAYER_DEF);
    }
}
