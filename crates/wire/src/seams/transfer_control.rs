//! The Transfer Saga ⇄ gateway seam (`docs/design/connection_plane.md` §2.2): the
//! phased, separately-acked, individually-compensatable command vocabulary.
//!
//! The saga (orchestrator-side, internal FSM `PREPARE → FLUSH → FENCE_DEMOTE →
//! PROMOTE → CLEANUP`) drives the gateway exclusively through these commands; the
//! gateway never decides a transfer. Every phase has a defined compensator —
//! `FreezeSource`'s is `ThawSource` — so no partial state ever strands a player.
//! There is deliberately NO single command fanning freeze+resume to two shards
//! "atomically" (that would be an unfenced 2-of-2 distributed write).

use serde::{Deserialize, Serialize};
use vd_core::{Fence, NodeId, SessionId, TransferId};

use crate::channels::TransferRejectReason;
use crate::seams::directory::DirectoryKey;

/// Saga → gateway commands. Idempotent: re-delivery of any command is a no-op
/// acknowledged with the same ack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferControl {
    /// Ensure the destination subscription + frozen ghost mirror exist.
    PrepareSubscribe {
        transfer: TransferId,
        session: SessionId,
        dest: NodeId,
    },
    /// Ask the client (via CONTROL) to emit the in-band CUT_MARKER.
    RequestCut {
        transfer: TransferId,
        session: SessionId,
    },
    /// Route `seq <= marker` to source, buffer `seq > marker` for dest.
    FreezeSource {
        transfer: TransferId,
        session: SessionId,
        marker_seq: u64,
        /// The dest authority the cut buffers `seq > marker_seq` toward. Carried from
        /// the saga's durable `SagaCtx.dest` (the SAME source as `PrepareSubscribe.dest`),
        /// NOT re-derived in the gateway: the gateway is soft-state and re-registers from
        /// the saga on resume/adoption, so the durable carrier is the saga, not gateway RAM.
        dest: NodeId,
    },
    /// THE route swap: a single atomic store after the directory CAS won.
    CommitAuthority {
        transfer: TransferId,
        session: SessionId,
        new_fence: Fence,
        /// The transfer SUBJECT (the directory key the CAS moved) — carried VERBATIM from the
        /// saga's `SagaCtx.subject` so the gateway can hand it to the dest's `OpenInputSlot`
        /// (the dest extracts the `Entity` to ADOPT the transferred avatar; a non-`Entity`
        /// subject, e.g. a Realm saga, is a counted no-op at the dest — never an extraction
        /// panic). Appended (postcard field order is positional; additive-only).
        subject: DirectoryKey,
    },
    /// COMPENSATOR for FreezeSource: input resumes flowing to the source.
    ThawSource {
        transfer: TransferId,
        session: SessionId,
    },
    /// Tear down the dest ghost/buffer; the source remains authoritative.
    AbortTransfer {
        transfer: TransferId,
        session: SessionId,
    },
    /// Close the source subscription after demote grace (event-driven, no magic ticks).
    ReleaseSubscribe {
        transfer: TransferId,
        session: SessionId,
        src: NodeId,
    },
}

/// Gateway → saga acks: one per command, each carrying what the next phase needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferControlAck {
    Prepared {
        transfer: TransferId,
        result: PrepareResult,
    },
    CutConfirmed {
        transfer: TransferId,
        marker_seq: u64,
    },
    SourceFrozen {
        transfer: TransferId,
        /// The source applied input through exactly this seq.
        drained_seq: u64,
    },
    Committed {
        transfer: TransferId,
    },
    SourceThawed {
        transfer: TransferId,
    },
    Aborted {
        transfer: TransferId,
    },
    Released {
        transfer: TransferId,
    },
}

/// Preparation outcome: spatial-precondition failure is TYPED (the hull-trap class
/// aborts with feedback), never a generic readiness bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrepareResult {
    Ready,
    Rejected(PrepareReject),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrepareReject {
    /// The dest cannot place the entity (e.g. disembark point obstructed).
    Spatial(SpatialReject),
    /// The dest's known-tag floor cannot represent the entity's required state.
    VersionFloor,
    /// The dest subscription could not be established.
    SubscriptionFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialReject {
    Obstructed,
    OutOfBounds,
}

impl PrepareReject {
    /// The client-facing rejection reason for cosmetic feedback.
    #[must_use]
    pub fn client_reason(self) -> TransferRejectReason {
        match self {
            PrepareReject::Spatial(_) => TransferRejectReason::SpatiallyObstructed,
            PrepareReject::VersionFloor => TransferRejectReason::VersionFloor,
            PrepareReject::SubscriptionFailed => TransferRejectReason::DestinationUnavailable,
        }
    }
}

impl TransferControl {
    /// The transfer this command belongs to (every command carries it). The directory
    /// key the saga is serialized on derives from this.
    #[must_use]
    pub fn transfer(self) -> TransferId {
        match self {
            TransferControl::PrepareSubscribe { transfer, .. }
            | TransferControl::RequestCut { transfer, .. }
            | TransferControl::FreezeSource { transfer, .. }
            | TransferControl::CommitAuthority { transfer, .. }
            | TransferControl::ThawSource { transfer, .. }
            | TransferControl::AbortTransfer { transfer, .. }
            | TransferControl::ReleaseSubscribe { transfer, .. } => transfer,
        }
    }

    /// The session this command targets (every command carries it). The gateway
    /// correlates a `TransferControl` to its `Session` by this — the same identity the
    /// saga serializes on — so the consumer never destructures all seven arms by hand.
    #[must_use]
    pub fn session(self) -> SessionId {
        match self {
            TransferControl::PrepareSubscribe { session, .. }
            | TransferControl::RequestCut { session, .. }
            | TransferControl::FreezeSource { session, .. }
            | TransferControl::CommitAuthority { session, .. }
            | TransferControl::ThawSource { session, .. }
            | TransferControl::AbortTransfer { session, .. }
            | TransferControl::ReleaseSubscribe { session, .. } => session,
        }
    }

    /// The saga PHASE this command is, as a stable idempotency step id: `(transfer,
    /// step_id)` is journaled in `applied_steps` before the gateway applies the command,
    /// so a re-delivery at the same step is a no-op (the at-least-once half of HR1's
    /// side-effecting contract). The value is the phase ORDER, frozen with the vocabulary;
    /// the matching ack shares it (`FreezeSource`/`SourceFrozen` are both phase 2).
    #[must_use]
    pub fn step_id(self) -> u32 {
        match self {
            TransferControl::PrepareSubscribe { .. } => 0,
            TransferControl::RequestCut { .. } => 1,
            TransferControl::FreezeSource { .. } => 2,
            TransferControl::CommitAuthority { .. } => 3,
            TransferControl::ThawSource { .. } => 4,
            TransferControl::AbortTransfer { .. } => 5,
            TransferControl::ReleaseSubscribe { .. } => 6,
        }
    }
}

impl TransferControlAck {
    /// The transfer this ack belongs to (every ack carries it).
    #[must_use]
    pub fn transfer(self) -> TransferId {
        match self {
            TransferControlAck::Prepared { transfer, .. }
            | TransferControlAck::CutConfirmed { transfer, .. }
            | TransferControlAck::SourceFrozen { transfer, .. }
            | TransferControlAck::Committed { transfer }
            | TransferControlAck::SourceThawed { transfer }
            | TransferControlAck::Aborted { transfer }
            | TransferControlAck::Released { transfer } => transfer,
        }
    }

    /// The saga phase this ack reports, parallel to the command's [`TransferControl::step_id`]
    /// (the ack of `FreezeSource` is `SourceFrozen`, both phase 2) — the saga's idempotency
    /// key for "phase N acknowledged".
    #[must_use]
    pub fn step_id(self) -> u32 {
        match self {
            TransferControlAck::Prepared { .. } => 0,
            TransferControlAck::CutConfirmed { .. } => 1,
            TransferControlAck::SourceFrozen { .. } => 2,
            TransferControlAck::Committed { .. } => 3,
            TransferControlAck::SourceThawed { .. } => 4,
            TransferControlAck::Aborted { .. } => 5,
            TransferControlAck::Released { .. } => 6,
        }
    }
}

/// The compensator pairing, encoded as data so the saga's compensation chain is
/// testable without the gateway: which command undoes which.
#[must_use]
pub fn compensator_of(cmd: TransferControl) -> Option<TransferControl> {
    match cmd {
        TransferControl::FreezeSource {
            transfer, session, ..
        } => Some(TransferControl::ThawSource { transfer, session }),
        TransferControl::PrepareSubscribe {
            transfer, session, ..
        } => Some(TransferControl::AbortTransfer { transfer, session }),
        // Cut requests need no compensation (an unused marker is inert), and the
        // post-commit phases are forward-only (commit is the point of no return).
        TransferControl::RequestCut { .. }
        | TransferControl::CommitAuthority { .. }
        | TransferControl::ThawSource { .. }
        | TransferControl::AbortTransfer { .. }
        | TransferControl::ReleaseSubscribe { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: TransferId = TransferId(9);
    const S: SessionId = SessionId(3);

    fn all_commands() -> Vec<TransferControl> {
        vec![
            TransferControl::PrepareSubscribe {
                transfer: T,
                session: S,
                dest: NodeId(2),
            },
            TransferControl::RequestCut {
                transfer: T,
                session: S,
            },
            TransferControl::FreezeSource {
                transfer: T,
                session: S,
                marker_seq: 17,
                dest: NodeId(2),
            },
            TransferControl::CommitAuthority {
                transfer: T,
                session: S,
                new_fence: Fence(4),
                subject: DirectoryKey::Entity(vd_core::EntityId(7)),
            },
            TransferControl::ThawSource {
                transfer: T,
                session: S,
            },
            TransferControl::AbortTransfer {
                transfer: T,
                session: S,
            },
            TransferControl::ReleaseSubscribe {
                transfer: T,
                session: S,
                src: NodeId(1),
            },
        ]
    }

    /// One ack per VARIANT (the phase order), shared by the roundtrip + step-id tests.
    fn all_acks() -> Vec<TransferControlAck> {
        vec![
            TransferControlAck::Prepared {
                transfer: T,
                result: PrepareResult::Ready,
            },
            TransferControlAck::CutConfirmed {
                transfer: T,
                marker_seq: 17,
            },
            TransferControlAck::SourceFrozen {
                transfer: T,
                drained_seq: 17,
            },
            TransferControlAck::Committed { transfer: T },
            TransferControlAck::SourceThawed { transfer: T },
            TransferControlAck::Aborted { transfer: T },
            TransferControlAck::Released { transfer: T },
        ]
    }

    #[test]
    fn commands_and_acks_roundtrip() {
        for cmd in all_commands() {
            let bytes = postcard::to_allocvec(&cmd).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferControl>(&bytes).expect("decode"),
                cmd
            );
        }
        // Every ack variant + the Rejected `PrepareResult` arm (the Ready arm is in all_acks).
        let rejected = TransferControlAck::Prepared {
            transfer: T,
            result: PrepareResult::Rejected(PrepareReject::Spatial(SpatialReject::Obstructed)),
        };
        for ack in all_acks().into_iter().chain(std::iter::once(rejected)) {
            let bytes = postcard::to_allocvec(&ack).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferControlAck>(&bytes).expect("decode"),
                ack
            );
        }
    }

    #[test]
    fn transfer_and_step_id_cover_every_command_and_ack() {
        use std::collections::BTreeSet;
        // Every command exposes its transfer + a DISTINCT stable phase step_id.
        let cmds = all_commands();
        let mut cmd_steps = BTreeSet::new();
        for cmd in &cmds {
            assert_eq!(cmd.transfer(), T);
            assert_eq!(cmd.session(), S, "every command carries its session");
            cmd_steps.insert(cmd.step_id());
        }
        assert_eq!(
            cmd_steps.len(),
            cmds.len(),
            "each command phase has a distinct step_id"
        );

        let acks = all_acks();
        let mut ack_steps = BTreeSet::new();
        for ack in &acks {
            assert_eq!(ack.transfer(), T);
            ack_steps.insert(ack.step_id());
        }
        assert_eq!(
            ack_steps.len(),
            acks.len(),
            "each ack phase has a distinct step_id"
        );

        // The ack of a command shares its phase (parallel numbering): FreezeSource (a
        // command) and SourceFrozen (its ack) are both phase 2.
        let freeze = TransferControl::FreezeSource {
            transfer: T,
            session: S,
            marker_seq: 5,
            dest: NodeId(2),
        };
        let frozen = TransferControlAck::SourceFrozen {
            transfer: T,
            drained_seq: 5,
        };
        assert_eq!(freeze.step_id(), frozen.step_id());
    }

    #[test]
    fn freeze_has_thaw_as_compensator() {
        let freeze = TransferControl::FreezeSource {
            transfer: T,
            session: S,
            marker_seq: 5,
            dest: NodeId(2),
        };
        assert_eq!(
            compensator_of(freeze),
            Some(TransferControl::ThawSource {
                transfer: T,
                session: S
            })
        );
    }

    #[test]
    fn prepare_compensates_to_abort_and_the_rest_are_terminal() {
        let prepare = TransferControl::PrepareSubscribe {
            transfer: T,
            session: S,
            dest: NodeId(2),
        };
        assert_eq!(
            compensator_of(prepare),
            Some(TransferControl::AbortTransfer {
                transfer: T,
                session: S
            })
        );
        for cmd in all_commands() {
            let compensable = matches!(
                cmd,
                TransferControl::FreezeSource { .. } | TransferControl::PrepareSubscribe { .. }
            );
            assert_eq!(compensator_of(cmd).is_some(), compensable, "{cmd:?}");
        }
    }

    #[test]
    fn every_prepare_reject_maps_to_a_client_reason() {
        let cases = [
            (
                PrepareReject::Spatial(SpatialReject::Obstructed),
                TransferRejectReason::SpatiallyObstructed,
            ),
            (
                PrepareReject::Spatial(SpatialReject::OutOfBounds),
                TransferRejectReason::SpatiallyObstructed,
            ),
            (
                PrepareReject::VersionFloor,
                TransferRejectReason::VersionFloor,
            ),
            (
                PrepareReject::SubscriptionFailed,
                TransferRejectReason::DestinationUnavailable,
            ),
        ];
        for (reject, expected) in cases {
            assert_eq!(reject.client_reason(), expected);
        }
    }
}
