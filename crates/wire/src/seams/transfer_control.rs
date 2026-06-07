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
    },
    /// THE route swap: a single atomic store after the directory CAS won.
    CommitAuthority {
        transfer: TransferId,
        session: SessionId,
        new_fence: Fence,
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
            },
            TransferControl::CommitAuthority {
                transfer: T,
                session: S,
                new_fence: Fence(4),
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

    #[test]
    fn commands_and_acks_roundtrip() {
        for cmd in all_commands() {
            let bytes = postcard::to_allocvec(&cmd).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferControl>(&bytes).expect("decode"),
                cmd
            );
        }
        let acks = vec![
            TransferControlAck::Prepared {
                transfer: T,
                result: PrepareResult::Ready,
            },
            TransferControlAck::Prepared {
                transfer: T,
                result: PrepareResult::Rejected(PrepareReject::Spatial(SpatialReject::Obstructed)),
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
        ];
        for ack in acks {
            let bytes = postcard::to_allocvec(&ack).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferControlAck>(&bytes).expect("decode"),
                ack
            );
        }
    }

    #[test]
    fn freeze_has_thaw_as_compensator() {
        let freeze = TransferControl::FreezeSource {
            transfer: T,
            session: S,
            marker_seq: 5,
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
