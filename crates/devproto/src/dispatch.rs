//! The dev-control command protocol (HR6): vdctl → client requests, client → vdctl
//! responses, and the pure [`InputAction`] seam that lets this leaf stay free of any
//! `vd-client` dependency — vd-client matches `InputAction` onto its input setters.

use serde::{Deserialize, Serialize};

use crate::predicate::WaitPredicate;
use crate::state::DevState;

/// A command from vdctl to a dev-control client (JSON-lines, internally tagged for
/// readability: `{"cmd":"move","axes":[1.0,0.0,0.0]}`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum DevRequest {
    /// Set the held movement axes (forward, strafe, vertical); clamped client-side.
    Move { axes: [f32; 3] },
    /// Accumulate a look delta (yaw, pitch) this frame.
    Look { delta: [f32; 2] },
    /// Set or clear an action (jump/interact/...). `bit` is a u32 BITMASK (a single
    /// set bit), OR-ed into the held action bits; `vdctl` builds it from a 0-based
    /// `<index>` (`1 << index`) so an agent can never press two at once.
    Action { bit: u32, pressed: bool },
    /// Politely close the session.
    Close,
    /// Clear all held input — the one MUTATING dev command (gated by
    /// `--allow-dev-control`; the stand-in for future privileged commands like
    /// board-ship / warp-to / dev-teleport).
    ResetInput,
    /// Read the current decoded delivered state (a non-mutating read).
    State,
    /// Block until a predicate over the delivered state holds, or `max_ticks` pass.
    WaitUntil {
        predicate: WaitPredicate,
        max_ticks: u64,
    },
}

/// A pure input mutation derived from a [`DevRequest`] — THE decoupling seam:
/// vd-client matches this (no `DevRequest`/serde surface) onto its input setters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputAction {
    Move([f32; 3]),
    Look([f32; 2]),
    Action { bit: u32, pressed: bool },
    Close,
    ResetInput,
}

impl DevRequest {
    /// The input action this request applies, if it IS an input command. `State` and
    /// `WaitUntil` are reads, not actions (`None`).
    #[must_use]
    pub fn as_input_action(self) -> Option<InputAction> {
        match self {
            DevRequest::Move { axes } => Some(InputAction::Move(axes)),
            DevRequest::Look { delta } => Some(InputAction::Look(delta)),
            DevRequest::Action { bit, pressed } => Some(InputAction::Action { bit, pressed }),
            DevRequest::Close => Some(InputAction::Close),
            DevRequest::ResetInput => Some(InputAction::ResetInput),
            DevRequest::State | DevRequest::WaitUntil { .. } => None,
        }
    }

    /// Whether this command requires the explicit `--allow-dev-control` gate (a
    /// privileged state manipulation, vs ordinary input the agent always drives).
    #[must_use]
    pub fn is_mutating(self) -> bool {
        match self {
            DevRequest::ResetInput => true,
            DevRequest::Move { .. }
            | DevRequest::Look { .. }
            | DevRequest::Action { .. }
            | DevRequest::Close
            | DevRequest::State
            | DevRequest::WaitUntil { .. } => false,
        }
    }
}

/// A response from the client to vdctl (JSON-lines, internally tagged).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "resp", rename_all = "snake_case")]
pub enum DevResponse {
    /// A command was applied (`Move`/`Look`/`Action`/`Close`/`ResetInput`).
    Ack,
    /// The current delivered state (reply to `State`, or a `WaitUntil` that fired).
    State { state: DevState },
    /// A wait-until elapsed without firing; carries the last state for diagnosis.
    Timeout { state: DevState },
    /// The command was rejected.
    Error { error: DevError },
}

/// Why a dev-control command was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevError {
    /// A mutating command arrived without `--allow-dev-control`.
    NotAllowed,
    /// The request line could not be decoded.
    BadRequest,
    /// The client's bounded input mailbox was full — the command was shed under
    /// back-pressure (an HONEST overload signal, not a silent drop; the running
    /// total also shows in `DevState::dev_commands_dropped`).
    Busy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::{WaitField, WaitOp};
    use crate::state::tests::sample;

    #[test]
    fn input_commands_map_to_actions_reads_do_not() {
        assert_eq!(
            DevRequest::Move {
                axes: [1.0, 0.0, 0.0]
            }
            .as_input_action(),
            Some(InputAction::Move([1.0, 0.0, 0.0]))
        );
        assert_eq!(
            DevRequest::Look { delta: [0.1, -0.2] }.as_input_action(),
            Some(InputAction::Look([0.1, -0.2]))
        );
        assert_eq!(
            DevRequest::Action {
                bit: 4,
                pressed: true
            }
            .as_input_action(),
            Some(InputAction::Action {
                bit: 4,
                pressed: true
            })
        );
        assert_eq!(
            DevRequest::Close.as_input_action(),
            Some(InputAction::Close)
        );
        assert_eq!(
            DevRequest::ResetInput.as_input_action(),
            Some(InputAction::ResetInput)
        );
        assert_eq!(DevRequest::State.as_input_action(), None);
        assert_eq!(
            DevRequest::WaitUntil {
                predicate: WaitPredicate {
                    field: WaitField::Active,
                    op: WaitOp::Eq,
                    value: 1,
                },
                max_ticks: 100,
            }
            .as_input_action(),
            None
        );
    }

    #[test]
    fn only_reset_input_is_a_mutating_command() {
        assert!(DevRequest::ResetInput.is_mutating());
        assert!(!DevRequest::Move { axes: [0.0; 3] }.is_mutating());
        assert!(!DevRequest::Look { delta: [0.0; 2] }.is_mutating());
        assert!(
            !DevRequest::Action {
                bit: 1,
                pressed: false
            }
            .is_mutating()
        );
        assert!(!DevRequest::Close.is_mutating());
        assert!(!DevRequest::State.is_mutating());
        assert!(
            !DevRequest::WaitUntil {
                predicate: WaitPredicate {
                    field: WaitField::Active,
                    op: WaitOp::Eq,
                    value: 1,
                },
                max_ticks: 1,
            }
            .is_mutating()
        );
    }

    #[test]
    fn requests_and_responses_roundtrip_through_json() {
        let req = DevRequest::Move {
            axes: [1.0, 0.0, -1.0],
        };
        let json = serde_json::to_string(&req).expect("encode");
        assert!(json.contains("\"cmd\":\"move\""));
        assert_eq!(
            serde_json::from_str::<DevRequest>(&json).expect("decode"),
            req
        );

        for resp in [
            DevResponse::Ack,
            DevResponse::State { state: sample() },
            DevResponse::Timeout { state: sample() },
            DevResponse::Error {
                error: DevError::NotAllowed,
            },
            DevResponse::Error {
                error: DevError::Busy,
            },
        ] {
            let json = serde_json::to_string(&resp).expect("encode");
            assert_eq!(
                serde_json::from_str::<DevResponse>(&json).expect("decode"),
                resp
            );
        }
    }
}
