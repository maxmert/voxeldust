//! The dev-control command protocol (HR6): vdctl → client requests, client → vdctl
//! responses, and the pure [`InputAction`] seam that lets this leaf stay free of any
//! `vd-client` dependency — vd-client matches `InputAction` onto its input setters.

use serde::{Deserialize, Serialize};

use crate::predicate::WaitPredicate;
use crate::state::DevState;

/// A command from vdctl to a dev-control client (JSON-lines, internally tagged for
/// readability: `{"cmd":"move","axes":[1.0,0.0,0.0]}`). Not `Copy` — `Screenshot`/
/// `Record` carry an optional `label` string (the agent's name for a capture).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Capture a screenshot (HR6). `at_tick` defers capture until the delivered UNIVERSE
    /// tick reaches that value (a run-stable, join-independent alignment — so a capture
    /// lands on the same world state across runs); `None` captures the next frame.
    /// `label` is the agent's optional name for the shot (recorded in the manifest).
    /// Reply: [`DevResponse::Captured`].
    Screenshot {
        at_tick: Option<u64>,
        label: Option<String>,
    },
    /// Record a reduced-rate frame sequence for `secs` at `fps` (HR6). `label` is the
    /// agent's optional name for the recording. Reply: [`DevResponse::Recorded`].
    Record {
        fps: u32,
        secs: f64,
        label: Option<String>,
    },
    /// Closed-loop: drive the own entity toward a world `target` (within
    /// `arrive_epsilon`) for up to `max_ticks`. Reply: `State` (arrived) / `Timeout`.
    WalkTo {
        target: [f64; 3],
        arrive_epsilon: f64,
        max_ticks: u64,
    },
    /// Closed-loop: turn the own entity to face a world `target` (within
    /// `align_epsilon`) for up to `max_ticks`. Reply: `State` (aligned) / `Timeout`.
    LookAt {
        target: [f64; 3],
        align_epsilon: f64,
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
    /// The single input action this request applies, if it IS one. Reads (`State`),
    /// blocking loops (`WaitUntil`/`WalkTo`/`LookAt` — driven tick-by-tick by the
    /// harness, not one action), and render commands (`Screenshot`/`Record`) are `None`.
    #[must_use]
    pub fn as_input_action(&self) -> Option<InputAction> {
        match self {
            DevRequest::Move { axes } => Some(InputAction::Move(*axes)),
            DevRequest::Look { delta } => Some(InputAction::Look(*delta)),
            DevRequest::Action { bit, pressed } => Some(InputAction::Action {
                bit: *bit,
                pressed: *pressed,
            }),
            DevRequest::Close => Some(InputAction::Close),
            DevRequest::ResetInput => Some(InputAction::ResetInput),
            DevRequest::State
            | DevRequest::WaitUntil { .. }
            | DevRequest::Screenshot { .. }
            | DevRequest::Record { .. }
            | DevRequest::WalkTo { .. }
            | DevRequest::LookAt { .. } => None,
        }
    }

    /// Whether this command requires the explicit `--allow-dev-control` gate (a
    /// privileged state manipulation, vs ordinary input/reads the agent always drives —
    /// walk-to/look-at decompose into ordinary Move/Look, and capture is read-only).
    #[must_use]
    pub fn is_mutating(&self) -> bool {
        match self {
            DevRequest::ResetInput => true,
            DevRequest::Move { .. }
            | DevRequest::Look { .. }
            | DevRequest::Action { .. }
            | DevRequest::Close
            | DevRequest::State
            | DevRequest::WaitUntil { .. }
            | DevRequest::Screenshot { .. }
            | DevRequest::Record { .. }
            | DevRequest::WalkTo { .. }
            | DevRequest::LookAt { .. } => false,
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
    /// A `Screenshot` was captured — the run-relative path + the actual delivered tick.
    Captured { path: String, tick: Option<u64> },
    /// A `Record` finished — the run-relative dir + the number of frames written.
    Recorded { path: String, frames: u64 },
    /// The command was rejected.
    Error { error: DevError },
}

/// Why a dev-control command was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevError {
    /// A mutating command arrived without `--allow-dev-control`.
    NotAllowed,
    /// The request line could not be decoded. RESERVED for undecodable input ONLY —
    /// a decodable-but-not-yet-implemented command is [`DevError::Unsupported`], so this
    /// signal keeps meaning exactly "malformed".
    BadRequest,
    /// The client's bounded input mailbox was full — the command was shed under
    /// back-pressure (an HONEST overload signal, not a silent drop; the running
    /// total also shows in `DevState::dev_commands_dropped`).
    Busy,
    /// The command decoded fine but its handler has not landed yet (the capture +
    /// closed-loop variants `Screenshot`/`Record`/`WalkTo`/`LookAt` arrive with the
    /// render window at T4/T5). Distinct from `BadRequest` so the agent sees "valid but
    /// unavailable", not "malformed".
    Unsupported,
}

/// The maximum 0-based action index: the action channel is a `u32` bitmask, so an index
/// must be in `0..32`. Single-sourced here so the keyboard map (vd-client-harness) and
/// `vdctl` apply the IDENTICAL bound — no divergent `1 << index` (one path panicking on
/// overflow, the other erroring).
pub const MAX_ACTION_INDEX: u32 = 31;

/// The single-bit action mask for a 0-based action `index`, or `None` if the index is
/// out of range (`> MAX_ACTION_INDEX`). THE one place the index→mask transform lives;
/// uses `then` (not `then_some`) so the shift is never evaluated for an out-of-range
/// index (which would overflow).
#[must_use]
pub fn action_bit(index: u32) -> Option<u32> {
    (index <= MAX_ACTION_INDEX).then(|| 1u32 << index)
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
                error: DevError::BadRequest,
            },
            DevResponse::Error {
                error: DevError::Busy,
            },
            DevResponse::Error {
                error: DevError::Unsupported,
            },
        ] {
            let json = serde_json::to_string(&resp).expect("encode");
            assert_eq!(
                serde_json::from_str::<DevResponse>(&json).expect("decode"),
                resp
            );
        }
    }

    #[test]
    fn action_bit_is_a_single_bit_mask_bounded_to_the_u32_channel() {
        assert_eq!(action_bit(0), Some(1));
        assert_eq!(action_bit(3), Some(0b1000));
        assert_eq!(action_bit(MAX_ACTION_INDEX), Some(1u32 << 31));
        // Out of the 0..32 bitmask range — None, never an overflowing shift.
        assert_eq!(action_bit(MAX_ACTION_INDEX + 1), None);
        assert_eq!(action_bit(99), None);
    }

    #[test]
    fn slice3_variants_are_non_input_non_mutating_and_roundtrip() {
        let requests = [
            DevRequest::Screenshot {
                at_tick: Some(100),
                label: Some("after_warp".to_owned()),
            },
            DevRequest::Screenshot {
                at_tick: None,
                label: None,
            },
            DevRequest::Record {
                fps: 15,
                secs: 2.0,
                label: None,
            },
            DevRequest::WalkTo {
                target: [1.0, 2.0, 3.0],
                arrive_epsilon: 0.5,
                max_ticks: 200,
            },
            DevRequest::LookAt {
                target: [0.0, 1.0, -1.0],
                align_epsilon: 0.01,
                max_ticks: 100,
            },
        ];
        for req in requests {
            // Render commands + closed loops are NOT single input actions, NOT privileged.
            assert_eq!(req.as_input_action(), None);
            assert!(!req.is_mutating());
            let json = serde_json::to_string(&req).expect("encode");
            assert_eq!(
                serde_json::from_str::<DevRequest>(&json).expect("decode"),
                req
            );
        }
        // The new responses roundtrip (Captured with + without a tick; Recorded).
        for resp in [
            DevResponse::Captured {
                path: "shots/0001.png".to_owned(),
                tick: Some(100),
            },
            DevResponse::Captured {
                path: "shots/0002.png".to_owned(),
                tick: None,
            },
            DevResponse::Recorded {
                path: "frames/".to_owned(),
                frames: 30,
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
