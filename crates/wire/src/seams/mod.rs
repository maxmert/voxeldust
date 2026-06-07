//! The three FROZEN seam contracts (`docs/design/connection_plane.md` §10): shared
//! types both sides of each seam import, with both-sided conformance tests — so
//! correctness is never defined against a charter that doesn't exist yet.
//!
//! - [`transfer_control`] — the Transfer Saga ⇄ gateway command/ack vocabulary.
//! - [`directory`] — the Ownership Directory key space, records, and CAS API.
//! - [`tickets`] — login/session/resume credential shapes (crypto lands P0.8).

pub mod directory;
pub mod tickets;
pub mod transfer_control;
