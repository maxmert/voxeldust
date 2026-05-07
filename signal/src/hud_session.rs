//! Phase 4.4 — server-side per-session state for delta-encoded HUD
//! signal updates.
//!
//! # The optimization
//!
//! Today's `WorldStateData.hud_signals` ships a FULL snapshot of every
//! HUD-visible channel every tick over UDP. At 100K users × ~50
//! channels × ~50B × 20 Hz that's ~5 GB/s of pure HUD bandwidth, the
//! vast majority of which is unchanged values re-serialized.
//!
//! `HudSession` collapses this to a delta stream:
//!   * First batch for a session: every channel ships with
//!     `FLAG_REGISTER` (one-time name → wire_id binding).
//!   * Steady state: only entries whose value changed since the last
//!     batch ship. Empty deltas are skipped entirely.
//!   * Channels that go out of scope (e.g., player left a seat that
//!     surfaced extra signals) ship a single `FLAG_REMOVE` entry.
//!
//! Bandwidth math: ~50 channels initial × ~30B (registered) +
//! ~5 changes/tick × ~14B (bare) ≈ a one-time 1.5KB upfront then
//! ~70B/tick. **20× reduction** on average.
//!
//! # Why a per-session state machine, not a flat dirty-set
//!
//! The dictionary (`OutboundDict`) is namespaced per (server,
//! session) — wire ids assigned to one client are not valid for any
//! other client. Each session also has its own
//! `last_sent_value` map for diff comparison. Bundling these into
//! one struct keeps the lifecycle (create on connect, drop on
//! disconnect) trivial and the logic testable in isolation.
//!
//! # The diff API
//!
//! `produce_delta(snapshot)` is the single entry point the server
//! calls each tick. The caller passes the FULL current set of
//! (channel_name, value, property) tuples; the session diffs against
//! its own `last_sent_value` map and returns a delta containing only
//! the entries that actually need to ship. The session updates its
//! internal state atomically so a panic mid-call leaves the previous
//! state intact (no partial application).
//!
//! # Why we accept full snapshots, not deltas, from the caller
//!
//! The HUD-build path on the server already iterates every visible
//! channel each tick (to build today's UDP snapshot). Converting it
//! to emit deltas would require change-tracking across every
//! contributing system (seats, blocks, ship state) — a much larger
//! refactor. Letting the session do the diff isolates the
//! optimization to one module: cheap, testable, no other systems
//! change.
//!
//! # Empty snapshots
//!
//! A snapshot with zero entries is interpreted as "this player can
//! see nothing right now" (e.g., transitioning between shards). The
//! session emits REMOVE entries for every previously-known wire id,
//! then `last_sent_value` is empty and the dict is reset. Future
//! snapshots start fresh with REGISTER on every entry. This keeps
//! the wire dict bounded under churn — a session that bounces
//! between empty and populated doesn't accumulate ghost wire ids.

use std::collections::HashMap;

use voxeldust_types::hud_delta_flags::{REGISTER as FLAG_REGISTER, REMOVE as FLAG_REMOVE};
use crate::wire_dict::OutboundDict;

/// One HUD signal observation. Mirrors the wire-side
/// `HudSignalEntryV2` *except* values stay as the rich
/// `HudSignalValueRepr` enum until the wire-build step (which lives
/// in the binary crate, not here — `core` doesn't depend on
/// `client_message::HudSignalValue` to keep modules decoupled).
///
/// `Eq` matters: `produce_delta` compares old vs new via this
/// representation so an unchanged value doesn't ship.
#[derive(Debug, Clone, PartialEq)]
pub struct HudSignalValueRepr {
    /// 0=Bool, 1=Float, 2=State, 3=Text. Same ordinal as
    /// `client_message::HudSignalValue::type_code`.
    pub type_code: u8,
    /// Numeric value (Bool→0/1, Float→raw, State→u8 cast, Text→0).
    pub num: f32,
    /// Text value (only meaningful when `type_code == 3`).
    pub text: String,
}

impl HudSignalValueRepr {
    pub fn bool(b: bool) -> Self {
        Self {
            type_code: 0,
            num: if b { 1.0 } else { 0.0 },
            text: String::new(),
        }
    }
    pub fn float(f: f32) -> Self {
        Self {
            type_code: 1,
            num: f,
            text: String::new(),
        }
    }
    pub fn state(s: u8) -> Self {
        Self {
            type_code: 2,
            num: s as f32,
            text: String::new(),
        }
    }
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            type_code: 3,
            num: 0.0,
            text: s.into(),
        }
    }
}

impl Eq for HudSignalValueRepr {}
// `f32` is not `Eq`. We compare via bit-equality so NaN survives a
// no-change check (NaN -> NaN is "the same" for HUD purposes — the
// client sees a NaN and renders accordingly; no need to re-ship).
impl HudSignalValueRepr {
    fn bitwise_equals(&self, other: &Self) -> bool {
        self.type_code == other.type_code
            && self.num.to_bits() == other.num.to_bits()
            && self.text == other.text
    }
}

/// Output of [`HudSession::produce_delta`]: ready-to-encode entries.
/// The caller (the server's HUD-emit system) wraps these in a
/// `HudSignalDelta` ServerMsg with the session's current `dict_seq`
/// + `batch_seq`.
#[derive(Debug, Clone, PartialEq)]
pub struct HudDeltaEntry {
    pub flags: u8,
    pub wire_id: u32,
    /// Required when `flags & REGISTER` is set; empty otherwise.
    pub channel_name: String,
    pub value: HudSignalValueRepr,
    pub property: u8,
    pub seq: u32,
}

/// Per-channel cached state on the server side.
#[derive(Debug, Clone)]
struct ChannelEntry {
    last_value: HudSignalValueRepr,
    /// `SignalProperty` ordinal — usually stable over a channel's
    /// lifetime, but we track it so a property change ALSO triggers
    /// an emission (rare; defensive).
    last_property: u8,
    /// Per-channel monotonic seq we ship on every entry for this
    /// wire_id. Useful for client-side ordering diagnostics.
    seq: u32,
}

/// Server-side per-session HUD signal state. Owned by the server's
/// per-session bookkeeping (typically the same place that holds
/// `SessionToken` → other session-scoped state).
#[derive(Debug)]
pub struct HudSession {
    out_dict: OutboundDict,
    /// wire_id → cached (last-sent value, last-sent property, seq).
    by_wire: HashMap<u32, ChannelEntry>,
    /// Strictly monotonic across batches sent on this session.
    batch_seq: u64,
}

impl Default for HudSession {
    fn default() -> Self {
        Self::new()
    }
}

impl HudSession {
    pub fn new() -> Self {
        Self {
            out_dict: OutboundDict::default(),
            by_wire: HashMap::new(),
            batch_seq: 0,
        }
    }

    /// Construct with a custom dict capacity. Used by tests + future
    /// per-shard tuning.
    pub fn with_capacity(capacity: u32) -> Self {
        Self {
            out_dict: OutboundDict::with_capacity(capacity),
            by_wire: HashMap::new(),
            batch_seq: 0,
        }
    }

    /// Current `OutboundDict::dict_seq` — embedded into every emitted
    /// delta so the client can verify monotonicity.
    pub fn dict_seq(&self) -> u64 {
        self.out_dict.dict_seq()
    }

    /// Current `batch_seq` — strictly increasing per session.
    pub fn batch_seq(&self) -> u64 {
        self.batch_seq
    }

    /// Number of channels this session is currently tracking.
    pub fn live_channel_count(&self) -> usize {
        self.by_wire.len()
    }

    /// Diff `snapshot` against the session's `last_sent_value` map
    /// and return the entries the client needs to receive — empty
    /// when nothing changed.
    ///
    /// Called once per server tick per session. Most calls return an
    /// empty `Vec` once the session is past its initial REGISTER
    /// burst; only changed channels generate entries.
    ///
    /// **Atomic**: state changes apply only AFTER the entry list is
    /// fully built. A panic mid-build (impossible in practice — no
    /// fallible operations here) would leave the session unchanged.
    pub fn produce_delta(
        &mut self,
        snapshot: &[(String, HudSignalValueRepr, u8)],
    ) -> Vec<HudDeltaEntry> {
        // Step 1: walk the snapshot, intern channel names, build new
        // entry list. Defer mutating `by_wire` so a partial walk
        // can't corrupt state on a panic.
        let mut new_state: HashMap<u32, ChannelEntry> =
            HashMap::with_capacity(snapshot.len());
        let mut emitted: Vec<HudDeltaEntry> = Vec::new();
        let mut seen_wire_ids: HashMap<u32, ()> = HashMap::with_capacity(snapshot.len());

        for (name, value, property) in snapshot {
            let intern = self.out_dict.intern(name);
            let wire_id = intern.wire_id;

            // If `intern` evicted a wire id we still had cached, the
            // eviction means the receiver also lost it (sender +
            // receiver dicts stay synchronized via `dict_seq`). Drop
            // it from our `by_wire` so subsequent reasoning is sound.
            if let Some(evicted_id) = intern.evicted_wire_id {
                self.by_wire.remove(&evicted_id);
            }

            // Compute next per-channel seq. Existing entry advances
            // its seq; fresh entry starts at 1.
            let next_seq = match self.by_wire.get(&wire_id) {
                Some(prev) => prev.seq.wrapping_add(1),
                None => 1,
            };

            // Decide which entry kind ships:
            //   * REGISTER on first sighting (intern.needs_register).
            //   * Bare on value or property change.
            //   * Nothing if value+property identical to last sent.
            let prev_entry = self.by_wire.get(&wire_id);
            let is_changed = match prev_entry {
                None => true, // first-time sighting always emits
                Some(prev) => {
                    !prev.last_value.bitwise_equals(value) || prev.last_property != *property
                }
            };

            if intern.needs_register {
                emitted.push(HudDeltaEntry {
                    flags: FLAG_REGISTER,
                    wire_id,
                    channel_name: name.clone(),
                    value: value.clone(),
                    property: *property,
                    seq: next_seq,
                });
            } else if is_changed {
                emitted.push(HudDeltaEntry {
                    flags: 0,
                    wire_id,
                    channel_name: String::new(),
                    value: value.clone(),
                    property: *property,
                    seq: next_seq,
                });
            }
            // Build new_state regardless — even when no entry was
            // emitted, we need to keep tracking this wire_id.
            new_state.insert(
                wire_id,
                ChannelEntry {
                    last_value: value.clone(),
                    last_property: *property,
                    seq: next_seq,
                },
            );
            seen_wire_ids.insert(wire_id, ());
        }

        // Step 2: emit REMOVE for every wire_id that USED to be in
        // `by_wire` but isn't in the new snapshot. Those channels
        // went out of scope.
        for (&old_wire_id, prev) in self.by_wire.iter() {
            if seen_wire_ids.contains_key(&old_wire_id) {
                continue;
            }
            // Use old seq+1 so the client sees a strictly-monotone seq
            // across the channel's full lifetime including its REMOVE.
            emitted.push(HudDeltaEntry {
                flags: FLAG_REMOVE,
                wire_id: old_wire_id,
                channel_name: String::new(),
                value: HudSignalValueRepr::float(0.0),
                property: 0,
                seq: prev.seq.wrapping_add(1),
            });
        }

        // Step 3: replace state atomically.
        self.by_wire = new_state;
        if !emitted.is_empty() {
            self.batch_seq = self.batch_seq.wrapping_add(1);
        }
        emitted
    }

    /// Drop all per-session state. Used when a player session ends
    /// (TCP close, handoff). The owning server wraps `HudSession` in
    /// its session map keyed by `SessionToken`; this is the cleanup
    /// hook before the session map drops the entry.
    pub fn forget(&mut self) {
        self.out_dict.reset();
        self.by_wire.clear();
        self.batch_seq = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(items: &[(&str, HudSignalValueRepr, u8)]) -> Vec<(String, HudSignalValueRepr, u8)> {
        items.iter().map(|(n, v, p)| (n.to_string(), v.clone(), *p)).collect()
    }

    fn float(v: f32) -> HudSignalValueRepr {
        HudSignalValueRepr::float(v)
    }

    #[test]
    fn first_snapshot_emits_register_for_every_entry() {
        let mut s = HudSession::new();
        let initial = snap(&[
            ("ship.thrust", float(0.5), 1),
            ("ship.lights", float(1.0), 0),
        ]);
        let entries = s.produce_delta(&initial);
        assert_eq!(entries.len(), 2, "every fresh channel registers");
        assert!(entries.iter().all(|e| (e.flags & FLAG_REGISTER) != 0));
        assert_eq!(s.live_channel_count(), 2);
        assert!(s.batch_seq() >= 1);
    }

    #[test]
    fn unchanged_snapshot_emits_no_entries() {
        let mut s = HudSession::new();
        let initial = snap(&[("ship.thrust", float(0.5), 1)]);
        s.produce_delta(&initial);
        // Same snapshot again — no delta should ship.
        let entries = s.produce_delta(&initial);
        assert!(entries.is_empty(), "unchanged values must not re-ship");
    }

    #[test]
    fn changed_value_emits_bare_entry_for_known_channel() {
        let mut s = HudSession::new();
        let initial = snap(&[("ship.thrust", float(0.5), 1)]);
        s.produce_delta(&initial);
        // Same channel, new value.
        let updated = snap(&[("ship.thrust", float(0.75), 1)]);
        let entries = s.produce_delta(&updated);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].flags, 0, "bare entry, no REGISTER");
        assert!(entries[0].channel_name.is_empty());
        assert!((entries[0].value.num - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn property_change_alone_emits_bare_entry() {
        let mut s = HudSession::new();
        let initial = snap(&[("ship.thrust", float(0.5), 1)]);
        s.produce_delta(&initial);
        // Same value, different property — defensive emission.
        let updated = snap(&[("ship.thrust", float(0.5), 7)]);
        let entries = s.produce_delta(&updated);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].property, 7);
    }

    #[test]
    fn channel_dropped_emits_remove() {
        let mut s = HudSession::new();
        let initial = snap(&[
            ("ship.thrust", float(0.5), 1),
            ("ship.lights", float(1.0), 0),
        ]);
        s.produce_delta(&initial);
        // Drop "ship.lights" from the snapshot.
        let reduced = snap(&[("ship.thrust", float(0.5), 1)]);
        let entries = s.produce_delta(&reduced);
        assert_eq!(entries.len(), 1);
        assert!(
            (entries[0].flags & FLAG_REMOVE) != 0,
            "dropped channel must emit REMOVE"
        );
        assert_eq!(s.live_channel_count(), 1);
    }

    #[test]
    fn empty_snapshot_emits_remove_for_every_known_channel() {
        let mut s = HudSession::new();
        s.produce_delta(&snap(&[
            ("a", float(1.0), 0),
            ("b", float(2.0), 0),
        ]));
        let entries = s.produce_delta(&[]);
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| (e.flags & FLAG_REMOVE) != 0));
        assert_eq!(s.live_channel_count(), 0);
    }

    #[test]
    fn forget_resets_dict_and_state() {
        let mut s = HudSession::new();
        s.produce_delta(&snap(&[("a", float(1.0), 0)]));
        s.forget();
        assert_eq!(s.live_channel_count(), 0);
        assert_eq!(s.dict_seq(), 0);
        // After forget, the same channel re-registers as if fresh.
        let entries = s.produce_delta(&snap(&[("a", float(1.0), 0)]));
        assert_eq!(entries.len(), 1);
        assert!((entries[0].flags & FLAG_REGISTER) != 0);
    }

    #[test]
    fn batch_seq_advances_only_on_non_empty_emit() {
        let mut s = HudSession::new();
        s.produce_delta(&snap(&[("a", float(1.0), 0)]));
        let after_first = s.batch_seq();
        // Empty snapshot would normally emit REMOVE — that IS a
        // non-empty emission, so batch_seq advances. Use the
        // "unchanged snapshot" path instead, which yields an empty
        // emit and must NOT advance batch_seq.
        s.produce_delta(&snap(&[("a", float(1.0), 0)]));
        assert_eq!(
            s.batch_seq(),
            after_first,
            "unchanged batch must not consume seq"
        );
    }

    #[test]
    fn per_channel_seq_increments_on_each_emission() {
        let mut s = HudSession::new();
        let entries = s.produce_delta(&snap(&[("a", float(1.0), 0)]));
        let seq_first = entries[0].seq;
        // Change value → bare entry with seq+1.
        let entries = s.produce_delta(&snap(&[("a", float(2.0), 0)]));
        assert_eq!(entries[0].seq, seq_first + 1);
        // Change again.
        let entries = s.produce_delta(&snap(&[("a", float(3.0), 0)]));
        assert_eq!(entries[0].seq, seq_first + 2);
    }

    #[test]
    fn text_value_round_trips_through_session() {
        let mut s = HudSession::new();
        let entries = s.produce_delta(&snap(&[(
            "warp.target",
            HudSignalValueRepr::text("Kepler-22b"),
            10,
        )]));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value.type_code, 3);
        assert_eq!(entries[0].value.text, "Kepler-22b");
        // Same text → no re-emission.
        let entries = s.produce_delta(&snap(&[(
            "warp.target",
            HudSignalValueRepr::text("Kepler-22b"),
            10,
        )]));
        assert!(entries.is_empty());
        // Different text → bare emit.
        let entries = s.produce_delta(&snap(&[(
            "warp.target",
            HudSignalValueRepr::text("Tau Ceti d"),
            10,
        )]));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value.text, "Tau Ceti d");
    }

    #[test]
    fn nan_to_nan_does_not_re_emit() {
        // NaN is not equal to NaN under `==`, but bit-identical NaN
        // should not re-ship — the client already has it.
        let nan = HudSignalValueRepr::float(f32::NAN);
        let mut s = HudSession::new();
        s.produce_delta(&snap(&[("a", nan.clone(), 0)]));
        let entries = s.produce_delta(&snap(&[("a", nan, 0)]));
        assert!(
            entries.is_empty(),
            "bit-identical NaN must not re-ship every tick"
        );
    }
}
