//! THE HAND-OFF HOLD LEDGER: what a shard still speaks for after it has stopped owning it.
//!
//! Owns: the per-`(subject, end)` holds a shard opens when it is a party to a crossing, their
//! budget, and the two questions they answer — does this realm still count the subject at its Empty
//! gate, and does its retained ghost still emit. Without them a shard goes silent the instant it
//! stops owning a subject, which is exactly the moment a crossing is happening and the moment
//! seamlessness is strictest.
//!
//! Does NOT own: a pose. The subject's pose is the retained ghost dot's, full stop — a copy kept
//! here would be a second store that only provably agrees today. RAM-only by design: a hold is a
//! live-crossing courtesy, and the crossing's own machinery carries the correctness.

use super::Dot;
use bevy_ecs::prelude::Resource;
use std::collections::BTreeMap;
use vd_core::{EntityId, Fence, TickId};

/// WHICH END of a hand-off this shard is holding — never a shard KIND (HR3). The compound key
/// `(EntityId, HoldRole)` is what lets a SAME-NODE re-home hold BOTH ends at once; a bare entity key
/// could not express that, and a same-node re-home is the common case on a co-hosting shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HoldRole {
    /// This shard is handing the subject AWAY.
    Source,
    /// This shard is RECEIVING the subject.
    Dest,
}

/// One in-progress hand-off this shard is a party to.
///
/// WHY IT EXISTS: without it a shard goes SILENT the instant it stops owning a subject — it stops keeping
/// the destination's children warm (the keep-alive is cleared when it applies the ordered `Demote`, see
/// `on_saga_demote`) and its own realm can go EMPTY mid-window, muting the SL7 `ChildLive` beat its parent's
/// liveness rides on (`speaks_for` is what keeps a held subject counting as presence until the take-over
/// lands). Since the render clock started driving what is drawn, a realm whose feed stops now FREEZES at
/// its last pose rather than drifting — and it would freeze at exactly the moment a crossing happens,
/// which is when the seamlessness rule is strictest.
///
/// ARMED BY `handoff_hold_ttl_ticks`: every helper below returns immediately at `0`, so a shard with an
/// unarmed budget behaves exactly as it did before the ledger existed.
///
/// NO POSE HERE (FG-2). The subject's pose is the retained ghost dot's, full stop — the dot is the single
/// pose truth (slice F deleted its last cross-shard writer, `refresh_source_ghost`, with the whole fed-ghost
/// pose feed; the surviving writers are the input integrator and the crossing adopt). A copy kept here would
/// be a second store that would only provably agree today, which is precisely the kind of agreement that
/// stops holding the day someone adds a writer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandoffHold {
    /// The fence the TAKE-OVER will carry. A hold is closed by a FENCE COMPARE against this, never by a
    /// boolean latch — a PRIOR crossing's replayed take-over must be structurally refused rather than
    /// silently honoured, which on an at-least-once mesh is a reachable delivery, not a hypothetical.
    pub takeover_fence: Fence,
    /// The local tick this hold OPENED. The budget is TOTAL from here: a same-fence re-open leaves it
    /// untouched, so a saga re-drive heartbeat can never keep a realm alive forever.
    pub opened_at: TickId,
}

/// The in-progress hand-offs this shard is a party to, keyed by `(subject, which end)`. RAM-only by
/// design — a hold is a live-crossing courtesy, not a durable fact; a shard restart legitimately drops
/// every one of them and the crossing's own machinery (which IS durable) carries the correctness.
#[derive(Resource, Default, Debug)]
pub struct HandoffHolds(pub BTreeMap<(EntityId, HoldRole), HandoffHold>);

/// What [`open_hold`] did — returned rather than inferred so a test asserts the exact arm instead of a
/// shape. The five arms are the whole state space and each is separately reachable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldOutcome {
    /// `ttl == 0` — the inert arm every rig takes today.
    Inert,
    /// No hold existed for this `(subject, end)`.
    Opened,
    /// A hold existed at an OLDER take-over fence — a genuinely newer crossing replaces it.
    Superseded,
    /// A hold existed at the SAME fence — a re-delivered Demote for the crossing already recorded. The
    /// budget is deliberately NOT restarted; that is the whole immortality guard.
    Refreshed,
    /// A hold existed at a NEWER fence — a delayed redelivery of a PRIOR crossing arriving after a second
    /// crossing already opened a fresh hold. Refused, because honouring it would rewind the ledger to a
    /// crossing that has already been superseded.
    RefusedStale,
}

/// Open (or update) a hold. Monomorphic; every arm separately reachable and separately asserted. The
/// compare reads the stored FENCE (not the entry) so the same-fence arm has nothing to bind and nothing
/// to write — THE IMMORTALITY GUARD is that `opened_at` survives a re-delivered Demote untouched, so a
/// saga that re-drives its demote every tick can never extend its own budget.
pub(crate) fn open_hold(
    holds: &mut HandoffHolds,
    key: (EntityId, HoldRole),
    takeover_fence: Fence,
    now: TickId,
    ttl: u32,
) -> HoldOutcome {
    if ttl == 0 {
        return HoldOutcome::Inert; // the disarmed arm: nothing is ever inserted
    }
    let opened = HandoffHold {
        takeover_fence,
        opened_at: now,
    };
    match holds.0.get(&key).map(|h| h.takeover_fence) {
        None => {
            holds.0.insert(key, opened);
            HoldOutcome::Opened
        }
        Some(existing) if takeover_fence > existing => {
            holds.0.insert(key, opened);
            HoldOutcome::Superseded
        }
        Some(existing) if takeover_fence == existing => HoldOutcome::Refreshed,
        Some(_) => HoldOutcome::RefusedStale,
    }
}

/// Close a hold on a POSITIVE take-over proof: the fence must match exactly. A stale proof (a replayed
/// take-over from a superseded crossing) leaves the hold standing.
///
/// The proof is the destination's ghost feed, which a SAME-NODE re-home does not send (there is nobody to
/// send it to). Such a hold therefore runs to its budget rather than closing early — harmless, because on
/// one shard the subject is simply Owned again by the receiving realm, so both consumers already answer
/// yes through ownership and the stale entry only sits there until the prune takes it.
pub(crate) fn close_hold_at_fence(
    holds: &mut HandoffHolds,
    key: (EntityId, HoldRole),
    proof: Fence,
) -> bool {
    match holds.0.get(&key) {
        None => false,
        Some(h) => {
            if h.takeover_fence == proof {
                holds.0.remove(&key);
                true
            } else {
                false
            }
        }
    }
}

/// Close a hold unconditionally — the terminals where the subject is simply gone (a detach, an abort, a
/// realm losing its lease). No fence to compare against, because there is no take-over to prove.
pub(crate) fn close_hold(holds: &mut HandoffHolds, key: (EntityId, HoldRole)) -> bool {
    holds.0.remove(&key).is_some()
}

/// Whether a hold is still inside its budget. `saturating_sub` makes a BACKWARDS clock read as age zero
/// (still live) rather than wrapping to a colossal age and expiring everything at once.
pub(crate) fn hold_live(h: &HandoffHold, now: TickId, ttl: u32) -> bool {
    now.0.saturating_sub(h.opened_at.0) < u64::from(ttl)
}

/// Drop every expired hold, returning the DROPPED KEYS. Called UNGATED at the top of the inbound
/// pass so an expired hold is reclaimed even on a shard that has lost its lease and is doing
/// nothing else. The keys go back to the caller because an expired SOURCE hold is a leaver-vanish
/// moment (slice F): the retained ghost stops emitting the instant the hold dies, and the
/// bystanders' clients must be told — the TTL backstop covers the take-over proof that never
/// arrives (a logout mid-crossing kills the dest dot before it can send one).
pub(crate) fn prune_holds(
    holds: &mut HandoffHolds,
    now: TickId,
    ttl: u32,
) -> Vec<(EntityId, HoldRole)> {
    let mut dropped = Vec::new();
    holds.0.retain(|key, h| {
        let live = hold_live(h, now, ttl);
        if !live {
            dropped.push(*key);
        }
        live
    });
    dropped
}

/// Is this shard STILL PARTY to a hand-off of `entity` — i.e. has it let go of the subject but not yet
/// seen the take-over land? THE ARMED CONSUMER SIDE of the ledger, and the answer to two questions that
/// used to be answered by ownership alone: does this realm still count the subject at its Empty gate
/// (so its one-bit `ChildLive` heartbeat keeps beating — no pose crosses; the per-occupant position
/// up-relay is DELETED, Step 5 slice D), and does this realm still count itself occupied.
///
/// Freshness is re-checked here rather than trusted from the inbound prune, because the prune runs on
/// inbound and this runs during the authoring pass — a hold that aged out between them must read expired
/// at the moment it is USED, never one tick late.
#[must_use]
pub(crate) fn handing_over(holds: &HandoffHolds, entity: EntityId, now: TickId, ttl: u32) -> bool {
    holds
        .0
        .get(&(entity, HoldRole::Source))
        .is_some_and(|h| hold_live(h, now, ttl))
}

/// Does this shard still SPEAK FOR this occupant — because it owns them, or because it is mid-hand-off
/// of them? ONE answer serving both places that used to ask "do we own them": whether this realm's
/// one-bit occupancy heartbeat keeps beating to its parent, and whether this realm considers itself
/// occupied (the Empty gate). Both are LOCAL — no pose leaves the shard (the per-occupant up-relay is
/// DELETED, Step 5 slice D). Deriving them from one predicate is what stops a realm reporting itself
/// empty in the same tick its bit still says someone is here.
///
/// Bitwise `|`, never `||` — both operands' false arms stay coverable (HR5). At a disarmed budget the
/// right operand is constant-false and this reduces to the old `simulates()` filter exactly.
#[must_use]
pub(crate) fn speaks_for(holds: &HandoffHolds, d: &Dot, now: TickId, ttl: u32) -> bool {
    d.authority.simulates() | handing_over(holds, d.entity, now, ttl)
}
