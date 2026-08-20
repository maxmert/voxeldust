//! THE CLIENT SNAPSHOT: what this shard states about the occupants it holds, once per tick.
//!
//! Owns: emit-eligibility (a dot is emitted if it simulates, or is a retained ghost whose hand-off
//! is still live), the restatement of every row into the ONE space this shard speaks in — its own
//! realm's frame — and the fence-stamped per-tick emit to every gateway with an attached session.
//!
//! Does NOT own: authority. No realm authority ⇒ no frames: an unowned shard is SILENT, never
//! speculative, because a frame is a claim about a realm and a shard that has lost its lease has no
//! standing to make one. Nor does it own any pose it emits — it restates, it never computes.

use super::{
    Dot, Dots, FrameCounter, HandoffHolds, HoldRole, Placements, RealmAuthority, RealmRegions,
    StubConfig, StubStats, is_retained_ghost,
};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut};
use vd_core::NodeId;
use vd_core::frame::transfer_frame;
use vd_core::placement::{PlacementBook, PlacementLedger};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_wire::channels::{EntitySnap, SnapshotDatagram, SubId, partition_entities};
use vd_wire::session_flow::ShardToGateway;

/// EMIT-eligibility (1d.5b.3b → slice F): a dot's pose is emitted if it SIMULATES (Owned — the
/// authority truth) OR is a retained source ghost WHOSE HAND-OFF HOLD IS STILL OPEN — the fill
/// covers exactly the demote→take-over window, and the leaver VANISHES from bystanders' screens at
/// hold closure (the SpawnV2 proof + the remove message), never freezing at the boundary. The
/// DERIVED union rides BITWISE `&`/`|` (never `&&`/`||`, so each operand's false arm stays
/// coverable; HR5). Authority/input/oracle truth remains `simulates()` ALONE — a ghost emits its
/// kinematic mirror but integrates/accepts NOTHING (FG-2).
#[must_use]
pub(crate) fn emits(holds: &HandoffHolds, d: &Dot) -> bool {
    d.authority.simulates()
        | (is_retained_ghost(d) & holds.0.contains_key(&(d.entity, HoldRole::Source)))
}

/// Build the wire entity list from the EMITTING dots (`emits`), every row RESTATED into the one space this
/// shard speaks in: its own realm's frame.
///
/// WHY THE RESTATEMENT EXISTS — it is the owner's "player rendered outside of the planet", on the render
/// side. Everything a client is handed by this shard is documented as "measured from this shard's own
/// centre", and the realm lane keeps that promise: it authors each moving child's box in this shard's own
/// frame. The entity lane did not. It shipped whatever frame the dot happened to be wearing, and an
/// occupant standing in a child realm wears the CHILD's frame — so one shard put two rows about the same
/// place on two feeds, in two spaces one level apart, in the same tick. Measured on the planet shard with
/// an occupant standing at the centre of the area that planet carries: the area's own box left as
/// `(4.9698, 0.5487, 0) @ PlanetCentered{7}` and the occupant standing in it left as `(0,0,0) @
/// AreaLocal{7,7}` — drawn five metres apart, the whole radius the area sits at. Adding where it put that
/// child is arithmetic only this shard holds, so it is done HERE and cannot be done anywhere downstream.
///
/// A row it CANNOT place keeps its label and its value, untouched, and is counted: a fed ghost's pose is
/// authored by a neighbour this shard may know nothing about, and restamping that label would claim the
/// number is measured from here. That is the honest degrade, not the normal path.
///
/// Nothing is composed against a universe root and no tick is re-stamped — both were removed with the fold
/// and stay removed. The only thing added back is one hop, made by the party that authored it.
#[must_use]
pub(crate) fn emitted_entities(
    dots: &Dots,
    holds: &HandoffHolds,
    own_frame: FrameRef,
    placements: &PlacementLedger,
    own_realm: RealmId,
    stats: &mut StubStats,
) -> Vec<EntitySnap> {
    // Each row reads the ledger book AT ITS OWN STAMP (usually the head — every re-stamped dot is at
    // NOW; a latched dot or a retained ghost carries an older stamp, inside the window by the window's
    // own derivation). A miss is counted and the row ships VERBATIM under its own label — the same
    // counted degrade a foreign-labelled row takes, never a silently substituted instant.
    dots.0
        .values()
        .filter(|d| emits(holds, d))
        .map(|d| {
            let pose = match placements.at(own_realm, d.pose.universe_tick) {
                Ok(book) => restate_for_own_clients(d.pose, own_frame, book, stats),
                Err(_) => {
                    stats.placement_book_miss += 1;
                    d.pose
                }
            };
            EntitySnap {
                entity: d.entity,
                pose,
            }
        })
        .collect()
}

/// One row's restatement into `own_frame`, with the degrade counted rather than hidden. Monomorphic —
/// both arms covered once here (HR5). A pose already in `own_frame` returns BIT-IDENTICAL —
/// `transfer_frame` short-circuits a same-frame transfer — which is what makes this a no-op for every
/// occupant standing in the shard's own realm.
#[must_use]
fn restate_for_own_clients(
    pose: StampedPose,
    own_frame: FrameRef,
    book: &PlacementBook,
    stats: &mut StubStats,
) -> StampedPose {
    match transfer_frame(&pose, own_frame, book) {
        Ok(restated) => restated,
        Err(_) => {
            stats.entity_rows_foreign_labelled += 1;
            pose
        }
    }
}

/// Emit one fence-stamped frame per tick to every gateway with an attached session.
/// No realm authority ⇒ no frames (an unowned shard is silent, never speculative).
///
/// WHAT LEAVES HERE IS THIS SHARD'S OWN OCCUPANTS AND NOTHING ELSE (Step 5 slice E: the entity relay
/// lane that used to merge neighbouring levels' rows into this feed — and ship this feed up and down
/// the chain — is deleted; occupant poses no longer cross a realm boundary at steady state, SL2). At
/// steady state a bystander sees an occupied sibling realm ITSELF as its occupants' proxy (SL7, the
/// lawful scene lanes). During a crossing: the LEAVER'S OWN client is covered by its dual subs (the
/// own-avatar row is exempt from the client's one-space filter and the cut flips cleanly); a
/// BYSTANDER in the source realm gets the retained ghost's frozen fill until its band-exit Despawn,
/// and then THE REMOVE MESSAGE (D-4(a), minor 14): the teardown emits `EntityRemoved` and the
/// bystander's client EVICTS the figure — it vanishes instead of freezing. Slice F retimes that
/// emit to hold closure when the retained-ghost fill itself dies. An empty gateway list is a plain
/// early return again: a shard with no attached session has nobody to draw for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_frames(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    dots: Res<Dots>,
    // Slice F: the emit gate consults the hand-off HOLD (a retained ghost fills exactly the
    // demote→take-over window; the leaver vanishes at hold closure).
    holds: Res<HandoffHolds>,
    mut counter: ResMut<FrameCounter>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // EMIT-eligibility is the DERIVED `simulates() | (is_retained_ghost & Source-hold-open)` (slice
    // F — see `emits`): an Owned dot (the authority truth) emits; a retained source ghost emits its
    // last-Owned pose ONLY while its hand-off hold is open, filling exactly the demote→take-over
    // window — the leaver VANISHES at hold closure (the SpawnV2 proof + the remove message). The
    // fed-ghost lane is DELETED (slice F). A Frozen or pre-grant-provisional Ghost emits NOTHING.
    let mut gateways: Vec<NodeId> = dots
        .0
        .values()
        .filter(|d| emits(&holds, d))
        .map(|d| d.gateway)
        .collect();
    gateways.sort_unstable();
    gateways.dedup();
    if gateways.is_empty() {
        return;
    }
    // The emitted rows, every one of them measured from THIS shard's own centre — the single space every
    // feed leaving here is in. The shard still knows nothing about where IT sits. A row whose pose label
    // cannot be restated into this frame is counted (`entity_rows_foreign_labelled` — the ghost feed's
    // foreign write, the §4u corruption slice F deletes) rather than hidden.
    let own_frame = regions.own_frame(config.realm);
    let entities = emitted_entities(
        &dots,
        &holds,
        own_frame,
        &placements.0,
        config.realm,
        &mut stats,
    );
    // Partition BY CONTENT so no datagram exceeds the MTU budget (audit GW-1): a
    // full-world snapshot ships as several independent self-contained frames. Per
    // connection_plane.md §6.3 EVERY chunk of one tick carries the SAME frame_id +
    // celestial_tick (each chunk self-contained, latest-wins) — the client merges
    // them and treats only a STRICTLY older frame_id as stale, so a reordered
    // sibling chunk of the same tick is never dropped. The counter advances once per
    // tick. The shared partitioner is the ONE place every shard type does this.
    let frame_id = counter.0;
    counter.0 += 1;
    for chunk in partition_entities(&entities, config.snapshot_datagram_budget) {
        let snapshot = SnapshotDatagram {
            // The shard always stamps sub 0; the gateway re-tags per session.
            sub: SubId(0),
            frame_id,
            source_tick: clock.local_tick,
            universe_tick: clock.universe_tick,
            entities: chunk,
        };
        let snapshot_bytes =
            postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly");
        let frame = ShardToGateway::Frame {
            realm_fence,
            source_tick: clock.local_tick,
            snapshot_bytes,
        };
        // ONE shared body per chunk, cloned (refcount bump) to every subscribing
        // gateway — never an O(entities) copy per gateway (SCALE-1).
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
        );
        for &gateway in &gateways {
            outbox.0.push((
                gateway,
                MsgClass::Snapshot,
                bytes.clone(),
                Durability::Ephemeral,
            ));
        }
    }
}
