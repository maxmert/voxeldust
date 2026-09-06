//! THE CLIENT SNAPSHOT: what this shard states about the occupants it holds, once per tick.
//!
//! Owns: emit-eligibility (a dot is emitted if it simulates, or is a retained ghost whose hand-off
//! is still live), the restatement of every row into the ONE space this shard speaks in — its own
//! realm's frame — and the fence-stamped per-tick emit to every gateway with an attached session.
//!
//! Does NOT own: authority. No realm authority ⇒ no frames: an unowned shard is SILENT, never
//! speculative, because a frame is a claim about a realm and a shard that has lost its lease has no
//! standing to make one. Nor does it own any pose it emits — it restates, it never computes.

use super::interest::{
    CellKey, InterestHeld, Observer, PlacedRow, cell_of, interest_rule, plan_interest,
};
use super::{
    Dot, Dots, FrameCounter, HandoffHolds, HoldRole, Placements, RealmAuthority, RealmRegions,
    StubConfig, StubStats, is_retained_ghost,
};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut};
use vd_core::frame::transfer_frame;
use vd_core::placement::{PlacementBook, PlacementLedger};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{NodeId, SessionId};
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

/// One emitted row: the wire snap, whether it was PLACED in this shard's own frame (a row that
/// could not be restated ships verbatim under its own label and is not placeable in a cube), and
/// its speed in this frame (the interest lead's other half).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EmittedRow {
    pub(crate) snap: EntitySnap,
    pub(crate) placed: bool,
    pub(crate) speed_mps: f64,
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
) -> Vec<EmittedRow> {
    // Each row reads the ledger book AT ITS OWN STAMP (usually the head — every re-stamped dot is at
    // NOW; a latched dot or a retained ghost carries an older stamp, inside the window by the window's
    // own derivation). A miss is counted and the row ships VERBATIM under its own label — the same
    // counted degrade a foreign-labelled row takes, never a silently substituted instant.
    dots.0
        .values()
        .filter(|d| emits(holds, d))
        .map(|d| {
            // A row already in this frame is PLACED whatever the book says (a same-frame transfer
            // is the identity); a row in another frame is placed only by a successful restatement.
            let (pose, placed) = match placements.at(own_realm, d.pose.universe_tick) {
                Ok(book) => restate_for_own_clients(d.pose, own_frame, book, stats),
                Err(_) => {
                    stats.placement_book_miss += 1;
                    (d.pose, d.pose.frame == own_frame)
                }
            };
            EmittedRow {
                snap: EntitySnap {
                    entity: d.entity,
                    pose,
                },
                placed,
                speed_mps: pose.vel.length(),
            }
        })
        .collect()
}

/// One row's restatement into `own_frame`, with the degrade counted rather than hidden. Monomorphic —
/// both arms covered once here (HR5). A pose already in `own_frame` returns BIT-IDENTICAL —
/// `transfer_frame` short-circuits a same-frame transfer — which is what makes this a no-op for every
/// occupant standing in the shard's own realm. The flag says whether the row is PLACED in this frame.
#[must_use]
fn restate_for_own_clients(
    pose: StampedPose,
    own_frame: FrameRef,
    book: &PlacementBook,
    stats: &mut StubStats,
) -> (StampedPose, bool) {
    match transfer_frame(&pose, own_frame, book) {
        Ok(restated) => (restated, true),
        Err(_) => {
            stats.entity_rows_foreign_labelled += 1;
            (pose, false)
        }
    }
}

/// Emit this tick's snapshot bodies — one per cube of occupants, to the observers that hold that
/// cube — plus the out-of-interest notices. No realm authority ⇒ no frames (an unowned shard is
/// silent, never speculative).
///
/// WHAT LEAVES HERE IS THIS SHARD'S OWN OCCUPANTS AND NOTHING ELSE (Step 5 slice E: the entity relay
/// lane that used to merge neighbouring levels' rows into this feed — and ship this feed up and down
/// the chain — is deleted; occupant poses no longer cross a realm boundary at steady state, SL2). At
/// steady state a bystander sees an occupied sibling realm ITSELF as its occupants' proxy (SL7, the
/// lawful scene lanes). During a crossing: the LEAVER'S OWN client is covered by its dual subs (the
/// own-avatar row is exempt from the client's one-space filter and the cut flips cleanly); a
/// BYSTANDER in the source realm gets the retained ghost's frozen fill until its band-exit Despawn,
/// and then THE REMOVE MESSAGE (D-4(a), minor 14): the teardown emits `EntityRemoved` and the
/// bystander's client EVICTS the figure — it vanishes instead of freezing.
///
/// ★ THE INTEREST (D-9, foundation slice 2, owner-approved 2026-09-05). It used to be ONE body of
/// every occupant to every gateway: three hundred people at a station, and the pilot at the far dock
/// received three hundred rows a tick. Now the rows are sorted into cubes of one reach in this
/// shard's own frame (`interest`), every observer — every dot this shard holds, whatever its
/// authority, at the pose it stands at — holds the cubes around it, and each cube ships as ONE body
/// to the observers holding it: `FrameFor` names them; a cube every observer holds rides the older
/// whole-realm `Frame`. A row that could not be placed in this frame ships to everybody, counted. An
/// occupant that left an observer's hold is told to that observer once (`EntityOutOfInterest`,
/// reliable), because a client evicts a figure only on a sound signal, never on silence.
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
    mut held: ResMut<InterestHeld>,
    mut counter: ResMut<FrameCounter>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    if dots.0.is_empty() {
        return;
    }
    let own_frame = regions.own_frame(config.realm);
    // The rows: the emitting dots, measured from THIS shard's own centre.
    let rows = emitted_entities(
        &dots,
        &holds,
        own_frame,
        &placements.0,
        config.realm,
        &mut stats,
    );
    // The rule: one reach of the largest figure any occupant here draws as, at the dot angle.
    let max_look = dots
        .0
        .values()
        .map(|d| d.look_extent_m)
        .fold(0.0_f64, f64::max);
    let rule = interest_rule(max_look, config.tick_dt_s);
    if held.side_m != rule.side_m {
        *held = InterestHeld {
            side_m: rule.side_m,
            ..InterestHeld::default()
        };
        stats.interest_rule_reset += 1;
    }
    // The observers: every dot this shard holds, at the cube its pose falls in. A dot whose pose
    // cannot be placed in this frame observes from wherever it last was placed by the book miss
    // degrade: its own label's value, read as if in this frame (the same counted degrade as its row).
    let tier = own_frame.tier();
    let observers: Vec<Observer> = dots
        .0
        .iter()
        .map(|(session, d)| {
            let pos = match placements.0.at(config.realm, d.pose.universe_tick) {
                Ok(book) => {
                    transfer_frame(&d.pose, own_frame, book)
                        .unwrap_or(d.pose)
                        .pos
                }
                Err(_) => d.pose.pos,
            };
            Observer {
                session: *session,
                gateway: d.gateway,
                cell: cell_of(
                    pos.delta_m(vd_core::pose::LatticePos::ORIGIN, tier),
                    rule.side_m,
                ),
                speed_mps: d.pose.vel.length(),
            }
        })
        .collect();
    let placed: Vec<PlacedRow> = rows
        .iter()
        .filter(|r| r.placed)
        .map(|r| PlacedRow {
            entity: r.snap.entity,
            cell: cell_of(
                r.snap
                    .pose
                    .pos
                    .delta_m(vd_core::pose::LatticePos::ORIGIN, tier),
                rule.side_m,
            ),
            speed_mps: r.speed_mps,
        })
        .collect();
    let plan = plan_interest(rule, &observers, &placed, &mut held);
    // The rows of each cube, in the deterministic row order.
    let mut by_cell: std::collections::BTreeMap<CellKey, Vec<EntitySnap>> =
        std::collections::BTreeMap::new();
    for (row, cell) in rows
        .iter()
        .filter(|r| r.placed)
        .zip(placed.iter().map(|p| p.cell))
    {
        by_cell.entry(cell).or_default().push(row.snap);
    }
    let unplaced: Vec<EntitySnap> = rows.iter().filter(|r| !r.placed).map(|r| r.snap).collect();
    let gateway_of: std::collections::BTreeMap<SessionId, NodeId> =
        observers.iter().map(|o| (o.session, o.gateway)).collect();
    let mut all_gateways: Vec<NodeId> = observers.iter().map(|o| o.gateway).collect();
    all_gateways.sort_unstable();
    all_gateways.dedup();
    // Every body of one tick shares one frame_id (the sibling-chunk rule: the client merges them and
    // treats only a STRICTLY older frame_id as stale). The counter advances once per tick.
    let frame_id = counter.0;
    counter.0 += 1;
    for (cell, recipients) in &plan.recipients {
        let cell_rows = &by_cell[cell];
        let to_everyone = recipients.len() == observers.len();
        let mut gateways: Vec<NodeId> = recipients.iter().map(|s| gateway_of[s]).collect();
        gateways.sort_unstable();
        gateways.dedup();
        stats.interest_rows_shipped += (cell_rows.len() * recipients.len()) as u64;
        for chunk in partition_entities(cell_rows, config.snapshot_datagram_budget) {
            let snapshot_bytes = snapshot_body(frame_id, &clock, chunk);
            let frame = if to_everyone {
                ShardToGateway::Frame {
                    realm_fence,
                    source_tick: clock.local_tick,
                    snapshot_bytes,
                }
            } else {
                ShardToGateway::FrameFor {
                    realm_fence,
                    source_tick: clock.local_tick,
                    recipients: recipients.clone(),
                    snapshot_bytes,
                }
            };
            stats.interest_bodies += 1;
            push_snapshot(&mut outbox, &frame, &gateways);
        }
    }
    // The degrade: a row this shard could not place ships to everybody, as it always did.
    if !unplaced.is_empty() {
        stats.interest_rows_unplaced += unplaced.len() as u64;
        for chunk in partition_entities(&unplaced, config.snapshot_datagram_budget) {
            let frame = ShardToGateway::Frame {
                realm_fence,
                source_tick: clock.local_tick,
                snapshot_bytes: snapshot_body(frame_id, &clock, chunk),
            };
            stats.interest_bodies += 1;
            push_snapshot(&mut outbox, &frame, &all_gateways);
        }
    }
    // The notices: RETAINED, like the remove message — a lost one is a figure frozen on a screen.
    for (session, entity) in &plan.removals {
        let bytes = postcard::to_allocvec(&ShardToGateway::EntityOutOfInterest {
            realm_fence,
            session: *session,
            entity: *entity,
            at: clock.universe_tick,
        })
        .expect("closed wire enums serialize infallibly");
        outbox.0.push((
            gateway_of[session],
            MsgClass::Control,
            crate::io::bytes(bytes),
            Durability::Retained,
        ));
        stats.interest_removals += 1;
    }
}

/// One snapshot body: the shard always stamps sub 0; the gateway re-tags per session.
fn snapshot_body(frame_id: u64, clock: &ClockSample, entities: Vec<EntitySnap>) -> Vec<u8> {
    let snapshot = SnapshotDatagram {
        sub: SubId(0),
        frame_id,
        source_tick: clock.local_tick,
        universe_tick: clock.universe_tick,
        entities,
    };
    postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly")
}

/// ONE shared body per chunk, cloned (refcount bump) to every gateway that serves a recipient —
/// never an O(entities) copy per gateway (SCALE-1).
fn push_snapshot(outbox: &mut OutboundBox, frame: &ShardToGateway, gateways: &[NodeId]) {
    let bytes = crate::io::bytes(
        postcard::to_allocvec(frame).expect("closed wire enums serialize infallibly"),
    );
    for &gateway in gateways {
        outbox.0.push((
            gateway,
            MsgClass::Snapshot,
            bytes.clone(),
            Durability::Ephemeral,
        ));
    }
}
