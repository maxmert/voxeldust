//! THE INTEREST (D-9, foundation slice 2): a client's bytes grow with its neighbours, never with
//! the realm's population. The pure plan is pinned first; then the emission is driven through the
//! real shard schedule with real attaches, and every body, recipient and notice is read back.

use super::*;
use crate::stub::interest::{
    CellKey, InterestHeld, InterestRule, Observer, PlacedRow, cell_of, cell_steps, interest_rule,
    plan_interest, take_steps,
};
use std::collections::BTreeMap;

// ---- the pure plan --------------------------------------------------------------------------

#[test]
fn a_cube_key_floors_toward_minus_infinity_and_steps_are_the_largest_axis_difference() {
    assert_eq!(
        cell_of(DVec3::new(0.5, -0.5, 99.0), 10.0),
        CellKey(0, -1, 9)
    );
    assert_eq!(
        cell_of(DVec3::new(-0.0, 10.0, -10.0), 10.0),
        CellKey(0, 1, -1)
    );
    assert_eq!(cell_steps(CellKey(0, 0, 0), CellKey(2, -1, 1)), 2);
    assert_eq!(cell_steps(CellKey(5, 5, 5), CellKey(5, 5, 5)), 0);
    assert_eq!(cell_steps(CellKey(0, 0, 0), CellKey(0, 0, -7)), 7);
}

#[test]
fn the_rule_is_one_reach_of_the_largest_look_and_a_horizon_of_buffer_plus_tick() {
    let rule = interest_rule(0.5, 0.05);
    let expected =
        vd_core::geometry::visibility_reach_m(0.5, vd_core::geometry::drawable_theta_min_rad());
    assert_eq!(rule.side_m, expected);
    assert!(
        (rule.side_m - 869.1).abs() < 1.0,
        "a one-metre figure is one pixel at ~869 m: {}",
        rule.side_m
    );
    assert!((rule.horizon_s - 0.17).abs() < 1e-12);
    // A standing crowd takes one step; a closing speed that covers half a cube in the horizon
    // takes one more; a kilometre a second, each way, takes one more too (340 m in the horizon).
    assert_eq!(take_steps(rule, 0.0), 1);
    assert_eq!(take_steps(rule, rule.side_m * 0.5 / rule.horizon_s), 2);
    assert_eq!(take_steps(rule, 2000.0), 2);
}

fn obs(session: u64, cell: CellKey) -> Observer {
    Observer {
        session: SessionId(u128::from(session)),
        gateway: GATEWAY,
        cell,
        speed_mps: 0.0,
    }
}

fn row(entity: u64, cell: CellKey) -> PlacedRow {
    PlacedRow {
        entity: EntityId(u128::from(entity)),
        cell,
        speed_mps: 0.0,
    }
}

const RULE: InterestRule = InterestRule {
    side_m: 100.0,
    horizon_s: 0.17,
};

#[test]
fn the_plan_takes_the_cubes_within_one_step_and_names_each_cubes_holders() {
    let mut held = InterestHeld::default();
    let observers = [obs(1, CellKey(0, 0, 0)), obs(2, CellKey(5, 0, 0))];
    let rows = [
        row(11, CellKey(0, 0, 0)),
        row(12, CellKey(1, 1, 0)),
        row(13, CellKey(2, 0, 0)),
        row(21, CellKey(5, 0, 0)),
    ];
    let plan = plan_interest(RULE, &observers, &rows, &mut held);
    let s1 = SessionId(1);
    let s2 = SessionId(2);
    let recipients: BTreeMap<CellKey, Vec<SessionId>> = [
        (CellKey(0, 0, 0), vec![s1]),
        (CellKey(1, 1, 0), vec![s1]),
        (CellKey(5, 0, 0), vec![s2]),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        plan.recipients, recipients,
        "the cube two steps out is not taken"
    );
    assert!(plan.removals.is_empty(), "nothing was delivered before");
    assert_eq!(
        held.delivered[&s1],
        [EntityId(11), EntityId(12)].into_iter().collect()
    );
    assert_eq!(held.delivered[&s2], [EntityId(21)].into_iter().collect());
}

#[test]
fn a_held_cube_is_kept_one_step_further_out_and_dropped_past_that_with_one_notice() {
    let mut held = InterestHeld::default();
    let a = SessionId(1);
    // Tick 1: the row is one step away — taken.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(1, 0, 0))],
        &mut held,
    );
    assert_eq!(plan.recipients.len(), 1);
    // Tick 2: two steps away — outside the take, inside the drop: still held, nothing said.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(2, 0, 0))],
        &mut held,
    );
    assert_eq!(plan.recipients.get(&CellKey(2, 0, 0)), Some(&vec![a]));
    assert!(plan.removals.is_empty());
    // Tick 3: three steps — dropped, and the observer is told exactly once.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(3, 0, 0))],
        &mut held,
    );
    assert!(plan.recipients.is_empty());
    assert_eq!(plan.removals, vec![(a, EntityId(9))]);
    // Tick 4: still three steps — silence, not a second notice.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(3, 0, 0))],
        &mut held,
    );
    assert!(plan.removals.is_empty());
    // Tick 5: the row returns — shown again, and no notice is owed for a return.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(0, 0, 0))],
        &mut held,
    );
    assert_eq!(plan.recipients.get(&CellKey(0, 0, 0)), Some(&vec![a]));
    assert!(plan.removals.is_empty());
    // A stranger standing two steps out beside nobody shown before is NOT taken: the keep is for
    // a figure the observer was shown, never a free extra step for everybody.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(9, CellKey(0, 0, 0)), row(10, CellKey(2, 0, 0))],
        &mut held,
    );
    assert_eq!(plan.recipients.get(&CellKey(2, 0, 0)), None);
    assert_eq!(held.delivered[&a], [EntityId(9)].into_iter().collect());
}

#[test]
fn an_observer_that_left_takes_its_hold_with_it_and_a_fast_row_widens_every_take() {
    let mut held = InterestHeld::default();
    let _ = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0)), obs(2, CellKey(9, 9, 9))],
        &[row(1, CellKey(0, 0, 0))],
        &mut held,
    );
    assert_eq!(held.delivered.len(), 2);
    // Observer 2 is gone; its hold goes with it, and no notice is owed to nobody.
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(1, CellKey(0, 0, 0))],
        &mut held,
    );
    assert_eq!(held.delivered.len(), 1);
    assert!(plan.removals.is_empty());
    // A row closing at 2 km/s is taken from four cubes further out than a standing one.
    let fast = PlacedRow {
        entity: EntityId(7u128),
        cell: CellKey(4, 0, 0),
        speed_mps: 2000.0,
    };
    let plan = plan_interest(
        RULE,
        &[obs(1, CellKey(0, 0, 0))],
        &[row(1, CellKey(0, 0, 0)), fast],
        &mut held,
    );
    assert_eq!(
        plan.recipients.get(&CellKey(4, 0, 0)),
        Some(&vec![SessionId(1)]),
        "take = 1 + ceil(2000 * 0.17 / 100) = 5 steps"
    );
}

// ---- the emission through the real schedule ------------------------------------------------

/// What one tick sent: `(to, class, bytes)` per frame.
type Sent = Vec<(NodeId, MsgClass, Vec<u8>)>;

const SESSION_B: SessionId = SessionId(2);

fn at(m: DVec3) -> StampedPose {
    StampedPose::at_rest(config().frame, m, UniverseTick(100))
}

/// A rig whose shard has a region of its own (root ⊃ own), so it speaks in its own frame and every
/// occupant's row is PLACED in it — the production shape; a bare rig has no frame and every row is
/// foreign-labelled.
fn planted_rig() -> Rig {
    let mut rig = Rig::new();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region()]);
    rig
}

/// Log two occupants in at the given positions (the full attach flow each), then run one quiet
/// tick and return what left toward the gateway.
fn two_occupants(a: DVec3, b: DVec3) -> (Rig, Vec<(NodeId, MsgClass, Vec<u8>)>) {
    let mut rig = planted_rig();
    rig.grant_realm();
    let _ = rig.attach_request_with(SESSION, GATEWAY, Some(at(a)));
    let _ = rig.confirm_entity_grant(SESSION);
    let _ = rig.attach_request_with(SESSION_B, GATEWAY, Some(at(b)));
    let _ = rig.confirm_entity_grant(SESSION_B);
    let sent = rig.tick(vec![]);
    (rig, sent)
}

/// The interest counters, read as a snapshot so a test can diff ONE tick (the attach ticks emit
/// bodies of their own).
fn counters(rig: &Rig) -> (u64, u64, u64) {
    let s = rig.world.resource::<StubStats>();
    (
        s.interest_bodies,
        s.interest_rows_shipped,
        s.interest_removals,
    )
}

/// One quiet tick and the counters it added.
fn quiet_tick(rig: &mut Rig) -> (Sent, (u64, u64, u64)) {
    let before = counters(rig);
    let sent = rig.tick(vec![]);
    let after = counters(rig);
    (
        sent,
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
    )
}

/// Every snapshot-class envelope that left toward the gateway this tick.
fn bodies(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<ShardToGateway> {
    sent.iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Snapshot))
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("envelope"))
        .collect()
}

/// The rows delivered to each session this tick: a `Frame` reaches every session the shard holds
/// (the gateway fans it to every subscriber); a `FrameFor` reaches the sessions it names.
fn rows_per_session(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
    all: &[SessionId],
) -> BTreeMap<SessionId, Vec<EntityId>> {
    let mut out: BTreeMap<SessionId, Vec<EntityId>> = BTreeMap::new();
    for body in bodies(sent) {
        let (recipients, bytes) = match body {
            ShardToGateway::Frame { snapshot_bytes, .. } => (all.to_vec(), snapshot_bytes),
            ShardToGateway::FrameFor {
                recipients,
                snapshot_bytes,
                ..
            } => (recipients, snapshot_bytes),
            other => panic!("a snapshot-class envelope is a body, got {other:?}"),
        };
        let snap: SnapshotDatagram = postcard::from_bytes(&bytes).expect("snapshot");
        for s in recipients {
            out.entry(s)
                .or_default()
                .extend(snap.entities.iter().map(|e| e.entity));
        }
    }
    out
}

fn notices(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<(SessionId, EntityId)> {
    gw_replies(sent)
        .into_iter()
        .filter_map(|m| match m {
            ShardToGateway::EntityOutOfInterest {
                session, entity, ..
            } => Some((session, entity)),
            _ => None,
        })
        .collect()
}

/// Entity ids are minted, not ordered by session: compare row sets sorted.
fn sorted(mut v: Vec<EntityId>) -> Vec<EntityId> {
    v.sort();
    v
}

fn entity_of(rig: &Rig, session: SessionId) -> EntityId {
    rig.world.resource::<Dots>().0[&session].entity
}

#[test]
fn two_neighbours_share_one_whole_realm_body_and_two_strangers_each_get_their_own() {
    // Ten metres apart: one cube or two adjacent ones, every observer holds them all, so the body
    // rides the whole-realm arm — exactly what a lone pilot's shard sent before this slice.
    let (rig, sent) = two_occupants(DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0));
    let (ea, eb) = (entity_of(&rig, SESSION), entity_of(&rig, SESSION_B));
    let per = rows_per_session(&sent, &[SESSION, SESSION_B]);
    assert_eq!(sorted(per[&SESSION].clone()), sorted(vec![ea, eb]));
    assert_eq!(sorted(per[&SESSION_B].clone()), sorted(vec![ea, eb]));
    assert!(
        bodies(&sent)
            .iter()
            .all(|b| matches!(b, ShardToGateway::Frame { .. })),
        "a body every observer holds is the whole-realm arm"
    );
    assert!(notices(&sent).is_empty());
    let mut rig = rig;
    let (_, (bodies_n, rows_n, _)) = quiet_tick(&mut rig);
    assert_eq!(
        (bodies_n, rows_n),
        (1, 4),
        "one body, 2 rows × 2 recipients"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().interest_rule_reset,
        1,
        "the first tick sets the cube side"
    );

    // Ten kilometres apart: eleven cubes between them, so each observer receives its own row only,
    // in a body that names it and nobody else.
    let (rig, sent) = two_occupants(DVec3::ZERO, DVec3::new(10_000.0, 0.0, 0.0));
    let (ea, eb) = (entity_of(&rig, SESSION), entity_of(&rig, SESSION_B));
    let per = rows_per_session(&sent, &[SESSION, SESSION_B]);
    assert_eq!(per[&SESSION], vec![ea]);
    assert_eq!(per[&SESSION_B], vec![eb]);
    let named: Vec<Vec<SessionId>> = bodies(&sent)
        .into_iter()
        .filter_map(|b| match b {
            ShardToGateway::FrameFor { recipients, .. } => Some(recipients),
            _ => None,
        })
        .collect();
    assert_eq!(named, vec![vec![SESSION], vec![SESSION_B]]);
    let mut rig = rig;
    let (_, (bodies_n, rows_n, _)) = quiet_tick(&mut rig);
    assert_eq!(
        (bodies_n, rows_n),
        (2, 2),
        "two bodies of one row, one recipient each"
    );
}

#[test]
fn an_occupant_that_walks_out_of_reach_is_told_to_the_observer_once_and_returns_silently() {
    let (mut rig, sent) = two_occupants(DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0));
    let (ea, eb) = (entity_of(&rig, SESSION), entity_of(&rig, SESSION_B));
    assert_eq!(
        sorted(rows_per_session(&sent, &[SESSION, SESSION_B])[&SESSION].clone()),
        sorted(vec![ea, eb])
    );
    // B is now ten kilometres away: A loses B and B loses A, one notice each.
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION_B)
        .expect("B")
        .pose = at(DVec3::new(10_000.0, 0.0, 0.0));
    let sent = rig.tick(vec![]);
    let mut told = notices(&sent);
    told.sort();
    let mut expected = vec![(SESSION, eb), (SESSION_B, ea)];
    expected.sort();
    assert_eq!(told, expected);
    assert_eq!(
        rows_per_session(&sent, &[SESSION, SESSION_B])[&SESSION],
        vec![ea]
    );
    // Still apart: silence, never a second notice.
    let sent = rig.tick(vec![]);
    assert!(notices(&sent).is_empty());
    // B walks back: the rows return and no notice is owed.
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION_B)
        .expect("B")
        .pose = at(DVec3::new(10.0, 0.0, 0.0));
    let sent = rig.tick(vec![]);
    assert!(notices(&sent).is_empty());
    assert_eq!(
        sorted(rows_per_session(&sent, &[SESSION, SESSION_B])[&SESSION].clone()),
        sorted(vec![ea, eb])
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.interest_removals, 2);
    // The notice rides the reliable control lane, retained (a lost one is a frozen figure). The
    // notice is a session-lane envelope, not an `InterShardFlow`, so it is read raw here.
    let sent: Vec<(NodeId, MsgClass, Vec<u8>, Durability)> = {
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION_B)
            .expect("B")
            .pose = at(DVec3::new(10_000.0, 0.0, 0.0));
        rig.world.resource_mut::<InboundBox>().0 = vec![];
        rig.schedule.run(&mut rig.world);
        std::mem::take(&mut rig.world.resource_mut::<OutboundBox>().0)
            .into_iter()
            .map(|(to, class, bytes, dur)| (to, class, bytes.to_vec(), dur))
            .collect()
    };
    let retained: Vec<Durability> = sent
        .iter()
        .filter(|(_, class, bytes, _)| {
            (*class == MsgClass::Control)
                & matches!(
                    postcard::from_bytes::<ShardToGateway>(bytes),
                    Ok(ShardToGateway::EntityOutOfInterest { .. })
                )
        })
        .map(|(_, _, _, dur)| *dur)
        .collect();
    assert_eq!(retained, vec![Durability::Retained, Durability::Retained]);
}

#[test]
fn a_row_that_cannot_be_placed_in_this_frame_ships_to_everybody_counted() {
    // A row wearing a frame this shard cannot restate into its own (the fed-ghost degrade) is
    // unplaced: it rides the whole-realm arm to every observer rather than vanishing into no cube.
    let (mut rig, _) = two_occupants(DVec3::ZERO, DVec3::new(10_000.0, 0.0, 0.0));
    let (ea, eb) = (entity_of(&rig, SESSION), entity_of(&rig, SESSION_B));
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION_B)
        .expect("B")
        .pose
        .frame = foreign_frame();
    let before = rig.world.resource::<StubStats>().interest_rows_unplaced;
    let sent = rig.tick(vec![]);
    let per = rows_per_session(&sent, &[SESSION, SESSION_B]);
    assert_eq!(
        sorted(per[&SESSION].clone()),
        sorted(vec![ea, eb]),
        "the unplaced row reaches A too"
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.interest_rows_unplaced - before, 1);
    assert!(stats.entity_rows_foreign_labelled >= 1);
}

#[test]
fn a_bigger_figure_re_keys_the_cubes_and_the_hold_starts_over() {
    let (mut rig, _) = two_occupants(DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0));
    assert_eq!(rig.world.resource::<StubStats>().interest_rule_reset, 1);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION_B)
        .expect("B")
        .look_extent_m = 2.0;
    let _ = rig.tick(vec![]);
    let held = rig.world.resource::<InterestHeld>();
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.interest_rule_reset, 2);
    assert!((held.side_m - interest_rule(2.0, 0.05).side_m).abs() < 1e-9);
    // And the hold was rebuilt for this tick: both observers were shown both figures.
    assert_eq!(held.delivered.len(), 2);
}

#[test]
fn each_observers_rows_grow_with_its_neighbours_not_with_the_realms_population() {
    // THE OWED GATE (D-9): four groups of thirty-two, twenty kilometres apart (23 cubes). Every observer
    // receives its own group — thirty-two rows — while the realm holds a hundred and twenty-eight.
    // Whole-realm emission would ship 128 × 128 rows a tick; this ships 128 × 32.
    let mut rig = planted_rig();
    rig.grant_realm();
    let mut sessions = Vec::new();
    for group in 0..4u64 {
        for i in 0..32u64 {
            let session = SessionId(u128::from(group * 32 + i + 10));
            let pos = DVec3::new(group as f64 * 20_000.0 + i as f64 * 0.5, 0.0, 0.0);
            let _ = rig.attach_request_with(session, GATEWAY, Some(at(pos)));
            let _ = rig.confirm_entity_grant(session);
            sessions.push(session);
        }
    }
    let (sent, (bodies_n, rows_n, _)) = quiet_tick(&mut rig);
    let per = rows_per_session(&sent, &sessions);
    assert_eq!(per.len(), 128, "every observer was served");
    for (session, rows) in &per {
        assert_eq!(rows.len(), 32, "{session:?} receives its own group only");
    }
    assert_eq!(rows_n, 128 * 32);
    // One body per group per MTU chunk: thirty-two rows exceed one datagram, so each group ships
    // in the same number of chunks, and every chunk names the same thirty-two recipients.
    let mut recipient_sets: Vec<Vec<SessionId>> = bodies(&sent)
        .into_iter()
        .filter_map(|b| match b {
            ShardToGateway::FrameFor { recipients, .. } => Some(recipients),
            _ => None,
        })
        .collect();
    recipient_sets.sort();
    recipient_sets.dedup();
    assert_eq!(recipient_sets.len(), 4, "four groups, four recipient sets");
    assert!(recipient_sets.iter().all(|r| r.len() == 32));
    assert_eq!(bodies_n % 4, 0, "the same chunk count per group");
    assert!(rows_n < 128 * 128);
    // And the dense case — everyone in one place — keeps everyone seeing everyone.
    let mut dense = planted_rig();
    dense.grant_realm();
    let mut all = Vec::new();
    for i in 0..128u64 {
        let session = SessionId(u128::from(i + 10));
        let _ = dense.attach_request_with(
            session,
            GATEWAY,
            Some(at(DVec3::new(i as f64 * 0.5, 0.0, 0.0))),
        );
        let _ = dense.confirm_entity_grant(session);
        all.push(session);
    }
    let sent = dense.tick(vec![]);
    let per = rows_per_session(&sent, &all);
    assert!(per.values().all(|rows| rows.len() == 128));
}
