//! The placement-carry derivation (how far a shard may extrapolate a moving placement) and the pose-stamp invariant, plus the RLM 5f RG-1 reactive greeting: a shard announces itself to every booked peer after a silence interval, suppressed per-peer by inbound contact, identical across shard kinds. Locals that travel: presence, presence_ticks, benign_contact. ANCESTOR (177) sits in this block but is used at 13379+ and must NOT move.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn the_placement_carry_cap_is_derived_from_the_tick_rate_and_never_zero() {
    // A NAMED derivation, not a literal: the cap is the configured wall-clock budget expressed in
    // whatever ticks this cluster runs at. It rounds UP and floors at one, because a cap of zero would
    // refuse every moving placement that arrived even a single tick late — i.e. all of them.
    //
    // It used to live on the GATEWAY's `TransportTuning`, which is a router's transport parameter; how
    // far a simulation may extrapolate along a velocity it authored is not the router's to decide, and
    // the router no longer applies a placement at all.
    let ms = PlacementCarry::BUDGET_MS;
    assert_eq!(
        PlacementCarry::skew_ticks_for(50),
        50 * ms / 1_000,
        "at 50 Hz the cap is the budget in ticks"
    );
    assert_eq!(PlacementCarry::skew_ticks_for(20), 20 * ms / 1_000);
    // A tick rate so slow that the budget is under one tick still yields one, not zero.
    assert_eq!(PlacementCarry::skew_ticks_for(1), 1);
    // A zero tick rate is a misconfiguration, not a division by zero.
    assert_eq!(PlacementCarry::skew_ticks_for(0), 1);
    // The struct constructor and the scalar are the SAME derivation, so a caller cannot get two answers.
    assert_eq!(
        PlacementCarry::for_tick_rate(50),
        PlacementCarry {
            max_skew_ticks: PlacementCarry::skew_ticks_for(50)
        }
    );
}

/// The detector's stamp invariant, all three arms by name (HR5): a LATCHED dot is exempt even
/// with a frozen stamp; a non-latched dot holding the clock's stamp passes.
#[test]
fn the_stamp_invariant_exempts_a_latched_dot_and_accepts_a_current_stamp() {
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let mut latched = BTreeMap::new();
    latched.insert(entity, TransferId(1));
    debug_assert_nonlatched_stamp_is_current(&latched, entity, UniverseTick(1), UniverseTick(9));
    debug_assert_nonlatched_stamp_is_current(
        &BTreeMap::new(),
        entity,
        UniverseTick(9),
        UniverseTick(9),
    );
}

/// …and the panic arm: a non-latched dot whose stamp trails the clock is the `readvance_dots`
/// ordering guarantee broken — the invariant fails loudly rather than scanning a stale world.
#[test]
#[should_panic(expected = "a non-latched simulating dot's stamp must equal the clock")]
fn the_stamp_invariant_panics_on_a_non_latched_stale_stamp() {
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    debug_assert_nonlatched_stamp_is_current(
        &BTreeMap::new(),
        entity,
        UniverseTick(1),
        UniverseTick(9),
    );
}

#[test]
fn presence_absent_sends_no_greeting() {
    // No PresenceAnnounce ⇒ announce_presence no-ops ⇒ byte-identical (a static shard).
    let mut rig = Rig::new();
    assert!(presence_ticks(&rig.tick_raw(vec![])).is_empty());
}

#[test]
fn greets_every_booked_peer_while_silent() {
    let mut rig = Rig::new();
    rig.world
        .insert_resource(presence(&[GATEWAY, ANCESTOR, ORCH], 50));
    let ticks = presence_ticks(&rig.tick_raw(vec![]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([GATEWAY, ANCESTOR, ORCH]),
    );
    // The greeting carries the shard's local tick (1 in a fresh rig).
    assert_eq!(ticks[&GATEWAY], TickId(1));
}

#[test]
fn greeting_rides_the_reliable_saga_class() {
    // Learning fires ONLY on a reliable frame; a datagram would teach nothing.
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    let sent = rig.tick_raw(vec![]);
    let greeting = sent
        .iter()
        .find(|(_, _, flow, _)| matches!(flow, InterShardFlow::ShardPresence(_)))
        .expect("a greeting");
    assert_eq!(greeting.1, MsgClass::Saga);
}

#[test]
fn stays_silent_within_the_interval_after_greeting() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    rig.tick_raw(vec![]); // tick 1: greet ⇒ last_contact[GATEWAY] = 1
    rig.set_local_tick(40); // 40 − 1 = 39 < 50
    assert!(presence_ticks(&rig.tick_raw(vec![])).is_empty());
}

#[test]
fn re_greets_after_an_interval_of_silence() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    rig.tick_raw(vec![]); // tick 1: greet
    rig.set_local_tick(51); // 51 − 1 = 50 ≥ 50
    let ticks = presence_ticks(&rig.tick_raw(vec![]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([GATEWAY]),
    );
    assert_eq!(ticks[&GATEWAY], TickId(51));
}

#[test]
fn inbound_from_a_peer_suppresses_only_that_peers_greeting() {
    let mut rig = Rig::new();
    rig.world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    // A frame from GATEWAY this tick = contact ⇒ GATEWAY silent; ANCESTOR (unheard) still greeted.
    let ticks = presence_ticks(&rig.tick_raw(vec![benign_contact(GATEWAY)]));
    assert_eq!(
        ticks.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([ANCESTOR]),
    );
}

#[test]
fn a_non_wire_notice_is_not_contact() {
    let mut rig = Rig::new();
    rig.world.insert_resource(presence(&[GATEWAY], 50));
    // A NodeUnreachable notice is not an inbound frame ⇒ does NOT reset silence ⇒ GATEWAY greeted.
    let notice = Inbound::NodeUnreachable {
        to: GATEWAY,
        class: MsgClass::Saga,
        undelivered: MsgId(0),
    };
    assert!(presence_ticks(&rig.tick_raw(vec![notice])).contains_key(&GATEWAY));
}

#[test]
fn presence_new_rejects_a_zero_interval() {
    assert!(PresenceAnnounce::new(BTreeSet::from([GATEWAY]), 0).is_err());
    assert_eq!(
        PresenceAnnounce::new(BTreeSet::from([GATEWAY]), 1)
            .expect("ok")
            .interval_ticks,
        1,
    );
}

#[test]
fn greeting_is_g_identical_across_shard_kinds() {
    // HR4: the SAME greeting fixture on a System-realm shard and a Planet-realm shard emits the
    // IDENTICAL ShardPresence set — the greeting is shard-kind-blind (reachability, not gameplay).
    let mut system = Rig::with_config(config());
    let mut planet = {
        let mut cfg = config();
        cfg.realm = RealmId::Planet(7);
        cfg.held_realms = StubConfig::single_realm(RealmId::Planet(7));
        cfg.own_coord = StubConfig::root_coord(RealmId::Planet(7));
        Rig::with_config(cfg)
    };
    system
        .world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    planet
        .world
        .insert_resource(presence(&[GATEWAY, ANCESTOR], 50));
    assert_eq!(
        presence_ticks(&system.tick_raw(vec![])),
        presence_ticks(&planet.tick_raw(vec![])),
    );
}
