//! The Q2 window-relay lane and THE WINDOW LANE Slice A (banner 15096): sealed batches held verbatim with fail-closed admission, forwarded send-on-change and TTL-pruned, byte-identical across two hops, egress and union over-draw measured, the structural proof that production never opens a seal; the ship half (a shard sealing its own self-authored statements up to its resolved parent on resolve/change/cadence) and the read-only accessors; what a realm states about itself on a window (SL3 self-look) and the membership verdict's anti-flicker grace hold; then one frame per tick per open window, the pre-inverted hop row, self-look and point-of-light marker bodies, the send-on-change beat, the derived TTL, the Child-scope guards, the Q1 one-level fence and the membership verdict fold. Locals that travel: window_relays (13708), window_bodies (15127, used at 14082-14131 which is inside this module), WINDOW_CHILD_CENTER, window_rig, drive_window_emit.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

/// Slice C1, the Q2-relay receive (the tombstoned shape lane's successor — owner-approved
/// 2026-08-16, owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): a live
/// child's sealed batch is HELD VERBATIM (byte-equal, unopened — the parent's whole lawful
/// vocabulary is forward-or-drop), a receipt is a last-wins replace, and every admission arm
/// refuses fail-closed + counted: mis-route, unattested sender, a deposed incarnation's stale
/// fence.
#[test]
fn on_window_relay_holds_the_sealed_batch_verbatim_and_admits_fail_closed() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let mut held = RelayHeld::default();
    let mut stats = StubStats::default();
    let seal = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 1.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    let child_coord = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        level_of(story.system).expect("system level"),
        level_of(story.planet).expect("planet level"),
    ]))
    .expect("two-level path");
    let attested = ChildRealmNodes(BTreeMap::from([(story.planet, NodeId(70))]));
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relays_received, 1);
    let entry = &held.0[&story.planet];
    assert_eq!(entry.own, seal, "held VERBATIM — byte-equal, unopened");
    assert_eq!(
        entry.digest,
        relay_entry_digest(Fence(3), &seal, &[]),
        "the §5.4 baseline digest is computed once, at receive"
    );
    assert_eq!(entry.fence, Fence(3), "the child's own fence rides intact");
    // LAST-WINS REPLACE: a same-or-newer fence's batch replaces the held one whole.
    let newer = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Level {
            at: UniverseTick(10),
            rows: Vec::new(),
        },
    ]);
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(4),
            own: newer.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(held.0[&story.planet].own, newer, "replaced, not merged");
    // STALE FENCE: a deposed incarnation (fence below the held one) is refused counted.
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord.clone(),
            realm_fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_stale, 1);
    assert_eq!(
        held.0[&story.planet].fence,
        Fence(4),
        "the zombie replaced nothing"
    );
    // UNATTESTED: a sender the head does not name is refused fail-closed, counted.
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: child_coord,
            realm_fence: Fence(9),
            own: seal.clone(),
            interior: Vec::new(),
        },
        NodeId(71),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_unattested, 1);
    assert_eq!(
        held.0[&story.planet].fence,
        Fence(4),
        "nothing new believed"
    );
    // MIS-ROUTE: a sender whose parent link is not this realm drops counted, stores nothing.
    let foreign = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 999),
        RealmLevel::new(RealmKindTag::Planet, 1),
    ]))
    .expect("two-level path");
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: foreign,
            realm_fence: Fence(1),
            own: seal,
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg,
        &clock,
        &attested,
        &mut held,
        &mut stats,
    );
    assert_eq!(stats.window_relay_misrouted, 1);
    assert_eq!(held.0.len(), 1);
}

/// Slice C1, the Q2-relay forward: a held batch reaches every open window ONCE (send-on-change
/// per (window, child) — a fresh window is served everything currently held), the forwarded
/// bytes are the held bytes VERBATIM with the child's fence intact, a held entry outliving the
/// derived retain TTL is pruned (and its per-window baseline cleared so a re-held child
/// re-ships), and zero windows forward nothing while the prune still runs.
#[test]
fn emit_window_relays_forwards_verbatim_send_on_change_and_prunes_by_the_ttl() {
    let cfg = config();
    let clock = ClockSample {
        local_tick: vd_core::TickId(10),
        universe_tick: UniverseTick(10),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let mut held = RelayHeld::default();
    let mut windows = OpenWindows::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let seal = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: OTHER_REALM,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 2.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: clock.local_tick,
            fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(3), &seal, &[]),
        },
    );
    // ZERO WINDOWS: nothing forwards, nothing panics, the holder survives (still fresh).
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.window_relays_forwarded, 0);
    assert_eq!(held.0.len(), 1);
    // ONE WINDOW: the held batch forwards once, VERBATIM, then send-on-change holds its tongue.
    windows.0.insert(
        (GATEWAY, WindowId(1)),
        OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
    );
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.window_relays_forwarded, 1);
    // ONE constructed-expected equality (HR5: no decode-match with an unreachable
    // wildcard): the forwarded bytes ARE the envelope below — the forwarder's OWN fence,
    // the child's fence INTACT, the sealed statements byte-for-byte, unopened.
    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
        realm_fence: Fence(1),
        window: WindowId(1),
        child: OTHER_REALM,
        child_fence: Fence(3),
        statements: seal.clone(),
        interior: Vec::new(),
    })
    .expect("encodes");
    assert_eq!(outbox.0.len(), 1);
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(
        &bytes[..],
        &expected[..],
        "forwarded verbatim: own envelope fence, intact child fence, unopened seal"
    );
    emit_window_relays(
        &cfg,
        &clock,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 1,
        "unchanged — send-on-change holds its tongue"
    );
    // TTL: an entry past the derived retain TTL is pruned; the baseline clears with it, so a
    // re-held child re-ships to the same window.
    let later = ClockSample {
        local_tick: vd_core::TickId(10 + retain_ttl_ticks(&cfg) + 1),
        ..clock
    };
    emit_window_relays(
        &cfg,
        &later,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert!(held.0.is_empty(), "pruned by the derived TTL");
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: later.local_tick,
            fence: Fence(3),
            own: seal.clone(),
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(3), &seal, &[]),
        },
    );
    emit_window_relays(
        &cfg,
        &later,
        Fence(1),
        &mut held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 2,
        "a re-held child re-ships — the vanished entry cleared its baseline"
    );
}

/// G-VERBATIM (look_horizon.md §6 slice 3; owner-approved 2026-08-17 — RULINGS + §2 ASK A):
/// the bytes a GRANDPARENT forwards are BYTE-IDENTICAL to what the author sealed, across
/// both hops — a measurement that could fail. The full path, on the story world's real
/// lineage: the planet seals its own batch → the SYSTEM holds it and selects it into its
/// up-relay's interior (membership-gated, §3.4.5 — the out-of-verdict arm is driven too) →
/// the GALAXY holds the pair and forwards to a window subscriber — where the encoded
/// `interior[0].own` must be the planet's sealed bytes exactly, its fence intact. Plus the
/// §5.4 third digest component as a measurement: a grandchild-only change (same system own
/// batch) re-ships past send-on-change.
#[test]
fn g_verbatim_a_grandchilds_sealed_bytes_survive_both_hops_byte_identical() {
    let story = Story::new();
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    // THE AUTHOR: the planet's own sealed batch (its own look about ITSELF — SL3).
    let sealed_p = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 1.0 }),
            },
            authored_at: UniverseTick(9),
        },
    ]);
    // HOP 1 — the SYSTEM holds the planet's batch sealed.
    let cfg_s = story_config(&story, story.system);
    let p_coord = cfg_s
        .own_coord
        .child(level_of(story.planet).expect("planet level"));
    let mut s_held = RelayHeld::default();
    let mut stats = StubStats::default();
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: p_coord,
            realm_fence: Fence(3),
            own: sealed_p.clone(),
            interior: Vec::new(),
        },
        NodeId(70),
        &cfg_s,
        &clock,
        &ChildRealmNodes(BTreeMap::from([(story.planet, NodeId(70))])),
        &mut s_held,
        &mut stats,
    );
    // The §3.4.5 forward gate, BOTH arms: out of the verdict ⇒ the batch stays home;
    // in the verdict ⇒ it joins the interior, fence + bytes VERBATIM.
    assert!(
        build_relay_interior(&s_held, &BTreeSet::new()).is_empty(),
        "a child outside the realm's own in-band verdict is not forwarded (§3.4.5)"
    );
    let interior = build_relay_interior(&s_held, &BTreeSet::from([story.planet]));
    assert_eq!(interior.len(), 1);
    assert_eq!(interior[0].child, story.planet);
    assert_eq!(
        interior[0].child_fence,
        Fence(3),
        "the author's OWN fence, intact"
    );
    assert_eq!(interior[0].own, sealed_p, "hop 1: byte-identical");
    // HOP 2 — the GALAXY holds the system's (own + interior) pair sealed...
    let sealed_s = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Level {
            at: UniverseTick(9),
            rows: Vec::new(),
        },
    ]);
    let cfg_g = story_config(&story, story.galaxy);
    let s_coord = cfg_g
        .own_coord
        .child(level_of(story.system).expect("system level"));
    let mut g_held = RelayHeld::default();
    let g_attested = ChildRealmNodes(BTreeMap::from([(story.system, NodeId(71))]));
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: s_coord.clone(),
            realm_fence: Fence(5),
            own: sealed_s.clone(),
            interior: interior.clone(),
        },
        NodeId(71),
        &cfg_g,
        &clock,
        &g_attested,
        &mut g_held,
        &mut stats,
    );
    // ...and forwards it to a window subscriber. ONE constructed-expected byte equality:
    // the wire bytes ARE the planet's sealed bytes riding inside, unopened at every hop.
    let mut windows = OpenWindows::default();
    windows.0.insert(
        (GATEWAY, WindowId(1)),
        OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
    );
    let mut outbox = OutboundBox::default();
    emit_window_relays(
        &cfg_g,
        &clock,
        Fence(6),
        &mut g_held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
        realm_fence: Fence(6),
        window: WindowId(1),
        child: story.system,
        child_fence: Fence(5),
        statements: sealed_s.clone(),
        interior: interior.clone(),
    })
    .expect("encodes");
    assert_eq!(outbox.0.len(), 1);
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(
        &bytes[..],
        &expected[..],
        "G-VERBATIM: what the grandparent forwards is what the author sealed, byte for byte"
    );
    // THE §5.4 THIRD COMPONENT, measured: the planet's picture changes while the system's
    // OWN statements stay identical — send-on-change must re-ship (the re-keyed digest), or
    // a grandchild's change would be invisible on the load-bearing draw path.
    let sealed_p2 = vd_wire::session_flow::seal_relay_statements(&[
        vd_wire::session_flow::RelayedStatement::Body {
            subject: story.planet,
            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                bag: vd_core::look::look_bag(&Boundary::Shell { r: 2.0 }),
            },
            authored_at: UniverseTick(10),
        },
    ]);
    on_window_relay(
        vd_wire::intershard::WindowRelay {
            child: s_coord,
            realm_fence: Fence(5),
            own: sealed_s.clone(), // the system's own half UNCHANGED
            interior: vec![vd_wire::intershard::InteriorRelay {
                child: story.planet,
                child_fence: Fence(3),
                own: sealed_p2.clone(),
            }],
        },
        NodeId(71),
        &cfg_g,
        &clock,
        &g_attested,
        &mut g_held,
        &mut stats,
    );
    emit_window_relays(
        &cfg_g,
        &clock,
        Fence(6),
        &mut g_held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.window_relays_forwarded, 2,
        "a grandchild-only change re-keys the digest and re-ships (§5.4)"
    );
    // The read-only gate views hand back BYTES, never values (the same seal discipline):
    // the held own half and the held interior half, verbatim.
    assert_eq!(g_held.statements_for(story.system), Some(sealed_s));
    assert_eq!(
        g_held.interior_for(story.system),
        Some(vec![vd_wire::intershard::InteriorRelay {
            child: story.planet,
            child_fence: Fence(3),
            own: sealed_p2,
        }])
    );
}

/// look_horizon.md §6 SLICE 6 — RELAY EGRESS, measured against §5.2's formula
/// `W × C × blob × rate`, WITH the gateway multiplier MEASURED rather than assumed: the same
/// held children re-ship the same number of times under one open window and under two, and
/// the two-window egress must be EXACTLY double; with a second held child the egress must be
/// exactly the per-child blob sum times W times the re-ship count. The blob is the
/// DEPARTURE-FIXTURE class, built through the REAL codecs at THE world's own numbers (the
/// home system's sealed batch: its 5-row level + its own look + 5 markers, plus the 5
/// slice-3 interior batches), and its size is PINNED — §5.4's law that this design does not
/// change rows per fold has its bytes-per-relay counterpart pinned here.
#[test]
fn slice6_relay_egress_measured_equals_w_times_c_times_blob_times_rate() {
    let cfg_w = vd_physics::worldgen::UniverseConfig::world(15.0, 0.05);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg_w);
    let galaxy = vd_core::worldgen::GALAXY;
    let systems: Vec<RealmId> = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(galaxy))
        .map(|r| r.realm)
        .collect();
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("home");
    let sibling = *systems
        .iter()
        .find(|s| **s != home)
        .expect("a ring sibling");
    let luma: BTreeMap<RealmId, (u8, f64)> =
        vd_physics::worldgen::system_photometrics_for_config(0, &cfg_w)
            .into_iter()
            .map(|(r, p)| (r, vd_physics::worldgen::marker_datum(&p)))
            .collect();
    let shape_of = |realm: RealmId| {
        world
            .regions()
            .iter()
            .find(|r| r.realm == realm)
            .expect("rostered")
            .shape
    };
    // ONE system's departure-fixture batch at tick `t`, through the real codecs: the level
    // rows carry the planets' true closed-form positions, so consecutive ticks genuinely
    // change the bytes (the re-ship trigger is the fingerprint over row VALUES).
    let batch_at =
        |system: RealmId, t: u64| -> (Vec<u8>, Vec<vd_wire::intershard::InteriorRelay>) {
            let at = UniverseTick(t);
            let movers = vd_physics::worldgen::moving_children_for_config(0, &cfg_w, system);
            assert_eq!(
                movers.len(),
                9,
                "THE world's systems orbit the derived 9 planets"
            );
            let sys_frame = vd_core::pose::frame_for_realm(system, Some(galaxy)).expect("frame");
            let rows: Vec<vd_wire::channels::RealmSnap> = movers
                .iter()
                .map(|(p, e)| {
                    let st = vd_physics::motion::Motion::Kepler(*e)
                        .state_at(t as f64 * cfg_w.interest.tick_dt_s);
                    vd_wire::channels::RealmSnap {
                        realm: *p,
                        frame: vd_core::pose::frame_for_realm(*p, Some(system)).expect("frame"),
                        pose: vd_core::pose::StampedPose {
                            frame: sys_frame,
                            pos: st.anchor(),
                            vel: st.velocity,
                            orient: vd_core::glam::DQuat::IDENTITY,
                            universe_tick: at,
                        },
                    }
                })
                .collect();
            let mut stmts = vec![
                vd_wire::session_flow::RelayedStatement::Level { at, rows },
                vd_wire::session_flow::RelayedStatement::Body {
                    subject: system,
                    stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                        bag: vd_core::look::look_bag(&shape_of(system)),
                    },
                    authored_at: at,
                },
            ];
            for (p, _) in &movers {
                stmts.push(vd_wire::session_flow::RelayedStatement::Body {
                    subject: *p,
                    stmt: vd_wire::session_flow::BodyStmt::Marker {
                        luma: vd_core::look::marker_bag(
                            luma.get(p).copied(),
                            shape_of(*p).circumscribed_extent(),
                        ),
                    },
                    authored_at: at,
                });
            }
            let own = vd_wire::session_flow::seal_relay_statements(&stmts);
            let interior = movers
                .iter()
                .map(|(p, _)| vd_wire::intershard::InteriorRelay {
                    child: *p,
                    child_fence: Fence(3),
                    own: vd_wire::session_flow::seal_relay_statements(&[
                        vd_wire::session_flow::RelayedStatement::Level {
                            at,
                            rows: Vec::new(),
                        },
                        vd_wire::session_flow::RelayedStatement::Body {
                            subject: *p,
                            stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                                bag: vd_core::look::look_bag(&shape_of(*p)),
                            },
                            authored_at: at,
                        },
                    ]),
                })
                .collect();
            (own, interior)
        };
    // One measured run: `w` open windows, `children` held, `n` digest-changing re-ships.
    const N_RESHIPS: u64 = 3;
    let run = |w: u64, children: &[RealmId]| -> (u64, BTreeMap<RealmId, usize>, u64) {
        let cfg = config();
        let clock = ClockSample {
            local_tick: vd_core::TickId(4),
            universe_tick: UniverseTick(0),
            epoch: vd_core::EpochId(1),
            synced: true,
        };
        let mut held = RelayHeld::default();
        let mut windows = OpenWindows::default();
        for i in 0..w {
            windows.0.insert(
                (NodeId(400 + i), WindowId(1)),
                OpenWindow::opened(WindowScope::Occupants, clock.local_tick),
            );
        }
        let mut stats = StubStats::default();
        let mut outbox = OutboundBox::default();
        for t in 0..N_RESHIPS {
            for c in children {
                let (own, interior) = batch_at(*c, t);
                let digest = relay_entry_digest(Fence(5), &own, &interior);
                held.0.insert(
                    *c,
                    RelayHeldEntry {
                        seen: clock.local_tick,
                        fence: Fence(5),
                        own,
                        interior,
                        digest,
                    },
                );
            }
            emit_window_relays(
                &cfg,
                &clock,
                Fence(6),
                &mut held,
                &mut windows,
                &mut stats,
                &mut outbox,
            );
        }
        // THE WIRE, matched byte-for-byte against a CONSTRUCTED expectation (g_verbatim's
        // move): the emit order is deterministic (BTreeMap: windows by node, then held
        // children by realm), so every message's full bytes are re-derivable — no decode,
        // no refutable match, and per-child blob sizes fall out of the zip. One child's
        // blob size must be constant across re-ships (values change, layout does not).
        let mut kids: Vec<RealmId> = children.to_vec();
        kids.sort_unstable();
        let mut blobs: BTreeMap<RealmId, usize> = BTreeMap::new();
        let mut total = 0u64;
        let mut i = 0usize;
        for t in 0..N_RESHIPS {
            for wi in 0..w {
                for c in &kids {
                    let (own, interior) = batch_at(*c, t);
                    let expected = postcard::to_allocvec(&ShardToGateway::WindowRelayed {
                        realm_fence: Fence(6),
                        window: WindowId(1),
                        child: *c,
                        child_fence: Fence(5),
                        statements: own,
                        interior,
                    })
                    .expect("encodes");
                    let (to, _, bytes, _) = &outbox.0[i];
                    assert_eq!(*to, NodeId(400 + wi), "the emit order is the derived one");
                    assert_eq!(
                        &bytes[..],
                        &expected[..],
                        "each relay is byte-identical to its constructed expectation"
                    );
                    total += bytes.len() as u64;
                    let prior = blobs.insert(*c, bytes.len());
                    assert_eq!(
                        prior.unwrap_or(bytes.len()),
                        bytes.len(),
                        "one child's blob size is constant across re-ships"
                    );
                    i += 1;
                }
            }
        }
        assert_eq!(i, outbox.0.len(), "every emitted message was matched");
        (total, blobs, stats.window_relays_forwarded)
    };
    // W = 1, C = 1 — the base measurement.
    let (total_w1, blobs_w1, fwd_w1) = run(1, &[home]);
    let blob = *blobs_w1.get(&home).expect("the home blob was measured");
    assert_eq!(
        fwd_w1, N_RESHIPS,
        "every tick re-ships (the digest changed)"
    );
    assert_eq!(
        total_w1,
        blob as u64 * N_RESHIPS,
        "W=1, C=1: egress == blob × rate"
    );
    // THE GATEWAY MULTIPLIER, measured: two windows double the egress exactly.
    let (total_w2, _, fwd_w2) = run(2, &[home]);
    assert_eq!(fwd_w2, 2 * N_RESHIPS);
    assert_eq!(
        total_w2,
        2 * total_w1,
        "W=2 doubles the egress: the W term is real, not assumed"
    );
    // THE CHILD MULTIPLIER, measured: a second held child (the ring sibling, same 5-planet
    // class) adds exactly its own per-child blob, per window, per re-ship.
    let (total_c2, blobs_c2, fwd_c2) = run(2, &[home, sibling]);
    let blob_sib = *blobs_c2
        .get(&sibling)
        .expect("the sibling blob was measured");
    assert_eq!(fwd_c2, 2 * 2 * N_RESHIPS);
    assert_eq!(
        total_c2,
        2 * N_RESHIPS * (blob as u64 + blob_sib as u64),
        "W=2, C=2: egress == W × Σ_child blob × rate",
    );
    // THE PIN (slice 6): the departure-fixture blob in bytes — §5.2's ~1 045 B estimate,
    // measured. A wire or codec change that moves this number flips it loudly.
    // RE-BASELINED 1118 → 1142 B at the cell activation (real-scale addendum §A6.4, the ONE
    // measured wire cost): every pose now rides normalized, so the postcard varints carry the
    // integer cells (+24 B here). Schema unchanged, PROTO_MINOR unmoved.
    assert_eq!(
        blob, 2124,
        "the pinned departure-fixture relay blob (bytes)"
    );
    eprintln!(
        "[slice6] RELAY EGRESS measured == formula: blob {blob} B (sibling {blob_sib} B; \
         §5.2 modelled ~1045 B) · W=1 egress {total_w1} B / {N_RESHIPS} re-ships · W=2 \
         doubles exactly · W=2,C=2 {total_c2} B == 2 × {N_RESHIPS} × ({blob}+{blob_sib}). \
         At the §5.1 reference model (W=50, C=3, 50 Hz re-ship): {:.1} MB/s per shard",
        50.0 * 3.0 * blob as f64 * 50.0 / 1.0e6,
    );
}

/// look_horizon.md §6 SLICE 6 — THE UNION OVER-DRAW, measured at interim scale (D-LOOK-2):
/// the shared per-realm verdict is the UNION over every observer's in-band set (the owner's
/// one-verdict-per-realm ruling), so rows are drawn for one observer because ANOTHER
/// observer's verdict is more generous. Two occupants of THE world's galaxy — one parked
/// near the home star, one 300 m short of a ring sibling — and the measurement: the fold's
/// published verdict equals the union of the two out-of-band singleton sets, and each
/// observer over-draws exactly the OTHER's child (its own look + its 5-planet interior = 6
/// drawn realm rows per fold, at the departure-fixture row model). The number the D-LOOK-2
/// ledger row records; UNMEASURED at near-real scale, by design.
#[test]
fn slice6_the_union_over_draw_is_measured_at_interim_scale() {
    let cfg_w = vd_physics::worldgen::UniverseConfig::world(15.0, 0.05);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg_w);
    let galaxy = vd_core::worldgen::GALAXY;
    let scope = world.neighbourhood(&BTreeSet::from([galaxy]));
    let galaxy_row = scope
        .iter()
        .find(|r| r.realm == galaxy)
        .expect("the galaxy is rostered");
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("home");
    // ★ AT THE GALAXY'S RUNG, NOT THE CHILD'S (slice S9). Every region read here is a direct child of
    // the galaxy, so its `center` is stated in the GALAXY's frame — and reading it with the child's
    // own ruler was out by 2048×. What made this one instructive is that the ORACLE and the fixture's
    // observer positions both used the wrong ruler, so they agreed with each other perfectly; only
    // the REAL fold, which reads the centre correctly, disagreed. Two wrongs that agree look exactly
    // like a right answer until something honest shows up.
    // The step is read off the galaxy ROW, which is the parent of every region this closure sees —
    // `metres_in` takes the parent itself, so a child's step cannot be substituted for it.
    let centre_of = |r: &vd_core::geometry::RealmRegion| r.center.metres_in(galaxy_row);
    let systems: Vec<(RealmId, DVec3)> = scope
        .iter()
        .filter(|r| r.parent == Some(galaxy))
        .map(|r| (r.realm, centre_of(r)))
        .collect();
    assert_eq!(systems.len(), 3, "THE galaxy holds 3 systems");
    let (sib, sib_centre) = *systems
        .iter()
        .find(|(s, _)| *s != home)
        .expect("a ring sibling");
    // Observer A: 300 m from the home star (deep inside its 11 458 m visibility band).
    // Observer B: 300 m short of the ring sibling, on the line toward the galaxy origin.
    let pos_a = DVec3::new(0.0, 0.0, -300.0);
    let pos_b = sib_centre - sib_centre.normalize() * 300.0;
    // THE OUT-OF-BAND SINGLETON SETS (the oracle): which children each observer's own
    // position puts in band — the per-observer verdict a per-session fold WOULD have
    // computed. Asserted disjoint singletons so the union measurement has teeth.
    let in_band_of = |pos: DVec3| -> BTreeSet<RealmId> {
        scope
            .iter()
            .filter(|r| r.parent == Some(galaxy))
            .filter(|r| r.aoi.in_range(false, (centre_of(r) - pos).length()))
            .map(|r| r.realm)
            .collect()
    };
    let set_a = in_band_of(pos_a);
    let set_b = in_band_of(pos_b);
    assert_eq!(
        set_a,
        BTreeSet::from([home]),
        "A sees exactly the home star"
    );
    assert_eq!(set_b, BTreeSet::from([sib]), "B sees exactly its sibling");
    // THE REAL FOLD, both observers held: the galaxy shard's own AoI tick publishes the
    // SHARED verdict.
    let cfg = StubConfig {
        realm: galaxy,
        held_realms: StubConfig::single_realm(galaxy),
        frame: galaxy_row.frame,
        own_coord: vd_core::worldgen::coord_of_realm(&scope, galaxy)
            .expect("the galaxy has a lineage"),
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    // The grant, keyed on THE GALAXY (the shared `grant_realm` helper grants the default
    // rig realm, which this fixture is not).
    {
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(galaxy),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
        );
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes,
        }]);
    }
    plant_aoi(&mut rig, scope.clone());
    let own_frame = galaxy_row.frame;
    for (i, pos) in [(1u32, pos_a), (2, pos_b)] {
        let entity = EntityId::pack(EntityKind::Player, 60, u64::from(i), i);
        let mut dot = slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) });
        dot.pose = vd_core::pose::StampedPose::at_rest(own_frame, pos, UniverseTick(100));
        rig.world
            .resource_mut::<Dots>()
            .0
            .insert(SessionId(u128::from(i)), dot);
    }
    let _ = rig.tick(vec![]);
    let verdict = rig.world.resource::<InBandVerdict>().0.clone();
    let union: BTreeSet<RealmId> = set_a.union(&set_b).copied().collect();
    assert_eq!(
        verdict, union,
        "the shared verdict IS the union of the observers' own sets"
    );
    // THE OVER-DRAW, recorded (D-LOOK-2's interim-scale measurement): each observer's fold
    // carries the OTHER observer's child — its own look plus its interior's 5 planets = 6
    // drawn realm rows per fold that this observer's own position never asked for.
    let interior_rows_of =
        |s: RealmId| 1 + vd_physics::worldgen::moving_children_for_config(0, &cfg_w, s).len();
    let overdraw_a: usize = union.difference(&set_a).map(|s| interior_rows_of(*s)).sum();
    let overdraw_b: usize = union.difference(&set_b).map(|s| interior_rows_of(*s)).sum();
    assert_eq!(
        (overdraw_a, overdraw_b),
        (10, 10),
        "each observer over-draws exactly the other's 1-look + 9-planet subtree"
    );
    eprintln!(
        "[slice6] UNION OVER-DRAW at interim scale (D-LOOK-2): union verdict {} children \
         (A's own {}, B's own {}); over-draw {} rows per fold per observer (1 own look + 5 \
         interior rows for the child only the OTHER observer needs) — measured at interim \
         scale ONLY; the near-real-scale number stays owed on the ledger row",
        union.len(),
        set_a.len(),
        set_b.len(),
        overdraw_a,
    );
}

/// G-STRUCTURAL-SEAL (look_horizon.md §6 slice 3): production code in `vd-sim` contains NO
/// call to any relay-open function — the HR1 guarantee across two hops as a TEST, not a
/// claim. The scan strips line comments and stops at each file's `#[cfg(test)]` boundary
/// (tests may lawfully open a seal to assert its content; a realm never does). Two positive
/// controls keep it honest: the needle exists in the raw sources (so the scanner hunts a
/// real name), and the crate's own files were actually walked.
#[test]
fn g_structural_seal_vd_sim_production_code_calls_no_open_function() {
    let needle = ["open_", "relay"].concat(); // split so this file's own scan cannot self-trip
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    let mut pending = vec![src];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("vd-sim src dir reads") {
            let path = entry.expect("dir entry reads").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                // EVERY file — a stray non-source file reads lossily and scans inert,
                // which is cheaper than an uncoverable extension-filter arm (HR5).
                files.push(path);
            }
        }
    }
    assert!(
        files.iter().any(|p| p.ends_with("stub.rs")),
        "anti-vacuity: the holder's own file is in the walk"
    );
    let mut raw_hits = 0usize;
    for path in &files {
        let content =
            String::from_utf8_lossy(&std::fs::read(path).expect("source reads")).into_owned();
        raw_hits += content.matches(&needle).count();
        let production = content
            .split_once("#[cfg(test)]")
            .map_or(content.as_str(), |(prod, _)| prod);
        for (n, line) in production.lines().enumerate() {
            let code = line.split("//").next().expect("split yields a first part");
            let clean = !code.contains(&needle);
            let at = format!("{}:{}", path.display(), n + 1);
            assert!(
                clean,
                "G-STRUCTURAL-SEAL: {at} calls a relay-open function in production code — \
                 a realm may hold and forward sealed bytes, never read them (HR1; the Q2 \
                 ruling; look_horizon.md §2 ASK A)"
            );
        }
    }
    assert!(
        raw_hits > 0,
        "anti-vacuity: the needle never appears anywhere — the scanner hunts a dead name"
    );
}

/// Slice C1, the Q2-relay ship (the up-shape ship's rewritten successor — owner-approved
/// 2026-08-16, owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): with a
/// resolved parent this shard ships its VERBATIM self-authored statements — sealed — on the
/// parent's resolve (the relay-subscription moment), holds its tongue while nothing changed
/// off the cadence, re-asserts on the AoI cadence (a restarted parent holds relays only in
/// RAM), ships nothing with no parent, and a shard with no self-look and no markers ships
/// nothing at all (absence of data, never an empty batch).
#[test]
fn the_relay_ship_sends_sealed_statements_on_resolve_change_and_cadence() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0),
        ],
    );
    // Parent resolves at an OFF-cadence tick: the memo never matches a fresh parent, so the
    // resolve itself ships (the relay-subscription moment) — no waiting for a beat.
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    rig.set_local_tick(31);
    let ships = window_relays(&rig.tick(vec![]));
    assert_eq!(ships.len(), 1, "the parent's resolve ships immediately");
    assert_eq!(ships[0].0, ANCESTOR, "addressed to the resolved parent");
    assert_eq!(
        ships[0].1.child.lowered(),
        OWN_REALM,
        "the routing key is this shard's own coord"
    );
    // The seal opens (HERE, in the test — the PARENT never opens it) to the child's verbatim
    // statements. ONE constructed-expected equality (HR5: no let-else destructure whose
    // refusal arm a green test can never take): the authored interior level FIRST — built
    // from the same placement head the producer read — then the realm's own look about
    // ITSELF (SL3: the boot extent, nothing else).
    let opened = vd_wire::session_flow::open_relay_statements(&ships[0].1.own)
        .expect("the child's own seal opens");
    let (expected_rows, at) = {
        let cfg_realm = rig.world.resource::<StubConfig>().realm;
        let at = rig.world.resource::<ClockSample>().universe_tick;
        let placements = rig.world.resource::<Placements>();
        let head = placements.0.head(cfg_realm).expect("the writer authored");
        let regions = rig.world.resource::<RealmRegions>();
        (regions.authored_realm_snaps(cfg_realm, head), at)
    };
    assert_eq!(
        expected_rows.iter().map(|r| r.realm).collect::<Vec<_>>(),
        vec![OTHER_REALM],
        "the authored interior — one row per direct child, nothing deeper"
    );
    assert_eq!(
        opened,
        vec![
            vd_wire::session_flow::RelayedStatement::Level {
                at,
                rows: expected_rows,
            },
            vd_wire::session_flow::RelayedStatement::Body {
                subject: OWN_REALM,
                stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                    bag: vd_core::look::look_bag(&own_region().shape),
                },
                authored_at: at,
            },
            vd_wire::session_flow::RelayedStatement::Body {
                subject: OTHER_REALM,
                stmt: vd_wire::session_flow::BodyStmt::Marker {
                    // The presence floor (look_horizon.md slice 1): the non-glowing child's
                    // point of light — one radius, nothing else.
                    luma: vd_core::look::marker_bag(None, 100.0),
                },
                authored_at: at,
            },
        ],
        "verbatim: the level leads, the self-look follows, one marker per direct child, \
         nothing deeper rides"
    );
    // Unchanged + off-cadence: send-on-change holds its tongue.
    rig.set_local_tick(32);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "nothing changed, no beat — no ship"
    );
    // The AoI cadence re-asserts (tick 40, cadence 10): a restarted parent re-learns the seal.
    rig.set_local_tick(40);
    assert_eq!(
        window_relays(&rig.tick(vec![])).len(),
        1,
        "the cadence re-assert ships"
    );
    // Parent unresolved (the root; a boot race): nothing ships, even on the cadence.
    rig.world.resource_mut::<ParentRealmNode>().0 = None;
    rig.set_local_tick(50);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "no resolved parent, no ship"
    );
    // A shard whose realm is absent from its forest states no look, and with no marker roster
    // it has NO bodies: parent resolved and on the cadence, it still ships nothing.
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    plant_aoi(&mut rig, vec![root_region()]);
    rig.set_local_tick(60);
    assert!(
        window_relays(&rig.tick(vec![])).is_empty(),
        "no bodies to state — an empty batch is never sent"
    );
}

#[test]
fn the_two_read_only_diagnosis_accessors_answer_both_ways() {
    // `RelayHeld::statements_for` and `AoiState::in_band` are the ONLY windows a harness or a
    // gate has onto two private stores, so both arms of each are driven here rather than only
    // from the crates that read them (HR5: a crate covers its own surface).
    let mut held = RelayHeld::default();
    assert_eq!(
        held.statements_for(OTHER_REALM),
        None,
        "nothing held for a child that never spoke"
    );
    held.0.insert(
        OTHER_REALM,
        RelayHeldEntry {
            seen: vd_core::TickId(1),
            fence: Fence(2),
            own: vec![7, 7, 7],
            interior: Vec::new(),
            digest: relay_entry_digest(Fence(2), &[7, 7, 7], &[]),
        },
    );
    assert_eq!(
        held.statements_for(OTHER_REALM),
        Some(vec![7, 7, 7]),
        "the SEALED bytes come back verbatim — this hands them over, it never opens them"
    );
    assert!(
        !AoiState::default().in_band(),
        "a fresh latch is out of band"
    );
    let (_, entered) = aoi_transition(AoiState::default(), true, 0);
    assert!(
        entered.expect("entering yields a state").in_band(),
        "and a latch that entered says so"
    );
}

#[test]
fn a_realm_states_its_own_look_with_no_child_and_no_parent() {
    // THE ROOM, STATED BY THE PARTY THAT OWNS IT. The box a player is standing INSIDE used to
    // be authored by the router out of its own copy of the seed forest. It is stated here
    // instead, by the shard that owns the realm, out of the one geometric fact a realm holds
    // about ITSELF: its own boundary. Nothing about where it sits is involved, so the ground
    // rule is untouched — and since Slice C2 nobody else can state it at all (SL3 reached).
    //
    // The fixture is deliberately BARE: no AoI child anywhere, no parent, nothing arriving.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let bodies = window_bodies(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(bodies.len(), 1, "exactly one body: this realm's own look");
    let (to, window, subject, stmt) = &bodies[0];
    assert_eq!(*to, GATEWAY, "routed to the window's opener");
    assert_eq!(*window, WindowId(1));
    assert_eq!(
        *subject, OWN_REALM,
        "a realm states a look only about itself"
    );
    // Compared as a WHOLE constructed value (HR5: a `let…else { panic! }` leaves an
    // uncoverable false arm): the statement IS a self-look carrying this realm's own boundary,
    // verbatim — and no position, because the type has no such field.
    assert_eq!(
        *stmt,
        BodyStmt::SelfLook {
            bag: vd_core::look::look_bag(&own_region().shape)
        },
        "a realm's own body is a SelfLook of its own boundary, never a marker"
    );
    // Send-on-change: an unchanged look is not re-stated next tick.
    assert!(
        window_bodies(&rig.tick(vec![])).is_empty(),
        "the room does not re-state itself every tick"
    );
}

#[test]
fn a_shard_whose_own_realm_is_absent_from_its_forest_states_no_look() {
    // The refusal arm of `own_shape`. A shard planted with a forest its own realm is not in
    // cannot state a boundary for itself — and the honest answer is to say nothing, not to
    // invent one. (`own_frame` degrades the same way, to the ambient root's frame.) With no
    // child carrying a marker either, the window is served no body at all.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region()]); // no `own_region()`
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    assert_eq!(
        rig.world
            .resource::<RealmRegions>()
            .own_shape(OWN_REALM)
            .map(|s| s.realm),
        None,
        "a realm absent from the planted forest has no outline to state"
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    assert!(
        window_bodies(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty(),
        "no look invented for a realm this shard was never told the shape of"
    );
}

#[test]
fn the_membership_verdict_holds_a_passed_realm_across_grace() {
    // No flicker: a realm the dot has PASSED stays in the verdict for the whole grace window
    // (the grace latch holds `next_in` true), so NOTHING ships until the grace lapses — then
    // exactly one removal. The verdict's anti-blink guarantee, the direct twin of the demand
    // grace hold, on the SAME transition (HR3 — no second AoI derivation exists).
    let grace = 3;
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, grace),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    // Tick 1: acquire ⇒ the realm is ADDED.
    let v = window_memberships(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(v.len(), 1, "one add on acquire");
    assert_eq!(v[0].2, vec![OTHER_REALM], "the child it acquired");
    // Move OUT of the AoI band ⇒ grace begins; the realm is HELD (still `next_in`) ⇒ nothing
    // ships for `grace` ticks.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0));
    for _ in 0..grace {
        assert!(
            window_memberships(&rig.tick(vec![])).is_empty(),
            "a passed realm is held across the grace window — no flicker"
        );
    }
    // Grace lapses ⇒ the realm finally leaves the verdict (exactly one removal).
    let v = window_memberships(&rig.tick(vec![]));
    assert_eq!(v.len(), 1, "one removal once grace lapses");
    assert!(v[0].2.is_empty(), "nothing added on the final release");
    assert_eq!(v[0].3, vec![OTHER_REALM], "the passed realm is removed");
}

/// ★ A STATIC ROSTER IS STATED ONCE, NOT TWENTY TIMES A SECOND (slice S10; owner-approved
/// 2026-08-27). THIS IS THE TEST THE CHANGE EXISTS FOR.
///
/// Before the split, the author's FULL direct-child roster rode the per-tick frame on the latest-wins
/// UNRELIABLE lane, where repetition IS the loss story. That is right for a mover and wrong for a star.
/// MEASURED at the target census: 150,000 rows × 95 bytes × 20 Hz = **285 MB/s per subscriber**, and not
/// one byte of it changes — a galaxy's systems do not move, which the world's own placement golden pins
/// across ticks 0, 1,000 and 50,000. The 14.2 MB message does not fit a datagram at all.
///
/// Driven over MANY ticks so a per-tick regression cannot hide inside a two-tick window.
#[test]
fn a_static_roster_is_stated_once_and_the_frame_keeps_only_its_stamp() {
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert_eq!(window_static_rows(&sent).len(), 1, "stated on open");

    // TWENTY MORE TICKS — one second of a world where nothing moves.
    let mut rosters = 0usize;
    let mut frames = 0usize;
    for t in 2..=21 {
        rig.set_local_tick(t);
        let sent = rig.tick(vec![]);
        rosters += window_static_rows(&sent).len();
        frames += window_frames(&sent).len();
    }
    // ★ THE ASSERTION. Under the old lane this would be TWENTY — one full roster per tick.
    assert_eq!(
        rosters, 0,
        "a roster that never changes must say nothing for a whole second"
    );
    // …and the frame still arrives every tick, because its STAMP is what the composer aligns chains
    // on. The saving is the rows, never the beat: a missing frame would be a different defect.
    assert_eq!(frames, 20, "the per-tick stamp is still owed every tick");
    // …carrying no rows at all, because every child here is static.
    rig.set_local_tick(22);
    let sent = rig.tick(vec![]);
    assert!(window_frames(&sent)[0].4.is_empty());
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_static_rows_sent, 1, "once, across 22 ticks");
    assert_eq!(
        stats.window_frame_rows_sent, 0,
        "no row rode the lossy lane"
    );
}

/// ★ THE SKY IS STATED ONCE PER SUBSCRIBER, IN PARTS (S11).
///
/// The catalogue is the only message whose size grows with the world — 7.0 MB at the target census,
/// which no carrier takes whole — and none of it changes, because stars do not move. So it is chunked,
/// and it is stated once: the generation IS the content, so a subscriber holding that number holds that
/// sky and needs nothing further.
///
/// Driven over many ticks, so a per-tick regression cannot hide inside a two-tick window.
#[test]
fn the_star_catalogue_is_stated_once_per_subscriber_in_parts() {
    let mut rig = window_rig();
    let star = |n: u64| vd_core::look::StarRow {
        realm: RealmId::System(n),
        cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
        class_code: 6,
        luma_lsun: 0.25,
    };
    let rows: Vec<vd_core::look::StarRow> = (1u64..=3).map(star).collect();
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
    *rig.world.resource_mut::<crate::stub::StarCatalogue>() = crate::stub::StarCatalogue {
        rows: rows.clone(),
        generation,
    };

    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let parts = star_catalogue_parts(&sent);
    assert!(!parts.is_empty(), "the sky is stated on open");
    // Every part carries the SAME generation — parts of two skies can never be spliced into a galaxy
    // that never existed.
    assert!(parts.iter().all(|p| p.0 == generation));
    // The parts reassemble to exactly the catalogue, in order.
    let seen: Vec<vd_core::look::StarRow> = parts.iter().flat_map(|p| p.3.clone()).collect();
    assert_eq!(seen, rows, "the parts are the sky, unaltered and in order");
    // …and they agree about how many there are, or a receiver can never know it is whole.
    let total = parts[0].2;
    assert!(parts.iter().all(|p| p.2 == total));
    assert_eq!(parts.len() as u32, total);

    // ★ TWENTY MORE TICKS, AND NOT ONE MORE PART. Under a per-tick lane this would be 7.0 MB × 20 at
    // the census; here the generation already matches what the subscriber holds.
    let mut later = 0usize;
    for t in 2..=21 {
        rig.set_local_tick(t);
        later += star_catalogue_parts(&rig.tick(vec![])).len();
    }
    assert_eq!(
        later, 0,
        "a sky that did not change says nothing for a second"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().star_catalogue_parts_sent as usize,
        parts.len(),
        "stated once, across 21 ticks"
    );
}

/// ★ THE SKY IS NOT RE-SENT WHEN A SUBSCRIBER RE-ASSERTS (S11).
///
/// **A WARP LEG IS TWO CROSSINGS** — out of your star system and into the next — and each one shakes
/// the subscription. If any of that re-issues the catalogue, a single journey re-transmits the whole
/// sky TWICE: 14 MB at the census, for a galaxy that did not move. Getting this wrong is worse than
/// never having split the lane at all.
///
/// The keep-alive re-assert is the sharp case, because it deliberately CLEARS the other send-on-change
/// baselines so the subscriber is re-served its bodies and its verdict. The sky must not ride that
/// beat: its baseline is keyed by GATEWAY and by the generation, so a re-assert says nothing about
/// whether the sky changed — and the sky did not change.
#[test]
fn a_keep_alive_re_assert_does_not_re_send_the_sky() {
    let mut rig = window_rig();
    let rows: Vec<vd_core::look::StarRow> = (1u64..=3)
        .map(|n| vd_core::look::StarRow {
            realm: RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        })
        .collect();
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
    *rig.world.resource_mut::<crate::stub::StarCatalogue>() =
        crate::stub::StarCatalogue { rows, generation };
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let first = star_catalogue_parts(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert!(!first.is_empty(), "the sky is stated on open");

    // THE KEEP-ALIVE RE-ASSERT: the same open again, which is the derived keep-alive. It CLEARS the
    // body and membership baselines by design — and must leave the sky alone.
    rig.set_local_tick(5);
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert!(
        !window_bodies(&sent).is_empty(),
        "the re-assert really did re-serve the bodies — without this the sky's silence proves nothing"
    );
    assert!(
        star_catalogue_parts(&sent).is_empty(),
        "★ and the sky rode none of it: a warp leg is two crossings, so a sky on this beat is the \
         whole galaxy twice per journey"
    );

    // A SECOND WINDOW on the same gateway is not a second sky either — the catalogue is not
    // window-scoped, and the client already holds it.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowOpen {
            window: WindowId(2),
            scope: WindowScope::Occupants,
        },
    )]);
    assert!(
        star_catalogue_parts(&sent).is_empty(),
        "a second window onto the same sky is not a second sky"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().star_catalogue_parts_sent as usize,
        first.len(),
        "stated exactly once across every re-assert and every window"
    );
}

/// ★ THE CASE THE PLAN ACTUALLY WARNS ABOUT: a window CLOSED and re-opened (S11).
///
/// A warp leg is two crossings — out of your star system and into the next. Crossing INTO the galaxy
/// opens a window on the galaxy shard; crossing OUT closes it; the return leg opens it again. If each
/// open re-issues the catalogue, one journey re-transmits the whole sky TWICE.
///
/// This drives it, and RECORDS WHAT TODAY ACTUALLY DOES rather than what it should do.
#[test]
fn a_closed_and_re_opened_window_re_states_the_sky_today() {
    let mut rig = window_rig();
    let rows: Vec<vd_core::look::StarRow> = (1u64..=3)
        .map(|n| vd_core::look::StarRow {
            realm: RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        })
        .collect();
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
    *rig.world.resource_mut::<crate::stub::StarCatalogue>() =
        crate::stub::StarCatalogue { rows, generation };
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let first = star_catalogue_parts(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert!(!first.is_empty());

    // LEAVE: the window closes, exactly as it does when an occupant crosses out of this realm.
    rig.set_local_tick(5);
    let _ = rig.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowClose {
            window: WindowId(1),
        },
    )]);

    // RETURN: the second crossing of the journey re-opens it.
    rig.set_local_tick(6);
    let again = star_catalogue_parts(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));

    // ★ THE MEASUREMENT, and it is the plan's warning made concrete. The sender remembers what IT
    // sent, and it forgets a gateway that holds no window — so the return leg re-states the whole sky
    // even though the client still holds it. At the census that is 7.0 MB per crossing, 14 MB per
    // round trip, for a galaxy that did not move.
    //
    // **THE CURE IS NOT A LONGER MEMORY ON THE SENDER.** S11 names it: the RECEIVER states the
    // generation it holds and the sender answers from THAT. A sender's memory is wrong across a
    // reconnect, a restart, or a second gateway — the client is the only party that knows what it has.
    // That needs the client's on-disk cache to exist, so it is the next piece and this test flips with
    // it (D-S11-SKY).
    // ★ MEASURED: the return leg states NOTHING — the right answer, reached for a fragile reason.
    // The emit runs only while some window is open, so a shard whose LAST window closed never reaches
    // the line that forgets the gateway. With a SECOND gateway keeping the shard busy, that line does
    // run and the closed gateway IS forgotten — proved below.
    assert!(
        again.is_empty(),
        "the return leg re-states nothing while this is the only gateway"
    );

    // THE FRAGILITY, MEASURED. A second gateway keeps a window open, so the emit runs every tick and
    // the first gateway's entry is dropped the moment its window closes. Its return leg then re-states
    // the whole sky — 7.0 MB at the census, per crossing, for a galaxy that did not move.
    let other = NodeId(77);
    let _ = rig.tick(vec![wire_msg(
        other,
        MsgClass::Control,
        &GatewayToShard::WindowOpen {
            window: WindowId(9),
            scope: WindowScope::Occupants,
        },
    )]);
    rig.set_local_tick(7);
    let _ = rig.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowClose {
            window: WindowId(1),
        },
    )]);
    rig.set_local_tick(8);
    let re_crossed =
        star_catalogue_parts(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(
        re_crossed.len(),
        first.len(),
        "★ WITH ANOTHER GATEWAY PRESENT, the return leg DOES re-state the whole sky"
    );

    // ★ AND THIS IS NOT A BUG TO PATCH — IT IS WHY S11 SPECIFIES THE OTHER DESIGN.
    //
    // The sender is remembering what IT sent. That memory is wrong in BOTH directions and it cannot
    // tell the two apart:
    //   FORGET TOO EAGERLY -> a returning client is re-sent a sky it already holds (measured above).
    //   REMEMBER TOO LONG  -> a NEW subscriber behind the same gateway is told nothing, and has no sky
    //                          at all, because a previous session once held one.
    //
    // No length of memory fixes that, because the sender is not the party that knows. **THE RECEIVER
    // STATES THE GENERATION IT HOLDS AND THE SENDER ANSWERS FROM THAT** — S11's own wording. It needs
    // the client's on-disk cache to exist, so it is the next piece, and this test flips with it.
    // Ledgered at D-S11-SKY.
}

/// ★ A SHARD THAT PARENTS NO STARS STATES NO SKY (S11) — most shards, and the arm that would otherwise
/// ship an empty catalogue to every subscriber of every planet in the galaxy.
#[test]
fn a_shard_with_no_stars_states_no_catalogue() {
    let mut rig = window_rig(); // its StarCatalogue resource is default: empty
    let sent = rig.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Occupants,
        },
    )]);
    assert!(star_catalogue_parts(&sent).is_empty());
    assert_eq!(
        rig.world.resource::<StubStats>().star_catalogue_parts_sent,
        0
    );
}

/// ★ A CHANGED BODY STILL SHIPS (slice S10) — the half a DIGEST could break.
///
/// The send-on-change baseline used to keep each subject's WHOLE bag and compare it byte by byte. It
/// keeps an 8-byte digest now, which removes one held vector per direct child per subscriber (150,000
/// of them at the target census), the per-tick comparison over all of them, and the free-plus-realloc
/// of every one on each keep-alive beat.
///
/// The existing suite pins the "unchanged holds its tongue" half. NOTHING pinned the other half, and it
/// is the one a digest can get wrong: if the baseline stopped noticing a real change, a subscriber would
/// keep drawing a stale body and no test would say so. This drives it — the same subject, a genuinely
/// different bag, must re-ship.
#[test]
fn a_changed_body_re_ships_past_the_digest_baseline() {
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let first = window_bodies(&sent);
    assert!(!first.is_empty(), "bodies ship on open");

    // Tick again, unchanged: the baseline holds its tongue. (Also the control for the assertion
    // below — without it, a re-ship could be the lane shipping every tick regardless.)
    rig.set_local_tick(2);
    assert!(
        window_bodies(&rig.tick(vec![])).is_empty(),
        "unchanged — nothing re-ships"
    );

    // NOW CHANGE THE CHILD'S OWN DATUM. Its marker bag genuinely differs, so the baseline must notice.
    rig.world
        .resource_mut::<ChildLuma>()
        .0
        .insert(OTHER_REALM, (6, 0.75));
    rig.set_local_tick(3);
    let after = window_bodies(&rig.tick(vec![]));
    assert!(
        after.iter().any(|b| b.2 == OTHER_REALM),
        "a changed body must re-ship: a digest that stopped noticing would leave the subscriber \
         drawing a stale marker forever, and only this assertion would see it. Got {after:?}"
    );
}

#[test]
fn an_occupants_window_ships_rows_bodies_and_membership_send_on_change() {
    // §2.9 steps 1/3/4/5 on an Occupants window: ONE WindowFrame per tick (hop: None, the
    // authored roster verbatim, stamped at the one universe tick), the self-look + the armed
    // child's marker ON OPEN, the SL7 verdict as one added-batch — then a SECOND tick ships
    // the per-tick frame again and NOTHING else (send-on-change holds its tongue).
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1, "ONE frame per tick per open window");
    let (to, window, at, hop, rows) = &frames[0];
    assert_eq!(*to, GATEWAY);
    assert_eq!(*window, WindowId(1));
    assert_eq!(*at, UniverseTick(100), "stamped at the one universe tick");
    assert_eq!(*hop, None, "an occupant already stands in this frame");
    // ★ RE-BASED IN S10: the per-tick frame carries the children that MOVE. Both children in this
    // fixture are static, so the frame's row list is EMPTY and the roster arrives once on the reliable
    // lane instead — 285 MB/s per subscriber saved at the target census, for bytes that never change.
    assert!(
        rows.is_empty(),
        "a static child does not ride the per-tick frame: {rows:?}"
    );
    // …and the FULL roster arrived ONCE, on the reliable lane, in the sender's own frame.
    let statics = window_static_rows(&sent);
    assert_eq!(statics.len(), 1, "one static roster per open window");
    let rows = &statics[0].2;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].realm, OTHER_REALM);
    assert_eq!(rows[0].frame, frame_of(OTHER_REALM));
    assert_eq!(rows[0].pose.frame, frame_of(OWN_REALM));
    assert_eq!(fm(rows[0].pose.pos), WINDOW_CHILD_CENTER);
    assert_eq!(rows[1].realm, RealmId::Planet(43));
    // The bodies: the realm's OWN look (its boot extent, SL3) + one point-of-light marker
    // per direct child, GLOWING OR NOT (look_horizon.md slice 1's presence floor): the armed
    // child's datum + extent, the quiet child's extent ALONE (the one-radius law).
    let bodies = window_bodies(&sent);
    assert_eq!(bodies.len(), 3);
    assert_eq!(bodies[0].2, OWN_REALM);
    assert_eq!(
        bodies[0].3,
        BodyStmt::SelfLook {
            bag: vd_core::look::look_bag(&Boundary::Shell { r: 100_000.0 })
        },
        "the look IS the realm's own boot extent, framed by the one shared codec"
    );
    assert_eq!(
        bodies[1],
        (
            GATEWAY,
            WindowId(1),
            OTHER_REALM,
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(Some((6, 0.25)), 100.0)
            }
        )
    );
    assert_eq!(
        bodies[2],
        (
            GATEWAY,
            WindowId(1),
            RealmId::Planet(43),
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(None, 100.0)
            }
        ),
        "the non-glowing child still states a correctly-sized point of light"
    );
    // The verdict: the armed child is in the dot's band; the quiet child (inert band) is not.
    assert_eq!(
        window_memberships(&sent),
        vec![(GATEWAY, WindowId(1), vec![OTHER_REALM], vec![])]
    );
    // Second tick: the per-tick frame repeats (its STAMP is what the composer aligns chains on, so it
    // is owed every tick even when it carries no rows); bodies, membership and the static roster are
    // all send-on-change quiet.
    rig.set_local_tick(2);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent).len(), 1);
    assert_eq!(window_bodies(&sent), vec![]);
    assert_eq!(window_memberships(&sent), vec![]);
    assert_eq!(
        window_static_rows(&sent),
        vec![],
        "★ THE WHOLE POINT: a roster that did not change says nothing on the second tick"
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_frames_sent, 2);
    // ★ RE-BASED IN S10, 4 → 0: both children are static, so no row rides the per-tick frame at all.
    // This counter is now the MOVING row count, which is what the lossy lane should ever carry.
    assert_eq!(stats.window_frame_rows_sent, 0);
    assert_eq!(stats.window_bodies_sent, 3);
    assert_eq!(stats.window_memberships_sent, 1);
    // …and the static roster was stated ONCE across BOTH ticks — not twice.
    assert_eq!(stats.window_static_rows_sent, 1);
    assert_eq!(stats.windows_open, 1);
}

/// THE PRESENCE FLOOR's producer half (look_horizon.md slice 1, Q2 APPROVED 2026-08-17 — the
/// STATION-LAPSE gate's root cause, DECLARED RED BEFORE THE SLICE): EVERY direct child states
/// a point-of-light marker — a glowing child's carries its photometric datum plus its
/// circumscribed extent, a non-glowing child's the extent alone. Before the floor, a
/// non-glowing child (a station, a city, a ship) had NO lawful bag content at all, so the
/// moment its own picture lapsed it VANISHED instead of degrading to a correctly-sized point.
#[test]
fn every_direct_child_states_a_point_of_light_marker_glowing_or_not() {
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let bodies = window_bodies(&sent);
    assert_eq!(
        bodies.len(),
        3,
        "the realm's own look + a marker for BOTH direct children (the quiet, non-glowing \
         child included — the presence floor): {bodies:?}",
    );
    let marker_of = |realm: RealmId| {
        bodies
            .iter()
            .find(|(_, _, subject, _)| *subject == realm)
            .map(|(_, _, _, stmt)| stmt.clone())
            .unwrap_or_else(|| panic!("{realm:?} states no marker — the presence floor hole"))
    };
    // The armed (glowing) child: its photometric datum AND its circumscribed extent, in the
    // ONE marker bag.
    assert_eq!(
        marker_of(OTHER_REALM),
        BodyStmt::Marker {
            luma: vd_core::look::marker_bag(Some((6, 0.25)), 100.0)
        },
    );
    // The quiet (non-glowing) child: the extent ALONE — one radius, nothing else (the
    // one-radius law: a bound is a promise about space; a look is a statement about
    // appearance).
    assert_eq!(
        marker_of(RealmId::Planet(43)),
        BodyStmt::Marker {
            luma: vd_core::look::marker_bag(None, 100.0)
        },
    );
}

#[test]
fn the_keep_alive_re_assert_re_serves_the_whole_set_so_silence_means_a_dead_realm() {
    // WINDOW LANE SLICE D — the beat the gateway's roster-loss window is derived from. A
    // send-on-change lane that only ever speaks on CHANGE cannot distinguish "nothing changed"
    // from "the realm behind this statement stopped speaking", so a keep-alive `WindowOpen`
    // (which the subscriber already sends on its own derived cadence) clears the baselines and
    // re-serves the whole body set and the whole membership verdict. Without this, a departed
    // system's last look would sit in the composer forever and never shrink back to a dot.
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let first_bodies = window_bodies(&sent);
    let first_membership = window_memberships(&sent);
    assert_eq!(first_bodies.len(), 3, "the full set on open");
    assert_eq!(first_membership.len(), 1);
    // A quiet tick is still quiet — the re-assert is the BEAT, not every tick.
    rig.set_local_tick(2);
    assert_eq!(window_bodies(&rig.tick(vec![])), vec![]);
    // The keep-alive: the SAME window, the SAME scope. Byte-for-byte the same statements come
    // back out (the realm is still stating exactly what it stated), counted as a re-assert.
    rig.set_local_tick(3);
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert_eq!(window_bodies(&sent), first_bodies);
    assert_eq!(window_memberships(&sent), first_membership);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_reasserted, 1);
    assert_eq!(stats.windows_opened, 1, "a keep-alive is not a new window");
    assert_eq!(stats.window_bodies_sent, 6, "three served twice");
    assert_eq!(stats.window_memberships_sent, 2);
}

#[test]
fn a_child_window_ships_the_hop_row_pre_inverted_at_the_author() {
    // §2.2 R1: the hop row is the author's own frame expressed in the child's frame, the ONE
    // inversion made at the author. Identity orientation + zero spin (every placement of THE
    // world today) ⇒ the inversion is exactly the negated placement — asserted EXACTLY.
    let mut rig = window_rig();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(2),
        scope: WindowScope::Child(OTHER_REALM),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    let (_, window, _, hop, rows) = &frames[0];
    assert_eq!(*window, WindowId(2));
    // ★ RE-BASED IN S10: the roster rides the reliable static lane, and it is the same roster whatever
    // the scope — which is what this line always asserted, now read where the rows are.
    assert!(rows.is_empty(), "no mover rides this fixture's frame");
    let statics = window_static_rows(&sent);
    assert_eq!(statics.len(), 1);
    assert_eq!(
        statics[0].2.len(),
        2,
        "the same authored roster rides every scope"
    );
    let hop = hop.as_ref().expect("a Child window carries the hop row");
    assert_eq!(hop.child, OTHER_REALM);
    // The inverted row is NORMALIZED (transfer_frame's output invariant since the cell
    // activation): the negated placement's value rides the integer anchor.
    assert_eq!(fm(hop.inv.anchor()), -WINDOW_CHILD_CENTER);
    assert_eq!(hop.inv.origin, DVec3::ZERO, "sub-cell residual only");
    assert_eq!(hop.inv.velocity, DVec3::ZERO);
    assert_eq!(hop.inv.orientation, DQuat::IDENTITY);
    assert_eq!(hop.inv.angular_velocity, DVec3::ZERO);
}

#[test]
fn the_hop_inversion_carries_velocity_and_spin_through_the_frame_core() {
    // A MOVING, SPINNING child (identity orientation, cell zero — invertible today): the hop
    // row's velocity is the frame core's own answer (the parent origin as seen from the
    // rotating child: q⁻¹(ω×o − v)) and the angular term is the child's spin reversed and
    // re-axed (−(q⁻¹·ω)) — exact numbers, no epsilon.
    let mut rig = Rig::new();
    rig.grant_realm();
    let mover = RealmId::Planet(45);
    let spin = RealmRegion {
        aoi: aoi_band(0),
        ..region(mover, Some(OWN_REALM), DVec3::ZERO, 100.0)
    };
    let placed = FramePlacement {
        origin_cell: vd_core::glam::I64Vec3::ZERO,
        origin: DVec3::new(7.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
        orientation: DQuat::IDENTITY,
        angular_velocity: DVec3::new(0.0, 0.0, 0.5),
    };
    let motions: BTreeMap<RealmId, MotionFn> =
        [(mover, MotionFn(std::sync::Arc::new(move |_| placed)))]
            .into_iter()
            .collect();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), spin]).with_moving_children(motions);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let open = GatewayToShard::WindowOpen {
        window: WindowId(3),
        scope: WindowScope::Child(mover),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    let hop = frames[0].3.as_ref().expect("hop row");
    assert_eq!(fm(hop.inv.anchor()), DVec3::new(-7.0, 0.0, 0.0));
    // ω×o − v = (0,0,0.5)×(7,0,0) − (0,1,0) = (0,3.5,0) − (0,1,0) = (0,2.5,0).
    assert_eq!(hop.inv.velocity, DVec3::new(0.0, 2.5, 0.0));
    assert_eq!(hop.inv.angular_velocity, DVec3::new(0.0, 0.0, -0.5));
}

#[test]
fn assert_window_emission_feature_anywhere() {
    // HR4 G-IDENTICAL: the IDENTICAL window-emission feature — one frame per tick per window,
    // the pre-inverted hop at the author, the look/marker bodies, the SL7 verdict — on a
    // SYSTEM shard (child = Planet) AND a PLANET shard (child = Area), under their REAL
    // capability profiles. The emitter is kind-BLIND, so the two runs differ ONLY in the
    // realm ids they name: every VALUE (poses, hop numbers, bags, stamps) is byte-equal.
    let a = drive_window_emit(
        NodeKind::Shard(crate::capability::profiles::system().expect("system profile")),
        OWN_REALM,
        FrameRef::SystemSpace { system_seed: 7 },
        RealmRegion {
            aoi: aoi_band(3),
            ..region_framed(
                OTHER_REALM,
                Some(OWN_REALM),
                WINDOW_CHILD_CENTER,
                100.0,
                FrameRef::PlanetCentered { planet_seed: 42 },
            )
        },
    );
    let b = drive_window_emit(
        NodeKind::Shard(crate::capability::profiles::planet().expect("planet profile")),
        RealmId::Planet(42),
        FrameRef::PlanetCentered { planet_seed: 42 },
        RealmRegion {
            aoi: aoi_band(3),
            ..region_framed(
                RealmId::Area(99),
                Some(RealmId::Planet(42)),
                WINDOW_CHILD_CENTER,
                100.0,
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
            )
        },
    );
    let (a_frames, a_bodies, a_members, a_static) = a;
    let (b_frames, b_bodies, b_members, b_static) = b;
    assert_eq!(a_frames.len(), 2, "one frame per open window (System run)");
    assert_eq!(b_frames.len(), 2, "one frame per open window (Planet run)");
    // ★ THE ROW COMPARISON, MOVED TO THE STATIC LANE (S10) so this gate keeps its teeth. One roster
    // per open window on each run; the VALUES must be identical across the two shard kinds and only
    // the realm NAMES may differ — which is the whole of what G-IDENTICAL asserts.
    assert_eq!(
        a_static.len(),
        2,
        "one static roster per window (System run)"
    );
    assert_eq!(
        b_static.len(),
        2,
        "one static roster per window (Planet run)"
    );
    for ((_, aw, arows), (_, bw, brows)) in a_static.iter().zip(b_static.iter()) {
        assert_eq!(aw, bw, "same window order");
        assert_eq!(arows.len(), 1);
        assert_eq!(brows.len(), 1);
        assert_eq!(arows[0].pose.pos, brows[0].pose.pos);
        assert_eq!(arows[0].pose.vel, brows[0].pose.vel);
        assert_eq!(arows[0].realm, OTHER_REALM);
        assert_eq!(brows[0].realm, RealmId::Area(99));
    }
    for ((_, aw, aat, ahop, arows), (_, bw, bat, bhop, brows)) in
        a_frames.iter().zip(b_frames.iter())
    {
        assert_eq!(aw, bw, "same window order");
        assert_eq!(aat, bat, "same universe stamp");
        // ★ RE-BASED IN S10. The child in both runs is STATIC, so its row rides the reliable static
        // roster and NOT the per-tick frame — the frame carries movers, and there are none here.
        // Comparing two empty lists would make this gate vacuous, so the comparison MOVES to where the
        // rows actually are (below); what stays here is that the frame agrees across kinds on being
        // empty, which is itself the kind-blindness this gate is about.
        assert!(arows.is_empty());
        assert!(brows.is_empty());
        // Option equality on the INV alone (HR5: plain equality, no match with an
        // uncoverable divergence arm): both absent on the Occupants frame, both the SAME
        // pre-inverted numbers on the Child frame — the shape and the values in one compare.
        assert_eq!(
            ahop.as_ref().map(|h| h.inv),
            bhop.as_ref().map(|h| h.inv),
            "the hop inversion is shape- and value-identical across kinds"
        );
        if let Some(h) = ahop {
            assert_eq!(h.child, OTHER_REALM);
        }
        if let Some(h) = bhop {
            assert_eq!(h.child, RealmId::Area(99));
        }
    }
    // The bodies, as WHOLE expected sets (HR5: plain equality, never matches!): the look and
    // marker BAGS are the same byte strings in both runs — the same boot extent through the
    // one codec, the same planted datum — and only the subject ids differ per run.
    let look = vd_core::look::look_bag(&Boundary::Shell { r: 100_000.0 });
    // The presence floor (look_horizon.md slice 1): the planted datum + the child's
    // circumscribed extent, one bag through the one codec, identical bytes on every profile.
    let luma = vd_core::look::marker_bag(Some((6, 0.25)), 100.0);
    let expected = |own: RealmId, child: RealmId| {
        vec![
            (
                GATEWAY,
                WindowId(1),
                own,
                BodyStmt::SelfLook { bag: look.clone() },
            ),
            (
                GATEWAY,
                WindowId(1),
                child,
                BodyStmt::Marker { luma: luma.clone() },
            ),
            (
                GATEWAY,
                WindowId(2),
                own,
                BodyStmt::SelfLook { bag: look.clone() },
            ),
            (
                GATEWAY,
                WindowId(2),
                child,
                BodyStmt::Marker { luma: luma.clone() },
            ),
        ]
    };
    assert_eq!(a_bodies, expected(OWN_REALM, OTHER_REALM));
    assert_eq!(b_bodies, expected(RealmId::Planet(42), RealmId::Area(99)));
    // The verdicts: the Occupants window ships the dot's in-band child; the Child window's
    // occupied-child proxy verdict is EMPTY here (no live bit) ⇒ exactly ONE message per run.
    assert_eq!(a_members.len(), 1);
    assert_eq!(b_members.len(), 1);
    assert_eq!(a_members[0].1, WindowId(1));
    assert_eq!(b_members[0].1, WindowId(1));
    assert_eq!(a_members[0].2, vec![OTHER_REALM]);
    assert_eq!(b_members[0].2, vec![RealmId::Area(99)]);
}

#[test]
fn a_dead_gateways_windows_die_by_the_derived_ttl_with_zero_leaked_emissions() {
    // THE SLICE-A CHAOS GATE (window_lane.md §4 Slice A "TTL-expiry chaos"), in-proc form of
    // the process-tier gateway kill (`rlm_demand_login`'s `kill_and_reap`): a gateway opens
    // windows, then DIES — its keep-alives stop. The shard's windows must expire on the
    // DERIVED TTL (2 beats + 1, owner law 3(a)) and not one emission may leak past expiry.
    let mut rig = window_rig();
    let sent = rig.tick(vec![
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(OTHER_REALM),
            },
        ),
    ]);
    assert_eq!(window_frames(&sent).len(), 2, "both windows serve");
    let ttl = window_ttl_ticks(&config());
    assert_eq!(
        ttl,
        RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(&config()) + 1,
        "the TTL is the derived 2-beats+1, never a literal"
    );
    // The LAST tick inside the window: a whole TTL of silence is still served (one lost
    // keep-alive never blinks a live subscriber).
    rig.set_local_tick(1 + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent).len(), 2, "alive at the TTL edge");
    assert_eq!(rig.world.resource::<StubStats>().windows_open, 2);
    // One past: BOTH windows die BEFORE anything ships — zero leaked emissions after expiry.
    rig.set_local_tick(2 + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(window_frames(&sent), vec![], "no frame leaks past the TTL");
    assert_eq!(window_bodies(&sent), vec![], "no body leaks past the TTL");
    assert_eq!(
        window_memberships(&sent),
        vec![],
        "no verdict leaks past the TTL"
    );
    assert_eq!(rig.world.resource::<OpenWindows>().0.len(), 0);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_ttl_expired, 2, "each expiry counted");
    assert_eq!(stats.windows_open, 0, "the gauge reads the empty registry");
    // And a keep-alive RESETS the clock: a re-opened window re-asserted at the edge survives
    // the next whole TTL from THAT refresh.
    let open = GatewayToShard::WindowOpen {
        window: WindowId(9),
        scope: WindowScope::Occupants,
    };
    rig.set_local_tick(100);
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    rig.set_local_tick(100 + ttl);
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]); // the re-assert
    rig.set_local_tick(100 + ttl + ttl);
    let sent = rig.tick(vec![]);
    assert_eq!(
        window_frames(&sent).len(),
        1,
        "the refreshed window outlives the original TTL horizon"
    );
}

#[test]
fn inv_body_at_origin_and_the_rotated_hop_inertness_are_pinned_on_the_world() {
    // TWO NAMED PINS ON THE WORLD (window_lane.md §2.12; SL5 — the one seed universe, at the
    // shipped posture the boot fences enforce):
    // 1. INV-BODY-AT-ORIGIN — a realm's own body sits at its own frame origin, so the origin
    //    of ITS frame, re-expressed through its parent's authored book, IS the parent's
    //    placement row, exactly; and the pre-inverted hop row round-trips back to the origin
    //    exactly.
    // 2. ROTATED-CROSS-CELL INERTNESS — the hop inversion cannot be refused on THE world
    //    today (a MEASUREMENT that fails the day the world grows a spinning realm a
    //    cell-block out before the P10 cell math lands — never an argument).
    let cfg = vd_physics::worldgen::UniverseConfig::world(
        vd_physics::worldgen::VISUAL_OCCUPANT_V_MAX_MPS,
        vd_physics::worldgen::AOI_TICK_DT_S,
    );
    let forest = vd_physics::worldgen::realm_regions_for_config(0, &cfg);
    let anchors: BTreeSet<RealmId> = forest.iter().filter_map(|r| r.parent).collect();
    let mut movers: BTreeMap<RealmId, vd_physics::celestial::OrbitalElements> = BTreeMap::new();
    for &anchor in &anchors {
        movers.extend(vd_physics::worldgen::moving_children_for_config(
            0, &cfg, anchor,
        ));
    }
    let regions = RealmRegions::new(forest).with_moving_children(kepler_motion_fns(movers));
    let tick_hz = 1.0 / vd_physics::worldgen::AOI_TICK_DT_S;
    let mut rows_pinned = 0usize;
    let mut moved_since_epoch = 0usize;
    let mut cross_rung_hops = 0usize;
    for &t in &[UniverseTick(0), UniverseTick(50_000)] {
        for &anchor in &anchors {
            let own = regions.own_frame(anchor);
            let book = regions.author_book(anchor, tick_hz, t);
            for (region, pose) in regions.child_rows(anchor, &book) {
                rows_pinned += 1;
                // (1a) The child's own body — the ORIGIN of its own frame — maps through the
                // parent's book onto EXACTLY the authored placement row.
                let body = transfer_frame(
                    &StampedPose::at_rest(region.frame, DVec3::ZERO, t),
                    own,
                    &book,
                )
                .expect("every direct child of THE world resolves through its parent's book");
                assert_eq!(
                    body.pos, pose.pos,
                    "{anchor} -> {}: body at origin",
                    region.realm
                );
                assert_eq!(body.vel, pose.vel);
                // (2) THE INVERSION, AND THE LIMIT IT HAS NOW REACHED (slice S9).
                //
                // ★ THIS PIN WAS BUILT TO FAIL ONE DAY, AND THAT DAY IS TODAY — which is the test
                // working, not breaking. It asserted the hop inversion is NEVER refused on THE world.
                // The inversion states the PARENT's body in the CHILD's frame, so it must count the
                // parent's origin in the child's unit. A star system counts in millimetres and its
                // galaxy's centre is now 0.7 LIGHT YEARS away, against a millimetre lattice that
                // reaches a quarter of one. There is no such count, and there never will be: this is
                // not a bug to fix but a fact about units, and the honest thing is to state it.
                //
                // So the pin splits by whether the two frames count in the SAME unit, and both arms
                // are asserted — the working one exactly, the refused one by name and with its
                // numbers, so it stays a measurement rather than becoming a silence.
                let inv = match invert_hop_placement(own, region.frame, &book) {
                    Ok(inv) => inv,
                    Err(refused) => {
                        // ★ THE LIMIT, NAMED. It is about REACH and not about the units differing:
                        // a galaxy sitting at the universe's origin inverts perfectly well across two
                        // rungs, because zero has a count in every unit. What has no count is a
                        // DISTANCE too large for the finer one. Asserted, so a rotation failure or an
                        // unknown frame arriving here would not be quietly absorbed as "expected".
                        assert!(
                            matches!(
                                refused,
                                vd_core::frame::FrameError::CrossTierCrossing(
                                    vd_core::pose::TierConversionError::BeyondReach { .. }
                                )
                            ),
                            "{anchor} -> {}: the only lawful refusal here is reach: {refused:?}",
                            region.realm
                        );
                        // …and it is only ever the coarse-to-fine direction that can run out.
                        assert!(
                            own.tier().step_exponent() > region.frame.tier().step_exponent(),
                            "a hop into a COARSER unit can always be counted"
                        );
                        cross_rung_hops += 1;
                        continue;
                    }
                };
                // (1b) ...and it round-trips: the parent's body, as the hop states it in the
                // child's frame, maps back to the parent's own origin EXACTLY (identity
                // orientations everywhere today ⇒ bit-exact, no epsilon).
                let back = transfer_frame(
                    &StampedPose {
                        frame: region.frame,
                        pos: LatticePos::at(inv.origin_cell, inv.origin),
                        vel: inv.velocity,
                        orient: inv.orientation,
                        universe_tick: t,
                    },
                    own,
                    &book,
                )
                .expect("the inverse rides the same book");
                assert_eq!(back.pos.cell(), vd_core::glam::I64Vec3::ZERO);
                assert_eq!(fm_at(back.pos, own.tier()), DVec3::ZERO);
                assert_eq!(back.vel, DVec3::ZERO);
                // ★ THE SIXTEENTH SITE, AND THE MOST INSTRUCTIVE (slice S9). Both sides of this
                // comparison were read at the CHILD's step, when both quantities are stated in the
                // ANCHOR's: the authored pose comes out of the anchor's own book, and the centre is
                // a position in the anchor's frame. Two wrongs of the same size cancel, so the
                // comparison gave the right answer for the wrong reason and nothing ever failed.
                //
                // Both now read `own`, the anchor's frame, which is the one unit they share.
                if t == UniverseTick(50_000)
                    && pose
                        .pos
                        .delta_m(vd_core::pose::LatticePos::ORIGIN, own.tier())
                        != region
                            .center
                            .in_parents_frame()
                            .delta_m(vd_core::pose::LatticePos::ORIGIN, own.tier())
                {
                    moved_since_epoch += 1;
                }
            }
        }
    }
    // NON-VACUITY, pinned: THE world's parent-authored rows — the galaxy under the universe,
    // three systems under the galaxy, five planets under each system — at BOTH instants.
    assert_eq!(
        rows_pinned,
        2 * (1 + 3 + 27 + 3 + 6),
        "THE world's full child-row set (galaxy + systems + planets + the T2 stars + the \
         T3 census moons)"
    );
    // ...and the movers actually MOVED between the two instants (the pin measured a live
    // world, not a static fixture): every planet is off its zeroed region center at t=50000.
    assert_eq!(
        moved_since_epoch, 33,
        "all twenty-seven planets and six census moons author live placements"
    );
    // ★ AND THE REFUSAL ARM IS NOT VACUOUS (slice S9). Pinned as a count, because "some hops are out
    // of reach" is worthless without knowing how many: if this silently went to zero the arm above
    // would stop being exercised and the pin would quietly narrow to the easy cases.
    //
    // FOUR: the two RING star systems, at both instants. Not six — the HOME system is anchored at
    // the galaxy's own origin, and zero has a count in every unit, so its hop inverts perfectly well
    // across the two rungs. The ring siblings sit roughly 0.7 light years out, and a system counts in
    // millimetres, which reach a quarter of a light year.
    //
    // That the home system is the exception is worth more than the number: it means the galaxy's
    // ORIGIN is the only place in it a millimetre-counting realm can name its parent from.
    //
    // Every other row in THE world — planets, stars, moons under their own parents, and the galaxy
    // under the universe — either shares its parent's unit or sits close enough to count.
    assert_eq!(
        cross_rung_hops, 4,
        "the two RING systems, at both instants, cannot count their galaxy's centre in millimetres"
    );
}

#[test]
fn the_live_siblings_interior_is_one_level_out_and_never_deeper_q1() {
    // THE Q1 FENCE PIN (owner ruling 2026-08-16, window_lane.md §5 RULINGS + §2.12): one
    // level into ANY live realm you are next to — generic, never a planet-specific case —
    // and NEVER deeper. Structurally: a parent's window statements name AT MOST its direct
    // children (its held interior outlines for a live child do NOT enter the window lane —
    // the parent's per-child message stays a placement and nothing else, SL3); the ONE level
    // of interior comes from the live realm's OWN window lane. Chained, an observer's
    // windows reach exactly two levels — inside the two-level visibility guarantee the boot
    // fence (`guard_visibility_climb_bounded`, look_horizon slice 2 — the MEASURED climb
    // against the look carrier's arity) proves on THE world.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    // (Since Slice C2 there is no store a parent could hold a child's interior in AT ALL —
    // the up-shape lane and its holding died together, so the leak this pin guards against is
    // now unrepresentable as well as untaken. HALF 1 below still measures it.)
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let sent = rig.tick(vec![
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::Planet(42)),
            },
        ),
    ]);
    // HALF 1 — the PARENT's lane: every realm the window statements name is a DIRECT child
    // (or the parent itself, as a body subject). The held grandchild outline is NOWHERE.
    let direct: BTreeSet<RealmId> = [RealmId::Planet(42), RealmId::Planet(43)]
        .into_iter()
        .collect();
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 2, "both windows serve (non-vacuous)");
    // ★ RE-BASED IN S10: the roster rides the reliable static lane now, one per open window. The
    // non-vacuity this guarded is preserved — it just reads the lane the rows actually travel on.
    let statics = window_static_rows(&sent);
    assert_eq!(statics.len(), 2, "a roster per window (non-vacuous)");
    for (_, _, rows) in &statics {
        assert!(
            !rows.is_empty(),
            "the roster rides every window (non-vacuous)"
        );
        for row in rows {
            assert!(
                direct.contains(&row.realm),
                "a window frame row names a non-direct-child: {}",
                row.realm
            );
        }
    }
    for (_, _, subject, _) in &window_bodies(&sent) {
        // Bitwise `|` (both operands pure, HR5 — a short-circuit RHS is an uncoverable region).
        assert!(
            (*subject == OWN_REALM) | direct.contains(subject),
            "a body subject beyond one level: {subject}"
        );
    }
    for (_, _, added, removed) in &window_memberships(&sent) {
        for id in added.iter().chain(removed.iter()) {
            assert!(direct.contains(id), "a verdict id beyond one level: {id}");
        }
    }
    // HALF 2 — the LIVE realm's OWN lane states the one level of interior (Q1 = YES), on a
    // DIFFERENT realm kind (a Planet shard with an Area child — the G-IDENTICAL discipline):
    // its window frame names ITS direct children and nothing deeper exists to leak.
    let planet = RealmId::Planet(42);
    let cfg = StubConfig {
        realm: planet,
        held_realms: StubConfig::single_realm(planet),
        frame: frame_of(planet),
        own_coord: StubConfig::root_coord(planet),
        ..config()
    };
    let mut inner = Rig::with_config(cfg);
    grant_realm_for(&mut inner, planet);
    plant_aoi(
        &mut inner,
        vec![
            region_framed(planet, None, DVec3::ZERO, 100_000.0, frame_of(planet)),
            region_framed(
                RealmId::Area(99),
                Some(planet),
                DVec3::new(1.0, 2.0, 3.0),
                5.0,
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
            ),
        ],
    );
    insert_owned_dot_framed(
        &mut inner,
        TRIG_SESSION,
        player(8),
        frame_of(planet),
        DVec3::new(50.0, 0.0, 0.0),
    );
    let sent = inner.tick(vec![wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Occupants,
        },
    )]);
    let frames = window_frames(&sent);
    assert_eq!(frames.len(), 1);
    // ★ RE-BASED IN S10: the area is STATIC, so the realm states it on the reliable roster rather than
    // on the per-tick frame. The Q1 fence this asserts — ONE level in, stated by that realm itself — is
    // unchanged; only which lane carries the statement moved.
    assert!(frames[0].4.is_empty(), "no mover at this level");
    let statics = window_static_rows(&sent);
    assert_eq!(statics.len(), 1);
    assert_eq!(
        statics[0].2.iter().map(|r| r.realm).collect::<Vec<_>>(),
        vec![RealmId::Area(99)],
        "one level INTO the live realm — stated by that realm itself"
    );
}

#[test]
fn a_child_window_guards_unrostered_ship_and_rotated_hops_counted() {
    // The Child-scope guards, each dropped + counted, never a guess and never a panic:
    // a stranger realm (`window_child_unrostered`), a Ship child (D-SHIP-1 — no lineage
    // coord until P8, the SAME counter every coord lane uses), and a ROTATED placement BEYOND
    // THE EXACT ROTATION REACH (the frame core's own restated refusal — real-scale addendum
    // §A4.6: an in-reach rotated hop now FOLDS exactly, so the guard fires only past
    // 2⁴² m ≈ 29.4 AU — `window_hop_refused`, R2's P10 trigger). The Occupants window beside
    // them keeps serving: one bad hop never mutes the lane.
    let ship_realm = RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 7, 3));
    let rotated = RealmId::Planet(44);
    let mut rig = Rig::new();
    rig.grant_realm();
    let placed = FramePlacement {
        // 2⁵³ cells = 2× the FINE rotation reach: the one rotated shape the restated law
        // still refuses.
        origin_cell: vd_core::glam::I64Vec3::new(1_i64 << 53, 0, 0),
        origin: DVec3::new(5.0, 0.0, 0.0),
        velocity: DVec3::ZERO,
        orientation: DQuat::from_rotation_z(0.3),
        angular_velocity: DVec3::ZERO,
    };
    let motions: BTreeMap<RealmId, MotionFn> =
        [(rotated, MotionFn(std::sync::Arc::new(move |_| placed)))]
            .into_iter()
            .collect();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(rotated, Some(OWN_REALM), DVec3::ZERO, 100.0),
        region(
            ship_realm,
            Some(OWN_REALM),
            DVec3::new(50_000.0, 0.0, 0.0),
            10.0,
        ),
    ])
    .with_moving_children(motions);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(5_000.0, 0.0, 0.0));
    let opens: Vec<Inbound> = [
        (WindowId(1), WindowScope::Occupants),
        (WindowId(2), WindowScope::Child(RealmId::Planet(555))), // rostered NOWHERE
        (WindowId(3), WindowScope::Child(ship_realm)),           // D-SHIP-1
        (WindowId(4), WindowScope::Child(rotated)),              // the P10-owed refusal
    ]
    .into_iter()
    .map(|(window, scope)| {
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen { window, scope },
        )
    })
    .collect();
    let sent = rig.tick(opens);
    let frames = window_frames(&sent);
    assert_eq!(
        frames.iter().map(|f| f.1).collect::<Vec<_>>(),
        vec![WindowId(1)],
        "only the Occupants window serves — every guarded Child window is withheld"
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.window_child_unrostered, 1);
    // The ship exclusion counts once in the AoI/demand fold and once on the window lane —
    // the SAME per-lane-pass counting the other D-SHIP-1 guards use.
    assert_eq!(stats.ship_child_regions_excluded, 2);
    assert_eq!(stats.window_hop_refused, 1);
    assert_eq!(
        stats.windows_open, 4,
        "guarded windows stay open — only their frames are withheld"
    );
    // The refusal is the frame core's own (one rule, inherited): asserted directly, plus the
    // unknown-frame arm the emitter can never reach (the roster resolves first).
    let book = rig
        .world
        .resource::<Placements>()
        .0
        .head(OWN_REALM)
        .expect("authored")
        .clone();
    assert_eq!(
        invert_hop_placement(frame_of(OWN_REALM), frame_of(rotated), &book),
        Err(FrameError::RotationBeyondExactReach)
    );
    assert_eq!(
        invert_hop_placement(
            frame_of(OWN_REALM),
            FrameRef::PlanetCentered { planet_seed: 777 },
            &book
        ),
        Err(FrameError::UnknownDestFrame)
    );
}

#[test]
fn window_membership_rides_the_one_fold_and_clears_with_the_last_observer() {
    // §2.9 step 5: the verdict is THE existing aoi_decide fold — per-dot for Occupants
    // (scoped to the OPENER's own dots: a second gateway's window holds an EMPTY verdict),
    // the occupied-child proxy set for Child scopes — and when the LAST observer leaves, the
    // emptiness pass clears every window's verdict (the removals ship once).
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let sent = rig.tick(vec![
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        // A SECOND subscriber (another gateway) with no dots here: its per-dot verdict is
        // empty — the fold is scoped per opener, never a global union.
        wire_msg(
            ANCESTOR,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowClose {
                window: WindowId(7), // unknown — the counted no-op rides the same tick
            },
        ),
        wire_msg(
            GATEWAY,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::Planet(42)),
            },
        ),
    ]);
    let members = window_memberships(&sent);
    // GATEWAY's dot stands in BOTH armed bands (42 at the origin, 43 at 500) ⇒ its Occupants
    // verdict is both; the occupied child 42 (reach = its own 1000 m extent) also reaches
    // both ⇒ the Child(42) verdict is both; ANCESTOR's window ships NOTHING (empty verdict).
    let both = vec![RealmId::Planet(42), RealmId::Planet(43)];
    assert_eq!(
        members,
        vec![
            (GATEWAY, WindowId(1), both.clone(), vec![]),
            (GATEWAY, WindowId(2), both.clone(), vec![]),
        ]
    );
    // The LAST observer leaves (the dot detaches, the child bit expires): the emptiness pass
    // ships the removals — every window's verdict returns to the empty set, exactly once.
    rig.world.resource_mut::<Dots>().0.clear();
    rig.world.resource_mut::<ChildLiveness>().0.clear();
    rig.set_local_tick(2);
    let sent = rig.tick(vec![]);
    assert_eq!(
        window_memberships(&sent),
        vec![
            (GATEWAY, WindowId(1), vec![], both.clone()),
            (GATEWAY, WindowId(2), vec![], both),
        ]
    );
    assert_eq!(
        rig.world.resource::<StubStats>().window_close_unknown,
        1,
        "the unknown close counted"
    );
}

/// ★ THE SKY BEATS EVEN THOUGH NOTHING CHANGED (S11; the SL6-approved arm, owner 2026-08-27).
///
/// This is the one statement on this lane that must NOT be suppressed by send-on-change, and the test
/// is the mirror image of the one above it: there, twenty ticks and not one more part; here, twenty
/// ticks and the beat KEEPS COMING.
///
/// The reason is that silence on the catalogue lane is doing two jobs at once. A galaxy does not
/// change, so a perfectly healthy server says nothing there essentially forever — which is
/// indistinguishable from an emitter that died:
///
/// ```text
///   the sky did not change   ──┐
///                              ├──► ...both look EXACTLY like this: nothing arrives.
///   the emitter is broken    ──┘
/// ```
///
/// The beat makes the healthy case audible. It also names the generation, so a client learns not only
/// that somebody is working but WHICH sky that somebody believes is current.
#[test]
fn the_sky_beats_on_a_cadence_even_when_it_did_not_change() {
    let mut rig = window_rig();
    let rows: Vec<vd_core::look::StarRow> = (1u64..=3)
        .map(|n| vd_core::look::StarRow {
            realm: RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        })
        .collect();
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
    *rig.world.resource_mut::<crate::stub::StarCatalogue>() =
        crate::stub::StarCatalogue { rows, generation };

    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);

    // Twenty ticks with an unchanging sky. The catalogue stays silent; the beat does not.
    let mut parts_later = 0usize;
    let mut beats = Vec::new();
    for t in 2..=21 {
        rig.set_local_tick(t);
        let sent = rig.tick(vec![]);
        parts_later += star_catalogue_parts(&sent).len();
        beats.extend(sky_alive_beats(&sent));
    }
    assert_eq!(parts_later, 0, "the catalogue is still send-on-change");
    assert!(
        !beats.is_empty(),
        "an unchanged sky must still SAY it is unchanged — silence is the failure this arm removes"
    );
    // It names the sky it is speaking for, so the client can tell "unchanged" from "changed, and you
    // missed it" — the second reading the beat exists to give.
    assert!(beats.iter().all(|(_, g)| *g == generation));
    assert!(beats.iter().all(|(to, _)| *to == GATEWAY));
    assert_eq!(
        rig.world.resource::<StubStats>().sky_alive_beats_sent as usize,
        beats.len()
    );
}

/// ★ A SHARD THAT PARENTS NO STARS DOES NOT BEAT (S11).
///
/// The beat vouches for a sky. A shard with no sky has nothing to vouch for, and no client holds a
/// catalogue from it to doubt — so the beat would be a statement with no subject. Same emptiness guard
/// as the catalogue it speaks for, deliberately: two guards that could drift apart would be a shard
/// that beats for a sky it never states.
#[test]
fn a_shard_with_no_sky_states_no_beat() {
    let mut rig = window_rig(); // its StarCatalogue resource is default: empty
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
    };
    let mut beats = Vec::new();
    for t in 1..=21 {
        rig.set_local_tick(t);
        let msgs = if t == 1 {
            vec![wire_msg(GATEWAY, MsgClass::Control, &open)]
        } else {
            vec![]
        };
        beats.extend(sky_alive_beats(&rig.tick(msgs)));
    }
    assert!(beats.is_empty(), "no sky, nothing to vouch for");
    assert_eq!(rig.world.resource::<StubStats>().sky_alive_beats_sent, 0);
}

/// ★ THE BEAT IS PER GATEWAY, NOT PER WINDOW (S11).
///
/// The sky is not window-scoped — it is the same galaxy however many views a gateway holds open. A
/// beat per window would make the cheapest statement on the lane scale with a number it has nothing to
/// do with, which is the shape this whole slice exists to remove.
#[test]
fn the_beat_is_stated_once_per_gateway_however_many_windows_it_holds() {
    let mut rig = window_rig();
    let rows: Vec<vd_core::look::StarRow> = vec![vd_core::look::StarRow {
        realm: RealmId::System(1),
        cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304, 7, -3),
        class_code: 6,
        luma_lsun: 0.25,
    }];
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
    *rig.world.resource_mut::<crate::stub::StarCatalogue>() =
        crate::stub::StarCatalogue { rows, generation };

    let open = |w: u64| GatewayToShard::WindowOpen {
        window: WindowId(w),
        scope: WindowScope::Occupants,
    };
    let _ = rig.tick(vec![
        wire_msg(GATEWAY, MsgClass::Control, &open(1)),
        wire_msg(GATEWAY, MsgClass::Control, &open(2)),
    ]);
    assert_eq!(
        rig.world.resource::<crate::stub::OpenWindows>().0.len(),
        2,
        "the fixture really does hold two windows — else this proves nothing"
    );

    // Counted PER TICK: the claim is that a tick which beats beats ONCE, however many views are open.
    // Counting the total instead would pass even if every tick beat twice.
    let mut per_tick = Vec::new();
    for t in 2..=21 {
        rig.set_local_tick(t);
        per_tick.push(sky_alive_beats(&rig.tick(vec![])).len());
    }
    assert!(
        per_tick.contains(&1),
        "some tick must beat, or the test proves nothing"
    );
    assert_eq!(
        per_tick.iter().copied().max(),
        Some(1),
        "two windows, one sky, one beat — never one per view"
    );
    assert!(per_tick.contains(&0), "and it is a CADENCE, not every tick");
}
