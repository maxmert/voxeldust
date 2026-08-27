//! Occupant input integration and THE SPEED LAW (banner 2831) plus the client-session lane: walk/yaw/pitch-clamp, per-axis clamp and diagonal normalization, the governed ceiling and its ramp, the ceiling applied to a transient, the realm time multiplier; then input-discard reasons, the finite gate, action_bits inertness, session-fence upgrade and stale-fence detach, per-tick snapshot frames and budget partitioning, entity minting and the bounded InputLog. Absorbs the one-test window-control registry run (3336-3404) by locality. Local helper clamped_forest (2836) travels.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn applied_input_moves_the_dot_and_is_logged() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // Forward input, yaw 0: heading is -Z.
    let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    let expected_step = 2.0 * 0.05; // speed * dt
    assert!(
        (total_m(&dot.pose.pos).z + expected_step).abs() < 1e-12,
        "moved -Z"
    );
    assert_eq!(total_m(&dot.pose.pos).x, 0.0);
    assert_eq!(dot.last_applied_seq, Some(1));
    assert_eq!(
        rig.world.resource::<InputLog>().applied(),
        vec![(SESSION, 1)]
    );
    // Velocity is displacement over dt.
    assert!((dot.pose.vel.z + 2.0).abs() < 1e-12);
}

#[test]
fn yaw_rotates_the_heading() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // Look 90° left (yaw = +π/2), then walk forward: heading becomes -X.
    let half_pi = std::f32::consts::FRAC_PI_2;
    let _ = rig.tick(vec![input_msg(
        1,
        Fence(1),
        [1.0, 0.0, 0.0],
        [half_pi, 0.0],
    )]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    let expected_step = 2.0 * 0.05;
    assert!(
        (total_m(&dot.pose.pos).x + expected_step).abs() < 1e-6,
        "moved -X"
    );
    assert!(total_m(&dot.pose.pos).z.abs() < 1e-6);
}

#[test]
fn pitch_is_clamped_at_the_gimbal_pole_never_wraps_past_vertical() {
    // WB-1: a huge look-up delta must NOT accumulate past ±π/2 (which would flip the
    // authoritative orientation). Two big up-pitches in a row stay clamped at the limit.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
    let _ = rig.tick(vec![input_msg(2, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.pitch,
        vd_core::kinematics::PITCH_LIMIT,
        "accumulated pitch is held at the limit, never wrapped past vertical"
    );
}

#[test]
fn strafe_and_vertical_axes_integrate() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // Strafe right + up, no forward; the per-axis clamp catches the out-of-range axis, and
    // THE SPEED LAW (S3) normalizes the over-unit diagonal to the realm's ceiling: the
    // pre-law arithmetic commanded √2·move_speed on this stick — ABOVE the one ceiling every
    // realm now states, which the law no longer permits (`vd_core::flight::
    // throttle_axes_scale`'s over-unit arm — the one measured behaviour change at human
    // scale, and it is a cure). Per axis: (move_speed·dt)/√2; the TOTAL step is exactly the
    // ceiling's move_speed·dt.
    let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 2.0, 1.0], [0.0, 0.0])]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    let expected_step = 2.0 * 0.05 / 2.0_f64.sqrt();
    assert!(
        (total_m(&dot.pose.pos).x - expected_step).abs() < 1e-12,
        "clamped strafe, ceiling-normalized"
    );
    assert!(
        (total_m(&dot.pose.pos).y - expected_step).abs() < 1e-12,
        "vertical, ceiling-normalized"
    );
    assert!(
        (total_m(&dot.pose.pos).length() - 2.0 * 0.05).abs() < 1e-12,
        "the diagonal's TOTAL step is the ceiling exactly — never √2× it"
    );
}

/// ★THE THROWAWAY OVERDRIVE INSTRUMENT, both arms. `set_cruise_overdrive` is the only way the
/// `VD_TEST_OVERDRIVE` knob reaches the sim, and its whole contract is one clamp: the instrument may
/// make a cruise FASTER and may never make it slower, so anything under the lawful `1.0` is refused
/// back to it and the caller is TOLD what was planted (the returned value, not the argument).
///
/// Covered here because it is reachable production code with no other caller inside this crate — the
/// shard binary is its only consumer, and a binary is outside the coverage domain. Without this the
/// clamp could invert and every gate would still be green while a test instrument silently GOVERNED
/// a flight it was only ever allowed to speed up.
#[test]
fn the_cruise_overdrive_instrument_may_only_ever_go_faster() {
    let mut regions = clamped_forest();
    // The default IS the law — an unset knob is inert.
    assert_eq!(regions.cruise_overdrive, 1.0);
    // The lawful arm: a factor above one is planted verbatim and reported back.
    assert_eq!(regions.set_cruise_overdrive(4.0), 4.0);
    assert_eq!(regions.cruise_overdrive, 4.0);
    // The refused arm: below the law it is clamped back to the law, not to the argument.
    assert_eq!(regions.set_cruise_overdrive(0.25), 1.0);
    assert_eq!(regions.cruise_overdrive, 1.0);
    // And the boundary itself is lawful rather than refused.
    assert_eq!(regions.set_cruise_overdrive(1.0), 1.0);
}

#[test]
fn the_governed_ceiling_is_the_realm_cap_lowered_by_the_child_arm() {
    // A wide own realm (cap 2·150 000/180 ≈ 1 667 m/s at foot 2) with one small child at the
    // origin: the governor is the own cap far away, the child arm as the subject nears, and
    // exactly the child's own cap AT the bound.
    let t = flight_tuning(&config());
    let own = region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 150_000.0);
    let child = region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 100.0);
    let regions = RealmRegions::new(vec![root_region(), own, child]);
    let book = regions.author_book(OWN_REALM, 20.0, UniverseTick(10));
    let at = |x: f64| LatticePos::from_metres(DVec3::new(x, 0.0, 0.0), Tier::Fine);
    let own_cap = flight::realm_speed_cap_mps(150_000.0, 2.0, t.traverse_s);
    let child_cap = flight::realm_speed_cap_mps(100.0, 2.0, t.traverse_s);
    assert_eq!(child_cap, 2.0, "a 100 m child clamps to the foot");
    // Far from the child (149 000 m out): the child arm is way above the own cap — the own
    // cap binds.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(149_000.0), &t),
        Some(own_cap),
    );
    // Nearing the child, the arm binds: child_cap + (dist − extent)/τ.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(600.0), &t),
        Some(flight::approach_ceiling_mps(
            child_cap,
            600.0 - 100.0,
            t.tau_s
        )),
    );
    // AT (and inside) the child's bound: exactly the child's own ceiling — you arrive at ITS
    // speed, never through it.
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(100.0), &t),
        Some(child_cap),
    );
    assert_eq!(
        governed_ceiling_in_book(&regions, OWN_REALM, &book, at(50.0), &t),
        Some(child_cap),
    );
    // A childless forest: the own cap alone (the loop's empty arm).
    let bare = RealmRegions::new(vec![root_region(), own]);
    let bare_book = bare.author_book(OWN_REALM, 20.0, UniverseTick(10));
    assert_eq!(
        governed_ceiling_in_book(&bare, OWN_REALM, &bare_book, at(0.0), &t),
        Some(own_cap),
    );
    // A realm absent from the forest: None — no region, no law.
    assert_eq!(
        governed_ceiling_in_book(&bare, OTHER_REALM, &bare_book, at(0.0), &t),
        None,
    );
}

#[test]
fn the_governed_ceiling_resolve_answers_none_off_the_law() {
    // The outer resolve's three no-law arms: an empty forest (unhosted frame), a hosted frame
    // with NO authored book yet (the pre-sync ledger), and the full Some path once the one
    // writer has run.
    let t = flight_tuning(&config());
    let pos = LatticePos::ORIGIN;
    let mut rig = Rig::new();
    // Empty forest (the default resource): the frame resolves to no realm.
    {
        let regions = rig.world.resource::<RealmRegions>();
        let placements = rig.world.resource::<Placements>();
        assert_eq!(
            governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
            None,
        );
    }
    // Planted forest, but the writer has not run (no tick yet): still None — never outrun
    // your feet before the realm has authored its world.
    rig.world.insert_resource(clamped_forest());
    {
        let regions = rig.world.resource::<RealmRegions>();
        let placements = rig.world.resource::<Placements>();
        assert_eq!(
            governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
            None,
        );
    }
    // One synced tick authors the book: the ceiling exists, and at this clamped scale it IS
    // the foot speed exactly.
    let _ = rig.tick(vec![]);
    let regions = rig.world.resource::<RealmRegions>();
    let placements = rig.world.resource::<Placements>();
    assert_eq!(
        governed_ceiling_for_frame(regions, &placements.0, config().frame, pos, &t),
        Some(2.0),
    );
}

#[test]
fn the_speed_law_is_bit_inert_wherever_the_ceiling_clamps() {
    // THE S3 INERTNESS MEASUREMENT, unit tier: the SAME input flight on a forestless rig (the
    // pre-law posture) and on a rig with a planted CLAMPED forest (every ceiling at the foot)
    // lands the dot at BIT-IDENTICAL poses — position, cell, velocity. This is the measured
    // form of "every sub-45 km realm runs at exactly today's speeds"; the process battery is
    // its world-scale twin.
    let fly = |plant: bool| {
        let mut rig = Rig::new();
        if plant {
            rig.world.insert_resource(clamped_forest());
        }
        rig.grant_realm();
        let _ = rig.attach();
        // A fractional stick, a full stick, a diagonal and a look-turn — the input shapes the
        // landed battery flies.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.6, 0.0, 0.0], [0.3, 0.1])]);
        let _ = rig.tick(vec![input_msg(2, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let _ = rig.tick(vec![input_msg(3, Fence(1), [0.0, 0.5, 0.5], [0.0, 0.0])]);
        rig.world.resource::<Dots>().0[&SESSION].pose
    };
    let (bare, planted) = (fly(false), fly(true));
    assert_eq!(bare.pos.cell(), planted.pos.cell());
    assert_eq!(bare.pos.offset(), planted.pos.offset());
    assert_eq!(bare.vel, planted.vel);
    assert_eq!(bare.orient, planted.orient);
}

#[test]
fn full_throttle_rides_the_proportional_ramp_up_to_the_realm_ceiling() {
    // A wide own realm (cap 2·1.8e6/180 = 20 000 m/s at foot 2): holding full throttle from
    // rest compounds the speed by e^(dt/τ) per tick from the foot floor — the §4.2(c) ramp —
    // and parks AT the ceiling, never above it.
    let mut rig = Rig::new();
    rig.world.insert_resource(RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1.8e6),
    ]));
    rig.grant_realm();
    let _ = rig.attach();
    let t = flight_tuning(&config());
    let g = (t.tick_dt_s / t.tau_s).exp();
    let cap = flight::realm_speed_cap_mps(1.8e6, 2.0, t.traverse_s);
    assert_eq!(cap, 20_000.0);
    let mut expected = 2.0; // the foot floor the ramp compounds from
    for seq in 1..=40u64 {
        let _ = rig.tick(vec![input_msg(seq, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        expected = (expected * g).min(cap);
        let v = rig.world.resource::<Dots>().0[&SESSION].pose.vel.length();
        assert!(
            (v - expected).abs() <= expected * 1e-9,
            "tick {seq}: v {v} vs ramp {expected}",
        );
    }
    // Releasing the stick stops DEAD — deceleration is instant (P3 "stopped means stopped");
    // the gradual arrival slow-down is the governor's falling ceiling, never a coast.
    let _ = rig.tick(vec![input_msg(41, Fence(1), [0.0, 0.0, 0.0], [0.0, 0.0])]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose.vel,
        DVec3::ZERO,
    );
}

#[test]
fn a_transient_faster_than_the_governed_ceiling_is_clamped_on_the_crossing_path() {
    // ★OQ-2 (owner-ruled): the realm's ceiling governs everything it contains, piloted or
    // not. On a clamped-scale forest (ceiling = foot = 2 m/s) a 10 m/s debris is cut to the
    // ceiling before its advance; a slower one is untouched BIT-FOR-BIT. (The forestless
    // no-clamp arm is `readvance_advances_held_transients_and_skips_the_uncounted_tiers`.)
    let mut rig = Rig::new();
    rig.world.insert_resource(clamped_forest());
    rig.grant_realm();
    let fast = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let slow = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let pose_with = |vel: DVec3| StampedPose {
        vel,
        ..StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(99))
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        for (id, vel) in [
            (fast, DVec3::new(10.0, 0.0, 0.0)),
            (slow, DVec3::new(0.0, 1.5, 0.0)),
        ] {
            owned.0.insert(
                id,
                Transient {
                    pose: pose_with(vel),
                    anchor_fence: Fence(1),
                    status: TransientStatus::Held { outbound: None },
                    prev_offset: LatticePos::ORIGIN,
                },
            );
        }
    }
    let _ = rig.tick(vec![]);
    let owned = rig.world.resource::<OwnedTransients>();
    assert_eq!(
        owned.0[&fast].pose.vel,
        DVec3::new(2.0, 0.0, 0.0),
        "the 10 m/s debris is governed down to the realm ceiling (OQ-2: no subject kind is \
         exempt)",
    );
    assert_eq!(
        owned.0[&fast].pose.universe_tick,
        UniverseTick(100),
        "clamped AND advanced — the governor never freezes a subject",
    );
    assert_eq!(
        owned.0[&slow].pose.vel,
        DVec3::new(0.0, 1.5, 0.0),
        "a sub-ceiling transient is untouched bit-for-bit",
    );
}

#[test]
fn occupant_movement_dilates_with_the_realm_time_multiplier() {
    // A realm's SUBJECTIVE time factor scales OCCUPANT movement: a slow-time realm (0.5) advances the
    // dot HALF as far per tick; the default (1.0) is byte-identical to the un-scaled `speed·dt` step.
    let step_len_at = |mult: f64| {
        let mut rig = Rig::with_config(StubConfig {
            time_multiplier: mult,
            ..config()
        });
        rig.grant_realm();
        let _ = rig.attach();
        // Pure forward input (movement = [forward, strafe, up]).
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        total_m(&rig.world.resource::<Dots>().0[&SESSION].pose.pos).length()
    };
    // Default 1.0 = the un-multiplied step (`move_speed 2.0 · dt 0.05` = 0.1) — byte-identical.
    assert!((step_len_at(1.0) - 2.0 * 0.05).abs() < 1e-12);
    // The 0.5-multiplier realm moved EXACTLY half as far (time dilation).
    assert!((step_len_at(0.5) - step_len_at(1.0) * 0.5).abs() < 1e-12);
}

#[test]
fn every_discard_reason_is_logged() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);

    // DuplicateSeq: same seq again.
    let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    // StaleFence: fence below the session's.
    let _ = rig.tick(vec![input_msg(
        6,
        Fence::GENESIS,
        [1.0, 0.0, 0.0],
        [0.0, 0.0],
    )]);
    // MalformedInput: undecodable payload.
    let bad = GatewayToShard::SessionInput {
        session: SESSION,
        fence: Fence(1),
        input_bytes: vec![0xFF],
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &bad)]);
    // UnknownSession.
    let unknown = GatewayToShard::SessionInput {
        session: SessionId(0xBB),
        fence: Fence(1),
        input_bytes: vec![],
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &unknown)]);
    // PendingAuthority: input for a provisional (ungranted) dot.
    let _ = rig.attach_request(SessionId(0xCC), GATEWAY);
    let pending = GatewayToShard::SessionInput {
        session: SessionId(0xCC),
        fence: Fence(1),
        input_bytes: vec![],
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &pending)]);
    // NonFiniteInput: a forged NaN component is caught by the finite gate.
    let _ = rig.tick(vec![input_msg(
        6,
        Fence(1),
        [f32::NAN, 0.0, 0.0],
        [0.0, 0.0],
    )]);

    let log = rig.world.resource::<InputLog>();
    assert_eq!(log.applied(), vec![(SESSION, 5)]);
    assert_eq!(
        log.discarded(),
        vec![
            (SESSION, Some(5), DiscardReason::DuplicateSeq),
            (SESSION, None, DiscardReason::StaleFence),
            (SESSION, None, DiscardReason::MalformedInput),
            (SessionId(0xBB), None, DiscardReason::UnknownSession),
            (SessionId(0xCC), None, DiscardReason::PendingAuthority),
            (SESSION, Some(6), DiscardReason::NonFiniteInput),
        ]
    );
}

#[test]
fn non_finite_input_is_discarded_and_never_poisons_the_pose() {
    // ROB-1 (whole-codebase audit): a forged/corrupt NaN or Inf input must NEVER
    // integrate — NaN sticks in the authoritative pose forever and fans out to every
    // observer. The finite gate discards + counts it, the pose stays untouched, and
    // the seq does NOT advance (the input was never applied), so a subsequent FINITE
    // datagram at the same seq applies normally — the session is not wedged.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let _ = rig.tick(vec![input_msg(
        1,
        Fence(1),
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 0.0],
    )]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        total_m(&dot.pose.pos),
        vd_core::glam::DVec3::ZERO,
        "pose untouched"
    );
    // Split asserts (no `&&` short-circuit branch — the HR5 coverage discipline).
    assert_eq!(dot.yaw, 0.0, "yaw untouched");
    assert_eq!(dot.pitch, 0.0, "pitch untouched");

    // The same seq, now finite: applies (the poisoned datagram never consumed it).
    let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    let log = rig.world.resource::<InputLog>();
    assert_eq!(log.applied(), vec![(SESSION, 1)]);
    assert_eq!(
        log.discarded(),
        vec![(SESSION, Some(1), DiscardReason::NonFiniteInput)]
    );
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert!(
        total_m(&dot.pose.pos).is_finite(),
        "authoritative pose finite"
    );
    assert!(
        total_m(&dot.pose.pos).z < 0.0,
        "the finite input integrated (moved -Z)"
    );
}

#[test]
fn action_bits_are_inert_two_datagrams_differing_only_in_action_bits_integrate_identically() {
    // D-41 / D-39.1 MECHANICAL-GUARD TRIPWIRE (exists-to-be-flipped). `action_bits` is INERT in the
    // current integrator — `integrate` reads ONLY `look` + `movement`, never `action_bits` — so two
    // inputs identical EXCEPT for `action_bits` MUST integrate to the identical authoritative pose
    // today. This flips RED the day the reliable client→shard discrete-action arm makes the sim
    // consume `action_bits` (its named consumers: P6 block-edit-forward + P11 PvP fire-registration,
    // DEFERRED D-39.1) — a hard guard that world-mutating actions (esp. PvP fire-reg) can NEVER be
    // silently gated onto the lossy UNRELIABLE input datagram that `action_bits` rides.
    let mut inert = Rig::new();
    inert.grant_realm();
    let _ = inert.attach();
    let mut set = Rig::new();
    set.grant_realm();
    let _ = set.attach();

    // A non-trivial input (movement + look both non-zero) so the pose actually MOVES — proving the
    // two agree on a REAL integration, not on a shared do-nothing origin.
    let movement = [1.0f32, 0.5, -0.25];
    let look = [0.3f32, 0.1];
    let _ = inert.tick(vec![input_msg_bits(1, Fence(1), movement, look, 0)]);
    let _ = set.tick(vec![input_msg_bits(1, Fence(1), movement, look, u32::MAX)]);

    let dot_inert = inert.world.resource::<Dots>().0[&SESSION];
    let dot_set = set.world.resource::<Dots>().0[&SESSION];
    // Dot is Copy + PartialEq (pose + yaw + pitch + vel): ONE equality assert is the strongest,
    // HR5-coverage-safe identical-pose check (no `matches!` false-arm, no `&&` short-circuit).
    assert_eq!(
        dot_inert, dot_set,
        "action_bits is inert: 0 vs u32::MAX must not change the integrated pose"
    );
}

#[test]
fn a_higher_input_fence_upgrades_the_session() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].session_fence,
        Fence(4)
    );
    assert_eq!(
        rig.world.resource::<InputLog>().applied(),
        vec![(SESSION, 1)]
    );
}

#[test]
fn detach_is_two_phase_release_via_the_directory() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    // Phase 1: the dot stays HELD (departing); the retry driver sends the
    // revoke on the following tick (and every tick until confirmed).
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert!(dot.departing, "held until the directory releases it");
    let sent = rig.tick(vec![]);
    let revokes: Vec<InterShardFlow> = sent
        .iter()
        .filter(|(to, _, _)| *to == ORCH)
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
        .collect();
    assert!(
        revokes.contains(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(entity),
            fence: Fence(1),
        })),
        "the entity revoke is on its way"
    );
    // Departing dots no longer consume input.
    let _ = rig.tick(vec![input_msg(9, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    assert_eq!(
        rig.world.resource::<InputLog>().discarded().last().copied(),
        Some((SESSION, None, DiscardReason::Departing))
    );
    // Phase 2: the headless entity head confirms the revoke — despawn + reply.
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: None,
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
    let confirms: Vec<ShardToGateway> = sent
        .iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
        .collect();
    assert_eq!(
        confirms,
        vec![ShardToGateway::SessionDetached { session: SESSION }]
    );
    // Unknown-session detach still confirms immediately (idempotent).
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    let confirm: ShardToGateway = postcard::from_bytes(&sent[0].2).expect("decode");
    assert_eq!(
        confirm,
        ShardToGateway::SessionDetached { session: SESSION }
    );
    assert_eq!(
        confirm.into_snapshot_bytes(),
        None,
        "only frames carry snapshot bytes"
    );
    // A provisional (ungranted) dot detaches immediately — no record exists.
    let _ = rig.attach_request(SessionId(0xDD), GATEWAY);
    let detach_pending = GatewayToShard::DetachSession {
        session: SessionId(0xDD),
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach_pending)]);
    assert!(
        !rig.world
            .resource::<Dots>()
            .0
            .contains_key(&SessionId(0xDD))
    );
    let confirms = sent
        .iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .count();
    assert_eq!(confirms, 1);
}

#[test]
fn window_control_registers_refreshes_and_closes_the_registry_idempotently() {
    // THE WINDOW LANE, Slice A (rebased from the Slice-0 fail-closed drop — the registry this
    // control lane was owed now EXISTS): an open REGISTERS under (opener, id); a duplicate
    // open of the same scope is the keep-alive (refreshes the TTL, counted apart); a re-used
    // id under a NEW scope replaces the window whole; a close removes it; a close of an
    // unknown id is a COUNTED no-op (the polite fast path racing the TTL backstop). Never
    // `undecodable`, never a reply, never a dot.
    let mut rig = Rig::new();
    rig.grant_realm();
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
        static_held: None,
    };
    let sent = rig.tick(vec![
        wire_msg(GATEWAY, MsgClass::Control, &open),
        wire_msg(GATEWAY, MsgClass::Control, &open), // the keep-alive re-assert
    ]);
    // A registered Occupants window on a region-less rig: the emitter runs (empty roster ⇒
    // an empty-rows WindowFrame each tick — the leaf's per-tick stamp) — nothing else.
    let control_replies = sent
        .iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .count();
    // ★ RE-BASED IN S10: a window open now DOES draw one reliable reply — the static roster, stated
    // once instead of riding every tick's frame. The point this guarded (a window open mints nothing
    // and starts no session) is asserted directly below and is untouched.
    assert_eq!(
        control_replies, 1,
        "exactly one Control reply from a window open: the static roster, sent once"
    );
    assert_eq!(rig.world.resource::<Dots>().0.len(), 0, "no dot minted");
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.windows_opened, 1, "one window registered");
        assert_eq!(stats.window_reasserted, 1, "the duplicate refreshed it");
        assert_eq!(stats.windows_open, 1, "the gauge reads the registry");
        assert_eq!(stats.undecodable, 0, "window control is NOT garbage");
    }
    let held = rig.world.resource::<OpenWindows>();
    assert_eq!(held.0.len(), 1);
    assert_eq!(
        held.0.get(&(GATEWAY, WindowId(1))).map(|w| w.scope),
        Some(WindowScope::Occupants)
    );
    // A re-used id under a NEW scope replaces the window (fresh baselines, counted as opened).
    let rescope = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Child(RealmId::Planet(9)),
        static_held: None,
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &rescope)]);
    assert_eq!(rig.world.resource::<StubStats>().windows_opened, 2);
    assert_eq!(
        rig.world
            .resource::<OpenWindows>()
            .0
            .get(&(GATEWAY, WindowId(1)))
            .map(|w| w.scope),
        Some(WindowScope::Child(RealmId::Planet(9)))
    );
    // Close removes it; a second close of the now-unknown id is a counted no-op.
    let close = GatewayToShard::WindowClose {
        window: WindowId(1),
    };
    let _ = rig.tick(vec![
        wire_msg(GATEWAY, MsgClass::Control, &close),
        wire_msg(GATEWAY, MsgClass::Control, &close),
    ]);
    assert_eq!(rig.world.resource::<OpenWindows>().0.len(), 0);
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.window_close_unknown, 1, "the second close counted");
        assert_eq!(stats.windows_open, 0, "zero subscribers ⇒ zero windows");
    }
}

#[test]
fn stale_fence_detach_is_discarded_with_reason() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // Upgrade the session fence, then detach with the old one.
    let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
    let stale = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &stale)]);
    assert_eq!(rig.world.resource::<Dots>().0.len(), 1, "dot survives");
    let log = rig.world.resource::<InputLog>();
    assert_eq!(
        log.discarded().last().copied(),
        Some((SESSION, None, DiscardReason::StaleFence))
    );
}

#[test]
fn frames_carry_all_dots_and_count_monotonically() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // A second session through the same gateway (full grant flow).
    let _ = rig.attach_request(SessionId(0xBB), GATEWAY);
    let _ = rig.confirm_entity_grant(SessionId(0xBB));
    let sent = rig.tick(vec![]);
    let frames = decode_frames(&sent);
    assert_eq!(frames.len(), 1, "one gateway, one frame");
    let snap = &frames[0];
    assert_eq!(snap.sub, SubId(0));
    assert_eq!(snap.entities.len(), 2, "both dots present");
    assert_eq!(snap.source_tick, vd_core::TickId(1));
    assert_eq!(snap.universe_tick, UniverseTick(100));
    // Frame ids increment.
    let next = decode_frames(&rig.tick(vec![]));
    assert_eq!(next[0].frame_id, snap.frame_id + 1);
}

#[test]
fn a_large_world_partitions_into_multiple_under_budget_frames() {
    // GW-1: many dots exceed the datagram budget, so the snapshot ships as
    // several same-frame_id sibling chunks, each encoding under the budget,
    // together carrying EVERY entity (no silent MTU drop).
    let mut rig = Rig::with_config(StubConfig {
        snapshot_datagram_budget: 300,
        ..config()
    });
    rig.grant_realm();
    // Insert 12 granted dots directly (bypassing the attach handshake).
    {
        let mut dots = rig.world.resource_mut::<Dots>();
        for n in 0..12u64 {
            dots.0.insert(
                SessionId(u128::from(n) + 1),
                Dot {
                    entity: EntityId::pack(EntityKind::Player, 10, n, n as u32),
                    account: AccountId(n as u128),
                    session_fence: Fence(1),
                    gateway: GATEWAY,
                    granted: true,
                    input_active: false,
                    adopting: false,
                    authority: Authority::Owned { fence: Fence(1) },
                    departing: false,
                    entity_fence: Fence(1),
                    pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100)),
                    yaw: 0.0,
                    pitch: 0.0,
                    last_applied_seq: None,
                    prev_offset: LatticePos::ORIGIN,
                },
            );
        }
    }
    let frames = decode_frames(&rig.tick(vec![]));
    assert!(
        frames.len() > 1,
        "12 dots must partition into multiple frames"
    );
    // Every chunk is under budget and the union is all 12 entities.
    let mut all_entities = std::collections::BTreeSet::new();
    for f in &frames {
        let encoded = postcard::to_allocvec(f).expect("encode").len();
        assert!(encoded <= 300, "chunk encodes to {encoded} > 300");
        for e in &f.entities {
            all_entities.insert(e.entity);
        }
    }
    assert_eq!(all_entities.len(), 12, "no entity lost across chunks");
    // §6.3: every chunk of ONE tick shares the SAME frame_id (each self-contained
    // latest-wins) so a reordered sibling chunk is never dropped as stale.
    let ids: std::collections::BTreeSet<u64> = frames.iter().map(|f| f.frame_id).collect();
    assert_eq!(ids.len(), 1, "all chunks of one tick share a frame_id");
    let tick0_id = *ids.iter().next().expect("at least one chunk");
    // The next tick's chunks all share a STRICTLY GREATER frame_id: the counter
    // advances exactly once per tick (monotonic between ticks, stable within one).
    let next = decode_frames(&rig.tick(vec![]));
    assert!(next.len() > 1, "still partitioned the next tick");
    let next_ids: std::collections::BTreeSet<u64> = next.iter().map(|f| f.frame_id).collect();
    assert_eq!(
        next_ids.len(),
        1,
        "next tick's chunks also share one frame_id"
    );
    assert_eq!(
        *next_ids.iter().next().expect("chunk"),
        tick0_id + 1,
        "frame_id advances exactly once per tick"
    );
}

#[test]
fn two_gateways_each_get_the_frame() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let other_gateway = NodeId(21);
    let _ = rig.attach_request(SessionId(0xBB), other_gateway);
    let _ = rig.confirm_entity_grant(SessionId(0xBB));
    let sent = rig.tick(vec![]);
    let snapshot_targets: Vec<NodeId> = sent
        .iter()
        .filter(|(_, class, _)| *class == MsgClass::Snapshot)
        .map(|(to, _, _)| *to)
        .collect();
    assert_eq!(snapshot_targets, vec![GATEWAY, other_gateway]);
}

#[test]
fn minted_entities_are_unique_and_structured() {
    let mut mint = EntityMint {
        seq: 0,
        rng: SplitMix64::new(1),
    };
    let a = mint_entity(&mut mint, SHARD);
    let b = mint_entity(&mut mint, SHARD);
    assert_ne!(a, b);
    assert_eq!(a.seq(), 0);
    assert_eq!(b.seq(), 1);
    assert_eq!(a.kind_tag(), EntityKind::Player as u8);
    assert_eq!(a.mint_shard(), 10);
}

#[test]
fn input_log_is_a_bounded_window_with_exact_totals() {
    // SCALE-3: a small window holds only the NEWEST entries (no unbounded
    // growth), while the totals are EXACT and evictions are counted.
    let mut log = InputLog::new(3);
    for seq in 0..10u64 {
        log.record_applied(SessionId(1), seq);
    }
    for seq in 0..4u64 {
        log.record_discarded(SessionId(2), Some(seq), DiscardReason::DuplicateSeq);
    }
    // Window holds the last 3 of each; totals count everything.
    assert_eq!(log.applied().len(), 3);
    assert_eq!(
        log.applied(),
        vec![(SessionId(1), 7), (SessionId(1), 8), (SessionId(1), 9)]
    );
    assert_eq!(log.discarded().len(), 3);
    assert_eq!(log.applied_total, 10);
    assert_eq!(log.discarded_total, 4);
    // 7 applied + 1 discarded evicted from the windows.
    assert_eq!(log.window_evictions, 8);
    // Capacity floors at 1.
    let mut tiny = InputLog::new(0);
    tiny.record_applied(SessionId(9), 1);
    tiny.record_applied(SessionId(9), 2);
    assert_eq!(tiny.applied(), vec![(SessionId(9), 2)]);
}
