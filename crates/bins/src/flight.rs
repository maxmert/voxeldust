//! THE flight helpers every process gate flies (Tier-B, `dev-control`-gated): ONE pilot for every
//! cluster test, so a navigation lesson learned once (the sticky throttle, the feedback-lag brake,
//! the rendezvous-and-park) is learned everywhere. Promoted from the demand suite
//! (`rlm_demand_login.rs`), where the rendezvous was proven green at FULL orbit speed; the walk
//! gate and the demand suite now fly the identical code.
//!
//! The flight law (the clusters plan §2.3): every static-cluster leg flies the ±Z POLAR corridor —
//! orbits lie near the XY plane and the star ring near XZ, so ±Z is clear of both by the I-AXIS /
//! I-POLE / I-RADIAL margins `world_roster` asserts — for the HOME system AND the ring SIBLING the
//! chain gate creeps into (per-system asserts; the sibling's orbits are their own seed draw, so the
//! home's margins never licensed them — batch review). A leg's outcome is always a REALM LABEL, never
//! a coordinate: past a crossing the pose reframes, and a target stated in the old frame is
//! meaningless (the demand suite's own run-2 lesson).

use std::time::{Duration, Instant};

use vd_core::glam::DVec3;
use vd_devproto::{DevRequest, DevResponse, DevState};

/// One dev-control round-trip, `None` on a blink (the callers' poll loops tolerate that).
fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    crate::dev_roundtrip(port, request).ok()
}

/// The current delivered state, `None` on a blink.
fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

/// The OWN entity's delivered world pose, when the row is delivered.
fn own_pos(state: &DevState) -> Option<DVec3> {
    let own = state.own_entity.as_deref()?;
    state
        .entities
        .iter()
        .find(|r| r.entity == own)
        .map(|r| DVec3::from_array(r.pos))
}

/// THE ONE FLY-IN (Stage-C: the orbit-slowdown knob is DELETED — SL5 forbids a scale knob, and a
/// fixture that only passes on a slowed world cannot gate a game played on a moving one). Every
/// crossing gate flies the SAME rendezvous-and-park pattern the flight gate proved at full orbit
/// speed: aim at where the planet WILL BE, park inside the acquire edge with the throttle cut, let
/// the planet sweep over the stationary ship, re-plan until `location` flips.
///
/// # Panics
/// When the deadline passes without the flip (with the closest approach measured), or dev-control
/// stays unreachable.
pub fn rendezvous_into_planet(
    devctl_port: u16,
    p: &crate::DevClusterParams,
    planet: vd_core::pose::RealmId,
    elements: &vd_physics::celestial::OrbitalElements,
    deadline: Duration,
) {
    let planet_seed = match planet {
        vd_core::pose::RealmId::Planet(s) => s,
        other => panic!("inner mover is a planet, got {other:?}"),
    };
    let tick_hz = 1.0 / p.tick_dt;
    let planet_at = |tick: u64| {
        vd_physics::celestial::orbital_state(
            elements,
            vd_core::kinematics::secs_since_epoch(tick, tick_hz),
        )
        .position
    };
    // RENDEZVOUS AND PARK (Stage B4; run-3/run-4 measurements). Chasing the live centre — even with
    // a led aim — keeps the ship at chase speed through the band, and the flush re-validation then
    // RIGHTLY refuses a crossing whose subject is already gone. A pilot lands by ARRIVING EARLY:
    // fly to where the planet WILL BE, stop, and let it sweep over the parked ship — the relative
    // speed is then the planet's own orbital ~m/s, the decision is still true at the flush, and the
    // crossing commits.
    const RENDEZVOUS_TICKS: u64 = 200;
    // One full sweep of the planet's shell — its diameter at the planet's own orbital speed, about
    // a second on THE world — plus generous pipeline margin. A budget with slack, deliberately not
    // a derived world number: the deadline loop re-plans anyway, so slack costs one extra leg at
    // worst. (The old "~8.3 m at ~6.3 m/s" arithmetic here was the pre-S4 world's — batch review.)
    const SWEEP_GRACE_TICKS: u64 = 120;
    // THE world's own SOI + acquire edge, derived at use for the failure diagnostics — never a
    // transcribed literal (batch review: this message still said "~4.16 m SOI" after the S4
    // re-solve moved the planet SOI to ~3.95 m, mislabelling a 4.05 m closest approach as inside;
    // and the SOI face was the wrong quantity anyway — containment ACQUIRES at soi − inset).
    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let soi_m = config.planet.planet_soi_r_m;
    let acquire_edge_m = soi_m - config.band.inset_m;
    let want = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();
    let started = Instant::now();
    let mut best = f64::INFINITY;
    let mut loc = String::new();
    let mut leg = 0u32;
    loop {
        let st = {
            let mut got = None;
            for _ in 0..20 {
                if let Some(s) = poll_state(devctl_port) {
                    got = Some(s);
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            match got {
                Some(s) => s,
                None => {
                    assert!(
                        started.elapsed() < deadline,
                        "dev-control unreachable (best {best:.2} m, loc {loc:?})"
                    );
                    continue;
                }
            }
        };
        loc = st.location.clone().unwrap_or_default();
        if loc == want {
            break;
        }
        // FLY TO THE RENDEZVOUS — where the planet will be `RENDEZVOUS_TICKS` from the client's last
        // report — then PARK there and let the planet sweep over the stationary ship.
        let now_tick = st.universe_tick.unwrap_or(0);
        let target = planet_at(now_tick + RENDEZVOUS_TICKS);
        eprintln!(
            "[fly-in] leg {leg}: tick={now_tick} loc={loc:?} own_len={:?} best={best:.2} — \
             rendezvous at tick {}",
            own_pos(&st).map(|v| v.length()),
            now_tick + RENDEZVOUS_TICKS,
        );
        leg += 1;
        let flew = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: 0.5,
                max_ticks: RENDEZVOUS_TICKS,
                // THE BRAKE (nav::walk_to), sized to the FEEDBACK LAG (run-7 measurement): the
                // controller steers by the DELIVERED pose (~2-3 ticks behind); (1 + lag) steps
                // commands at most dist/4 per tick — monotone, no overshoot.
                max_step_m: 4.0 * p.move_speed * p.tick_dt,
            },
        );
        // CUT THE THROTTLE before parking (run-6): Move input is STICKY — a held throttle during a
        // park is a full-speed straight-line runaway.
        let _ = devctl(
            devctl_port,
            &DevRequest::Move {
                axes: [0.0, 0.0, 0.0],
            },
        );
        eprintln!(
            "[fly-in]   leg {}: walk outcome {}",
            leg - 1,
            match &flew {
                Some(DevResponse::State { .. }) => "ARRIVED",
                Some(DevResponse::Timeout { .. }) => "TIMEOUT",
                other => {
                    let _ = other;
                    "OTHER"
                }
            },
        );
        // PARKED: hold until the sweep instant (plus grace) or the flip, whichever first.
        let wait_until = now_tick + RENDEZVOUS_TICKS + SWEEP_GRACE_TICKS;
        loop {
            std::thread::sleep(Duration::from_millis(200));
            let Some(s) = poll_state(devctl_port) else {
                break;
            };
            loc = s.location.clone().unwrap_or_default();
            if let Some(pos) = own_pos(&s) {
                best = best.min((pos - planet_at(s.universe_tick.unwrap_or(0))).length());
            }
            if loc == want || s.universe_tick.unwrap_or(0) > wait_until {
                break;
            }
        }
        if loc == want {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "NO CROSSING: flew rendezvous legs at {planet:?} for {}s but location never flipped \
             (loc {loc:?}); closest approach {best:.2} m vs the {soi_m:.2} m SOI (containment \
             acquires at {acquire_edge_m:.2} m).",
            started.elapsed().as_secs(),
        );
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[fly-in] location flipped to {loc:?} (closest approach {best:.2} m)");
    assert_eq!(loc, want);
}

/// One WalkTo's tick budget inside [`cross_leg`] — bounded so the loop re-reads the label often
/// (a target is only re-stated while the frame it was stated in still holds).
const CROSS_LEG_WALK_TICKS: u64 = 400;
/// The WATCHED variant's walk chunk (`cross_leg_watching_scene`): small enough that the
/// between-chunk polls bracket the epoch swap tightly (the no-flicker gate compares the LAST
/// old-epoch state against the FIRST new-epoch state — see the chunking note at the call).
const WATCHED_WALK_TICKS: u64 = 25;
/// The PARKED wait after an ARRIVED walk: the aim point sits where the crossing fires on a
/// stationary avatar, and detector cooldown (~75 ticks) + the saga + the client's delivered flip
/// all fit well inside this. MEASURED lesson (the first chain run): a 2 s grace expired BEFORE the
/// commit, the re-issued WalkTo then read the old-frame target in the NEW frame and drove the dot
/// straight back out — a sibling↔galaxy limit cycle burning one directory fence per ~0.7 s.
const CROSS_LEG_COMMIT_WAIT: Duration = Duration::from_secs(10);
const CROSS_LEG_POLL: Duration = Duration::from_millis(150);

/// Drive ONE crossing leg: WalkTo `aim(universe_tick)` (a target stated in the space the session
/// CURRENTLY stands in), park, and repeat until the delivered `location` label reads `want`. It
/// ASSERTS THE REALM LABEL REACHED, never a coordinate: at a crossing the pose reframes and a
/// target stated in the old frame is meaningless. `aim` takes the latest universe tick so a moving
/// aim point (a planet) re-plans per pass; a fixed waypoint ignores it.
///
/// THE PARK DISCIPLINE (shared with the rendezvous): the crossing must commit on a STATIONARY
/// avatar. Each walk burst ends with the throttle cut, and an ARRIVED burst is followed by a parked
/// [`CROSS_LEG_COMMIT_WAIT`] poll — a dot that has REACHED its aim is never walked again while the
/// commit is in flight, because re-planning across the commit instant re-reads the old-frame target
/// in the new frame and drives the dot back out (the run-2 trap, re-measured on the first chain run
/// as a fence-burning flap).
///
/// # Panics
/// When `deadline` passes without the label flipping to `want`.
pub fn cross_leg(
    devctl_port: u16,
    leg: &str,
    aim: impl Fn(u64) -> DVec3,
    want: &str,
    deadline: Duration,
) {
    let started = Instant::now();
    let mut loc = String::new();
    loop {
        if let Some(s) = poll_state(devctl_port) {
            loc = s.location.clone().unwrap_or_default();
            if loc == want {
                break;
            }
            let target = aim(s.universe_tick.unwrap_or(0));
            let walked = devctl(
                devctl_port,
                &DevRequest::WalkTo {
                    target: target.to_array(),
                    arrive_epsilon: 2.0,
                    max_ticks: CROSS_LEG_WALK_TICKS,
                    // The feedback-lag brake (see rendezvous_into_planet's knob).
                    max_step_m: 4.0 * crate::DEV.move_speed * crate::DEV.tick_dt,
                },
            );
            // CUT THE THROTTLE (sticky Move) — the flip lands on a stationary avatar.
            let _ = devctl(
                devctl_port,
                &DevRequest::Move {
                    axes: [0.0, 0.0, 0.0],
                },
            );
            if matches!(walked, Some(DevResponse::State { .. })) {
                // ARRIVED at the aim: stay PARKED and wait out the commit + the delivered flip.
                let wait_until = Instant::now() + CROSS_LEG_COMMIT_WAIT;
                while Instant::now() < wait_until && loc != want {
                    if let Some(g) = poll_state(devctl_port) {
                        loc = g.location.clone().unwrap_or_default();
                    }
                    std::thread::sleep(CROSS_LEG_POLL);
                }
            }
            if loc == want {
                break;
            }
        }
        assert!(
            started.elapsed() < deadline,
            "leg {leg}: the location label never flipped to {want:?} (last {loc:?}) — the leg \
             asserts the REALM LABEL reached, never a coordinate",
        );
        std::thread::sleep(CROSS_LEG_POLL);
    }
    // Settle: the leg ends parked, with the label asserted.
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[cross-leg] {leg}: label flipped to {want:?}");
    assert_eq!(loc, want, "leg {leg}: the reached realm label");
}

/// THE WATCHED CROSSING (Slice C1 — the §2.11 no-flicker gate's substrate): fly [`cross_leg`]'s
/// exact rendezvous pattern while WATCHING the composed scene through the swap, and hand back the
/// LAST delivered state of the old scene epoch and the FIRST of the new one — the two samples the
/// no-flicker verdict compares (a persisting body's delta across the swap, bounded by true
/// motion). Asserts as it flies: the drawn scene is NEVER ABSENT on any poll (§2.7: the old scene
/// renders until the new level lands — an empty `realm_boxes` mid-crossing is the flicker), and
/// the origin epoch advances EXACTLY ONCE across the leg (a double bump is a flapping swap).
///
/// # Panics
/// On a deadline pass, an absent scene, a missing origin marker, or an epoch that moved by
/// anything but exactly one.
pub fn cross_leg_watching_scene(
    devctl_port: u16,
    leg: &str,
    aim: impl Fn(u64) -> DVec3,
    want: &str,
    deadline: Duration,
) -> (Option<DevState>, DevState, DevState) {
    let started = Instant::now();
    let mut loc = String::new();
    let mut start_epoch: Option<u64> = None;
    let mut before_prev: Option<DevState> = None;
    let mut before: Option<DevState> = None;
    let mut after: Option<DevState> = None;
    // PRIME the old-epoch pair BEFORE any walking: a short crossing (a park near the boundary,
    // a fast saga) can fit entirely inside the first walk chunk, leaving the watcher a single
    // pre-swap sample — and the no-flicker verdict needs TWO same-epoch samples to MEASURE each
    // body's own per-tick motion. Two parked polls a few ticks apart give every body in the old
    // scene a measured rate whatever the leg's timing does.
    for _ in 0..2 {
        if let Some(s) = poll_state(devctl_port) {
            watch_scene_sample(
                leg,
                &s,
                &mut start_epoch,
                &mut before_prev,
                &mut before,
                &mut after,
            );
            loc = s.location.clone().unwrap_or_default();
        }
        std::thread::sleep(CROSS_LEG_POLL);
    }
    loop {
        if let Some(s) = poll_state(devctl_port) {
            watch_scene_sample(
                leg,
                &s,
                &mut start_epoch,
                &mut before_prev,
                &mut before,
                &mut after,
            );
            loc = s.location.clone().unwrap_or_default();
            if loc == want {
                break;
            }
            let target = aim(s.universe_tick.unwrap_or(0));
            let walked = devctl(
                devctl_port,
                &DevRequest::WalkTo {
                    target: target.to_array(),
                    arrive_epsilon: 2.0,
                    // SMALL chunks, deliberately (§2.11): a WalkTo BLOCKS until it arrives or
                    // its tick budget lapses, and one [`CROSS_LEG_WALK_TICKS`]-sized chunk
                    // swallows the entire crossing — the watcher would then hold NO delivered
                    // state near the swap, and the no-flicker verdict would compare states
                    // hundreds of ticks apart (measured: 509 ticks, a 49 m lawful orbital
                    // sweep misread as a swap teleport). Chunking keeps the samples within
                    // [`WATCHED_WALK_TICKS`] of the swap on both sides.
                    max_ticks: WATCHED_WALK_TICKS,
                    max_step_m: 4.0 * crate::DEV.move_speed * crate::DEV.tick_dt,
                },
            );
            let _ = devctl(
                devctl_port,
                &DevRequest::Move {
                    axes: [0.0, 0.0, 0.0],
                },
            );
            if matches!(walked, Some(DevResponse::State { .. })) {
                let wait_until = Instant::now() + CROSS_LEG_COMMIT_WAIT;
                while Instant::now() < wait_until && loc != want {
                    if let Some(g) = poll_state(devctl_port) {
                        watch_scene_sample(
                            leg,
                            &g,
                            &mut start_epoch,
                            &mut before_prev,
                            &mut before,
                            &mut after,
                        );
                        loc = g.location.clone().unwrap_or_default();
                    }
                    std::thread::sleep(CROSS_LEG_POLL);
                }
            }
            if loc == want {
                break;
            }
        }
        assert!(
            started.elapsed() < deadline,
            "leg {leg}: the location label never flipped to {want:?} (last {loc:?})",
        );
        std::thread::sleep(CROSS_LEG_POLL);
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    // The swap may land after the label flip's poll; wait until the NEW epoch is seen.
    let swap_deadline = Instant::now() + CROSS_LEG_COMMIT_WAIT;
    while after.is_none() {
        if let Some(s) = poll_state(devctl_port) {
            watch_scene_sample(
                leg,
                &s,
                &mut start_epoch,
                &mut before_prev,
                &mut before,
                &mut after,
            );
        }
        assert!(
            Instant::now() < swap_deadline,
            "leg {leg}: the label flipped but the new scene epoch never delivered"
        );
        std::thread::sleep(CROSS_LEG_POLL);
    }
    eprintln!("[cross-leg] {leg}: label flipped to {want:?}, epoch bumped once");
    (
        before_prev,
        before.unwrap_or_else(|| panic!("leg {leg}: no old-epoch state was ever delivered")),
        after.expect("the wait above ends only with the new epoch seen"),
    )
}

/// One watched sample of the crossing (the §2.11 no-flicker gate's per-poll asserts + the
/// before/after bookkeeping) — a plain fn so the two poll sites cannot drift.
fn watch_scene_sample(
    leg: &str,
    s: &DevState,
    start_epoch: &mut Option<u64>,
    before_prev: &mut Option<DevState>,
    before: &mut Option<DevState>,
    after: &mut Option<DevState>,
) {
    assert!(
        !s.realm_boxes.is_empty(),
        "leg {leg}: the drawn scene went ABSENT mid-crossing — the old scene must render \
         until the new level lands (§2.7), got {s:?}"
    );
    let (_, epoch) = s
        .origin
        .clone()
        .unwrap_or_else(|| panic!("leg {leg}: no origin marker on a delivered state: {s:?}"));
    let start = *start_epoch.get_or_insert(epoch);
    if epoch == start {
        // Keep the last TWO old-epoch samples: the pair measures each body's own per-tick
        // motion, which is what bounds a MOVING persister across the swap (the no-flicker
        // gate's derived allowance — an orbiting planet lawfully sweeps between two polls).
        *before_prev = before.take();
        *before = Some(s.clone());
    } else {
        assert_eq!(
            epoch,
            start + 1,
            "leg {leg}: the origin epoch must bump EXACTLY once across one crossing"
        );
        if after.is_none() {
            *after = Some(s.clone());
        }
    }
}

/// The creep throttle (an axes MAGNITUDE — the axes ride the wire as a throttle): 0.05 of the
/// 500 m/s dev speed = 25 m/s. Slow enough that the drift between the commit and the next label
/// poll is metres, fast enough that an acquire-edge approach is seconds.
const CREEP_AXES_MAG: f32 = 0.05;
const CREEP_POLL: Duration = Duration::from_millis(150);

/// Drive a crossing INTO a realm whose centre is NOT the session origin — held-axes CREEP, no
/// coordinate in flight. MEASURED lesson (the chain gate's leg E, both diagnostic runs): a WalkTo
/// aimed at `sibling_centre + (0,0,−140)` STRADDLES the commit — the instant authority re-homes,
/// the absolute target re-reads in the DEST's frame as a point ~12 km away, the still-running walk
/// drives the dot straight back out of the shell, and the two shards ping-pong it (one directory
/// fence per ~0.7 s, `sibling↔galaxy`, 80+ sagas). No re-statement of the aim can fix that while a
/// walk is IN FLIGHT across the commit; the only target-free drive is a held throttle. So: hold
/// `axes` (small — [`CREEP_AXES_MAG`]-scaled by the caller) from a parked standoff OUTSIDE the
/// acquire edge, poll the label, and CUT the throttle the moment it reads `want` — the post-commit
/// drift is bounded by one poll of creep (~4 m), leaving the dot resting INSIDE the dest.
///
/// # Panics
/// When `deadline` passes without the label flipping to `want`.
pub fn creep_into(devctl_port: u16, leg: &str, axes: [f32; 3], want: &str, deadline: Duration) {
    let started = Instant::now();
    let mut loc = String::new();
    loop {
        // Re-send every poll: Move is sticky server-side, but a re-send is idempotent and covers a
        // dropped input datagram (the creep must not silently stall short of the edge).
        let _ = devctl(devctl_port, &DevRequest::Move { axes });
        if let Some(s) = poll_state(devctl_port) {
            loc = s.location.clone().unwrap_or_default();
            if loc == want {
                break;
            }
        }
        assert!(
            started.elapsed() < deadline,
            "leg {leg}: crept on held axes {axes:?} but the location label never flipped to \
             {want:?} (last {loc:?})",
        );
        std::thread::sleep(CREEP_POLL);
    }
    // THE CUT: the flip is observed — stop dead. The drift since the commit is one poll of creep.
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[creep-into] {leg}: label flipped to {want:?}");
    assert_eq!(loc, want, "leg {leg}: the reached realm label");
}

/// The +Z polar creep (toward a shell entered from BELOW the pole): movement `[1,0,0]` is world −Z
/// at identity orientation, so `[-CREEP_AXES_MAG,0,0]` is +Z at creep throttle.
#[must_use]
pub fn creep_axes_plus_z() -> [f32; 3] {
    [-CREEP_AXES_MAG, 0.0, 0.0]
}
