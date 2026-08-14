//! THE flight helpers every process gate flies (Tier-B, `dev-control`-gated): ONE pilot for every
//! cluster test, so a navigation lesson learned once (the sticky throttle, the feedback-lag brake,
//! the rendezvous-and-park) is learned everywhere. Promoted from the demand suite
//! (`rlm_demand_login.rs`), where the rendezvous was proven green at FULL orbit speed; the walk
//! gate and the demand suite now fly the identical code.
//!
//! The flight law (the clusters plan §2.3): every static-cluster leg flies the ±Z POLAR corridor —
//! orbits lie near the XY plane and the star ring near XZ, so ±Z is clear of both by the I-AXIS /
//! I-POLE / I-RADIAL margins `world_roster` asserts. A leg's outcome is always a REALM LABEL, never
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
    // speed is then the planet's own ~6 m/s, the decision is still true at the flush, and the
    // crossing commits.
    const RENDEZVOUS_TICKS: u64 = 200;
    // One full sweep of the planet's shell (~8.3 m at ~6.3 m/s ≈ 66 ticks) plus pipeline margin.
    const SWEEP_GRACE_TICKS: u64 = 120;
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
             (loc {loc:?}); closest approach {best:.2} m vs ~4.16 m SOI.",
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
