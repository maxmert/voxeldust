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

/// THE DELIVERED-POSE LAG BUDGET, seconds — how far behind the server's live pose the dev-control
/// state a gate steers by can lawfully sit: the client's own interpolation buffer (the render
/// cursor deliberately trails the freshest snapshot — the no-prediction mandate) plus four ticks
/// of delivery/poll slack (one snapshot interval at the 20 Hz cadence is 2–3 shard ticks; one
/// more for the gate's own poll). Every governed-flight brake below is sized from THIS, because
/// at governed speeds the lag is no longer a metre of error but kilometres.
#[must_use]
pub fn pose_lag_s(tick_dt_s: f64) -> f64 {
    vd_client::tuning::ClientInterpTuning::DEFAULT.interp_buffer_ms / 1000.0 + 4.0 * tick_dt_s
}

/// THE GOVERNED WALK BRAKE (the S3 re-derivation of `max_step_m`): the pre-law brake
/// (`4·move_speed·dt` = 40 m) assumed a flat 10 m/tick; a governed approach arrives at its
/// target's own ceiling `v`, and a walk steered by a delivered pose `pose_lag_s` behind the truth
/// overshoots by `lag·v` — a brake narrower than that is a measured LIMIT CYCLE (the ship crosses
/// the target inside one feedback interval and the controller saws forever). The brake must open
/// before the lag can carry the ship past it while it still closes at the ceiling:
/// `M > lag·(v + M/τ)` ⇒ `M = 2·lag·v / (1 − lag/τ)` — the tight bound with a 2× margin, every
/// term named (the `M/τ` arm is the governor's own ceiling growth across the brake zone).
///
/// # Panics
/// If `pose_lag_s ≥ τ` — a demand pipeline slower than the pose lag cannot brake a governed walk
/// at all, and refusing loudly beats a wandering gate.
#[must_use]
pub fn governed_brake_m(v_target_ceiling_mps: f64, tick_dt_s: f64, tau_s: f64) -> f64 {
    let lag = pose_lag_s(tick_dt_s);
    assert!(
        lag < tau_s,
        "pose lag {lag:.3} s >= tau {tau_s:.3} s — a governed walk cannot brake"
    );
    2.0 * lag * v_target_ceiling_mps / (1.0 - lag / tau_s)
}

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
/// THE ONE GOVERNED-LEG BUDGET (true scale): a leg's deadline is 3× the speed law's closed-form
/// time plus a fixed slack for boot/commit tails. WHY 3×, measured (walk gate, 2026-08-19): the
/// home-system exit's closed form is ~105 s and the flown leg measured ~260 s — the DEV client's
/// input cadence spreads the ramp's per-applied-input compounding over more wall clock than the
/// per-tick ideal, and the crossing's saga tail (latch + re-drives + the delivered label flip)
/// rides after the geometry. 2× left that leg ~25 s short of its own commit; 3× + slack covers
/// the measured worst with margin while still failing a genuinely frozen leg loudly.
#[must_use]
pub fn governed_leg_budget(p: &crate::DevClusterParams, dist_m: f64, cap_mps: f64) -> Duration {
    let tuning = vd_core::flight::FlightTuning::derive(
        p.move_speed,
        p.tick_dt,
        (u64::from(p.tick_hz) / 2).max(1),
        u32::try_from(p.boot_ticks_p99).expect("boot p99 fits"),
    );
    let leg_s = vd_core::flight::leg_time_s(dist_m, cap_mps, p.move_speed, cap_mps, tuning.tau_s)
        .unwrap_or_else(|| tuning.tau_s * (1.0 + dist_m / (tuning.tau_s * p.move_speed)).ln());
    Duration::from_secs(60) + Duration::from_secs_f64(3.0 * leg_s)
}

/// THE DERIVED PLANET-EXIT BUDGET: [`governed_leg_budget`] over the planet's own solved bound
/// under the planet's own ceiling — leaving a ~1.2e7 m solved SOI at ~1.3e5 m/s is ~100 s of
/// lawful flight before any slack (the interim suites bounded planet exits at 90 s).
#[must_use]
pub fn planet_exit_budget(p: &crate::DevClusterParams, planet: vd_core::pose::RealmId) -> Duration {
    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let shell = vd_physics::worldgen::realm_regions_for_config(p.universe_seed, &config)
        .iter()
        .find(|r| r.realm == planet)
        .map(|r| r.shape.finite_extent())
        .expect("the exited planet is rostered on THE world");
    let cap =
        vd_core::flight::realm_speed_cap_mps(shell, p.move_speed, vd_core::flight::TRAVERSE_S);
    governed_leg_budget(p, shell + config.band.outset_m, cap)
}

pub fn rendezvous_into_planet(
    devctl_port: u16,
    p: &crate::DevClusterParams,
    planet: vd_core::pose::RealmId,
    elements: &vd_physics::celestial::OrbitalElements,
    deadline: Duration,
) {
    assert!(
        try_rendezvous_into_planet(devctl_port, p, planet, elements, deadline),
        "NO CROSSING: the chase never reached {planet:?} within {deadline:?}",
    );
}

/// [`rendezvous_into_planet`] that REPORTS instead of panicking — `true` when the label flipped,
/// `false` when the budget ran out with the client still live. The distinction exists for the
/// repeat-round-trip gate, whose subject is the FREEZE (does a crossing wedge the server?), not
/// the harness's ability to thread a moving 1.2e7 m shell on every attempt: at true scale the
/// intercept is genuinely hard from some starting geometries (see the chase notes below), and a
/// steering shortfall must never be reported as a product freeze. Every other caller wants the
/// crossing itself and uses the panicking form.
///
/// # Panics
/// If `planet` is not a [`vd_core::pose::RealmId::Planet`], or dev-control stays unreachable past
/// the deadline.
#[must_use]
pub fn try_rendezvous_into_planet(
    devctl_port: u16,
    p: &crate::DevClusterParams,
    planet: vd_core::pose::RealmId,
    elements: &vd_physics::celestial::OrbitalElements,
    deadline: Duration,
) -> bool {
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
    // THE LEAD-AIM CHASE UNDER THE GOVERNOR (true-scale restatement of the Stage-B4 park-and-
    // sweep). The park-and-sweep flew 200-tick legs each ENDING IN A THROTTLE CUT — and under the
    // speed law a cut zeroes the carried velocity, so every leg RESTARTED THE RAMP FROM THE FOOT
    // (measured: the fly-in never exceeded ~1e3 m/s against a 1.48e8 m/s governed ceiling, and a
    // ~2e9 m journey closed at walking pace). At true scale the sweep premise is dead anyway: the
    // inner planet's period is ~1.7 days, so "arrive early and let it sweep" waits days.
    //
    // The lawful chase: HOLD THE THROTTLE CONTINUOUSLY (the ramp compounds across re-aims —
    // Move input is sticky, so the throttle survives the poll gaps between chunks) and RE-AIM
    // each chunk at where the planet will be one chunk ahead. Pure pursuit converges because the
    // governed cruise (≥1e6 m/s mid-course) dwarfs the planet's own orbital ~8e4 m/s; the aim
    // staleness per chunk is the planet's one-chunk displacement (~1.6e5 m), two orders inside
    // the ~1.2e7 m acquire edge. ARRIVAL IS THE GOVERNOR'S, not a brake: the approach arm lowers
    // the ceiling onto the planet's own cap as the bound nears, so the crossing flush sees a
    // lawful closing speed and commits mid-flight — the label flip, never a parked epsilon, is
    // the exit condition.
    const REAIM_TICKS: u64 = 100;
    /// The ENDGAME re-aim cadence — a fifth of a second, so a pure-pursuit aim is never stale
    /// by more than a fraction of the planet's own sweep.
    const ENDGAME_TICKS: u64 = 10;
    /// The turn budget — a full second, enough for any heading change at the client's look rate.
    const TURN_TICKS: u64 = 50;
    // THE world's own SOI + acquire edge, derived at use — never a transcribed literal. The
    // TARGET planet's OWN shell (its gravitational SOI at its drawn mass — D-REAL-1): per-planet
    // since the true-size re-solve; no single config radius exists.
    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let soi_m = vd_physics::worldgen::realm_regions_for_config(p.universe_seed, &config)
        .iter()
        .find(|r| r.realm == vd_core::pose::RealmId::Planet(planet_seed))
        .map(|r| r.shape.finite_extent())
        .expect("the rendezvous target is a rostered planet of THE world");
    let acquire_edge_m = soi_m - config.band.inset_m;
    let want = vd_core::pose::FrameRef::PlanetCentered { planet_seed }.label();
    let started = Instant::now();
    let mut best = f64::INFINITY;
    let mut loc = String::new();
    let mut leg = 0u32;
    // THE ADAPTIVE LEAD (tail-chase cure, measured on the repeat-round-trip gate): a fixed
    // one-chunk lead under-leads a SLOW dot — a cycle that begins near the planet starts from
    // rest, and while the ramp climbs, the planet's own ~8e4 m/s orbital sweep drags the SOI
    // sideways faster than the dot closes; the pursuit stalls 10-70 km OUTSIDE the acquire
    // edge for the whole budget (closest 12,042,800 m vs 11,974,563 m, 244 s, no flip). The
    // pilot's cure is the classical intercept — lead by TIME-TO-ARRIVE, dist over the closing
    // rate — with TWO stabilisers, each measured in:
    //   * the closing rate is averaged over the WHOLE approach, never the last chunk: an
    //     instantaneous rate collapses to ~0 at the hover point, the time-to-go blows up, the
    //     aim runs 2e7 m ahead along the orbit and the dot tail-chases it outside the SOI
    //     forever (closest 11,984,599 m — 10 km out — with the naive lead);
    //   * the lead's arc is geometry-capped (the aim never runs farther along the orbit than
    //     the current separation), so a cold-start's noisy average cannot aim behind the sun.
    let mut prev_sample: Option<(u64, DVec3)> = None;
    // The planet's own orbital speed, measured off the SAME closed form the aim uses.
    let v_planet_mps = {
        let dt_probe = 1000.0 * p.tick_dt;
        (planet_at(1000) - planet_at(0)).length() / dt_probe
    };
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
        let now_tick = st.universe_tick.unwrap_or(0);
        let dist_now = own_pos(&st).map(|pos| (pos - planet_at(now_tick)).length());
        // THE INTERCEPT SOLUTION (the pilot's classical lead-collision course, and the third
        // and final cure this gate needed). Every FIXED-lead rule stalls, each at its own
        // radius, because the target moves ACROSS the line of sight:
        //   * lead one chunk           → parallel flight at ~59 km (best frozen to the metre);
        //   * pure pursuit (no lead)   → tail-chase standoff at 4,085 m = v_planet · pose lag;
        //   * pure pursuit + lag lead  → crossed on 4 of 5 cycles, stalled at 65 km on the 5th
        //     (the standoff depends on the approach geometry, so no constant lead fixes it).
        // The honest answer solves for WHERE THE TWO ARRIVE TOGETHER: T = |planet(t+T) − me| /
        // v_me, by fixed-point iteration (three passes converge at these speed ratios), plus
        // the pilot's own feedback lag because the command lands that much later. `v_me` is
        // MEASURED off consecutive delivered poses — never assumed — and an unmeasurable or
        // too-slow speed falls back to the one-chunk lead, which is the far-field behaviour
        // that already works.
        let me_now = own_pos(&st);
        let v_me_mps = match (prev_sample, me_now) {
            (Some((t0, p0)), Some(p1)) if now_tick > t0 => {
                (p1 - p0).length() / (((now_tick - t0) as f64) * p.tick_dt)
            }
            _ => 0.0,
        };
        let lag_ticks = (pose_lag_s(p.tick_dt) / p.tick_dt).ceil() as u64;
        // ENDGAME = within a few SOI radii, where the target's own tangential sweep, not the
        // distance, decides whether the pursuit closes.
        let endgame = dist_now.is_some_and(|d| d <= 4.0 * acquire_edge_m);
        let lead_ticks = match (me_now, dist_now) {
            _ if endgame => lag_ticks,
            (Some(me), Some(d)) if v_me_mps > v_planet_mps => {
                let mut t_s = d / v_me_mps;
                for _ in 0..3 {
                    let at = planet_at(now_tick + (t_s / p.tick_dt) as u64);
                    t_s = (at - me).length() / v_me_mps;
                }
                (t_s / p.tick_dt) as u64 + lag_ticks
            }
            _ => REAIM_TICKS,
        };
        // The re-aim cadence tightens as the intercept nears, so the solved course is never
        // stale by more than a fraction of the remaining flight.
        let chunk_ticks = if endgame {
            ENDGAME_TICKS
        } else {
            REAIM_TICKS.min(lead_ticks.max(ENDGAME_TICKS))
        };
        if let Some(d) = dist_now {
            best = best.min(d);
        }
        if let Some(me) = me_now {
            prev_sample = Some((now_tick, me));
        }
        // AIM THROUGH THE BODY, not at it (the last measured millimetre of this problem). A
        // pursuit steered by a DELIVERED pose settles at a standoff of the target's tangential
        // sweep times the feedback lag: measured 4,085 m, then 8,000 m, outside an 11,974,563 m
        // acquire edge — the pursuit converges to 0.07% and stops, because "arrive at the
        // centre" is a condition the lag never lets it satisfy. A pilot's answer is to aim at a
        // point BEYOND the body along the line of sight, so the flight PATH passes through the
        // shell: the standoff then lands the dot half an acquire edge INSIDE, and the crossing
        // commits on the way. Half the edge is the derived depth — deep enough to swallow any
        // lag-scale standoff, shallow enough that the governor's arm (already at the body's own
        // cap by then) keeps the closing speed lawful.
        let aimed_at = planet_at(now_tick + lead_ticks);
        let target = match (me_now, endgame) {
            (Some(me), true) => {
                let toward = (aimed_at - me).normalize_or_zero();
                aimed_at + toward * (0.5 * acquire_edge_m)
            }
            _ => aimed_at,
        };
        if leg.is_multiple_of(10) {
            eprintln!(
                "[fly-in] chunk {leg}: tick={now_tick} loc={loc:?} own_len={:?} best={best:.2} \
                 dist={:?} v_me={v_me_mps:.4e} v_planet={v_planet_mps:.4e} lead={lead_ticks} \
                 chunk={chunk_ticks}",
                own_pos(&st).map(|v| v.length()),
                dist_now,
            );
        }
        leg += 1;
        // ONE chunk of the chase. `arrive_epsilon` a quarter of the acquire edge: the chunk may
        // end EITHER by its tick budget (mid-course) or by standing inside the acquire
        // neighbourhood (endgame) — either way the next iteration re-aims; the true exit is the
        // label flip above.
        //
        // NO TAPER AT ALL (`max_step_m: 0.0`, the controller's explicit no-brake arm). MEASURED,
        // twice: ANY taper on a MOVING aim wedges the endgame. The taper multiplies the throttle
        // by `dist/taper`, the throttle map is geometric (`(v_cap/v_foot)^throttle`), so a small
        // throttle commands a near-FOOT speed; the ramp then re-derives from that collapsed
        // velocity and the dot hovers a few tens of km outside the SOI while the planet's own
        // ~8e4 m/s orbital sweep carries it away (closest 12,007,907 m, then 12,030,752 m, vs the
        // 11,974,563 m acquire edge — both runs out of budget without a flip). Full throttle is
        // SAFE here and is the design's own division of labour: the APPROACH GOVERNOR is the
        // brake, and at the bound its arm has already fallen to the planet's own cap
        // (2·bound/T ≈ 1.33e5 m/s ⇒ ~2.7 km per 20 ms tick against a 1.2e7 m shell), so a step
        // can never out-run the crossing.
        // POINT THE NOSE FIRST — the measured cap on this whole approach. `WalkTo` states its
        // step in the entity's LOCAL frame, and the movement encoding carries only what that
        // frame can express: flying at a target that is off the nose spends a chunk of the
        // command's magnitude (measured `axes_len` 0.8695 held for hundreds of chunks). Under
        // the GEOMETRIC throttle map that is not a small loss — `(v_cap/v_foot)^0.87` instead of
        // `^1.0` turns a lawful 1.907e5 m/s ceiling into 7.6e4 m/s, which is BELOW the planet's
        // own 7.69e4 m/s orbital speed, so the pursuit provably cannot close (measured: `v_me`
        // pinned at 7.6-8.0e4 for 470 chunks, seven km outside the acquire edge). A pilot's
        // answer, and no product change: turn to face the mark, then fly.
        // The turn gets its OWN budget, never the flight chunk's: `LookAt` returns the instant
        // it is aligned (so an already-pointed nose costs one round trip), but a chunk-sized
        // budget cannot finish a large turn, and a half-finished turn is exactly the off-nose
        // magnitude loss above — measured as a cycle that flies at 7e4 m/s while its neighbours
        // fly at 4e5-8e6 and cross.
        let _ = devctl(
            devctl_port,
            &DevRequest::LookAt {
                target: target.to_array(),
                align_epsilon: 0.01,
                max_ticks: TURN_TICKS,
            },
        );
        let flew = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: acquire_edge_m * 0.25,
                max_ticks: chunk_ticks,
                max_step_m: 0.0,
            },
        );
        // NO THROTTLE CUT between chunks — the cut is what killed the ramp (see above). The
        // sticky Move holds the last commanded axes through the poll gap; only ARRIVAL cuts.
        let outcome = match &flew {
            Some(DevResponse::State { .. }) => "ARRIVED",
            Some(DevResponse::Timeout { .. }) => "TIMEOUT",
            other => {
                let _ = other;
                "OTHER"
            }
        };
        if leg % 10 == 1 {
            eprintln!("[fly-in]   chunk {}: walk outcome {outcome}", leg - 1);
        }
        if started.elapsed() >= deadline {
            eprintln!(
                "[fly-in] NO CROSSING: chased {planet:?} for {}s but the location never flipped \
                 (loc {loc:?}); closest approach {best:.2} m vs the {soi_m:.2} m SOI (containment \
                 acquires at {acquire_edge_m:.2} m)",
                started.elapsed().as_secs(),
            );
            let _ = devctl(
                devctl_port,
                &DevRequest::Move {
                    axes: [0.0, 0.0, 0.0],
                },
            );
            return false;
        }
    }
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    eprintln!("[fly-in] location flipped to {loc:?} (closest approach {best:.2} m)");
    assert_eq!(loc, want);
    true
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
            if matches!(walked, Some(DevResponse::State { .. })) {
                // ARRIVED at the aim: CUT THE THROTTLE (sticky Move) and stay PARKED waiting out
                // the commit + the delivered flip. A mid-route chunk (`Timeout`) HOLDS the
                // throttle instead — cutting per chunk zeroed the carried velocity and restarted
                // the speed-law ramp from the foot every ~8 s, which pinned every governed leg at
                // walking pace on the true-size world (the same measured defect the rendezvous
                // chase fixed).
                let _ = devctl(
                    devctl_port,
                    &DevRequest::Move {
                        axes: [0.0, 0.0, 0.0],
                    },
                );
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

/// FULL-throttle +Z held axes — the TRUE-SCALE shell approach. The 0.05 creep throttle was sized
/// for a 150 m interim shell; under the geometric throttle map it commands ~1e3 m/s against a
/// ~1e11 m standoff (a dead leg measured in years). Held FULL axes keep the creep's load-bearing
/// property — NO coordinate in flight, so a mid-commit re-frame cannot straddle — while the
/// APPROACH GOVERNOR itself shapes the arrival: the ceiling falls onto the target realm's own cap
/// as its bound nears, so the crossing flush always sees a lawful closing speed.
#[must_use]
pub fn governed_axes_plus_z() -> [f32; 3] {
    [-1.0, 0.0, 0.0]
}

// ---------------------------------------------------------------------------------------------
// ★ THE HULL FLIGHT (owner ruling 2026-09-05, `owner_decisions_2026-09-05_suit.md` S4): the gates
// fly the SHIPPED PATH — a pilot boards a berthed hull and pushes, the way the owner flies — never
// a walking dot at warp. Guidance is BODY-FRAME: the client draws every realm as a box whose centre
// is expressed in the hull's own frame (the composed row), so "where is the planet from my nose"
// is `realm_boxes[planet].center`, with no world coordinates, no seed table and no second source.
// The nose is −Z (`look_at`'s forward); the stick's first axis pushes along it (`stick_from_input`
// → `local_axes_from_movement`), the second axis yaws (`yaw = −movement[1]`), and pitch rides the
// two pilot action bits. The hull turns and pushes at its own rating; the controller reads the
// error and the measured closing speed and commands bang-bang thrust with a stopping-distance
// brake and a damped turn. Every number here is a control gain or a fraction of the target's own
// shell, never a speed.
// ---------------------------------------------------------------------------------------------

/// Berth one test hull forty metres from the spawn of the roster's home system, BEFORE the shards
/// boot (the home system reads its berths at boot). The same tool the dev cluster uses
/// (`vd-build-ship`), pointed at the fixture's realm stores. Returns the hull's realm.
pub fn berth_test_hull(
    base_dir: &std::path::Path,
    p: &crate::DevClusterParams,
    owner_account: u64,
    push_mps2: f64,
    turn_radps2: f64,
) -> vd_core::pose::RealmId {
    let roster = crate::world_roster(p);
    let spawn = crate::boot_world(p.universe_seed, p.move_speed, p.tick_dt).default_home_offset_m();
    let hull = vd_core::pose::RealmId::Ship(vd_core::ids::EntityId::pack(
        vd_core::entity_kind::EntityKind::Ship,
        1,
        1,
        0,
    ));
    let parent_store = crate::realm_store_path(base_dir, roster.home);
    let ship_store = crate::realm_store_path(base_dir, hull);
    let micro = |v: f64| ((v * 1.0e6).round() as i64).to_string();
    let tool = std::env::var("CARGO_BIN_EXE_vd-build-ship").map_or_else(
        |_| {
            std::env::current_exe()
                .expect("own path")
                .parent()
                .and_then(|d| d.parent())
                .expect("the target profile dir")
                .join("vd-build-ship")
        },
        std::path::PathBuf::from,
    );
    let out = std::process::Command::new(&tool)
        .args([
            "--parent-store",
            &parent_store,
            "--ship-store",
            &ship_store,
            "--owner",
            &owner_account.to_string(),
            "--berth-x-m",
            &(spawn.x + BERTH_OFFSET_M).to_string(),
            "--berth-y-m",
            &spawn.y.to_string(),
            "--berth-z-m",
            &spawn.z.to_string(),
            "--max-push-micro-mps2",
            &micro(push_mps2),
            "--max-turn-micro-radps2",
            &micro(turn_radps2),
        ])
        .output()
        .unwrap_or_else(|e| panic!("vd-build-ship at {}: {e}", tool.display()));
    assert!(
        out.status.success(),
        "vd-build-ship: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    hull
}

/// Where the test hull is berthed, measured from the spawn: forty metres along +x (the dev
/// cluster's `berth_hull` example uses the same offset, so a walk aboard is the same walk).
pub const BERTH_OFFSET_M: f64 = 40.0;

/// Walk the pilot aboard the berthed hull: from the spawn, forty metres along +x, until the
/// client's location names a Ship. Returns the state aboard. The walk is BRAKED at the foot step
/// (`max_step_m`): an unbraked walk-to under the walk ramp overshoots the berth, reverses with its
/// speed kept, and runs away — measured at 115 km from the spawn on the first probe (2026-09-05).
pub fn board_hull(devctl_port: u16, p: &crate::DevClusterParams, deadline: Duration) -> DevState {
    // A LOW THROTTLE, NOT A WALK-TO (2026-09-05, measured): the dev cluster's foot speed is 1 km/s
    // — 20 m per tick — and a hull is smaller than one step, so a full-speed walk passes THROUGH
    // the hull inside one tick (no endpoint lands inside: no re-home) and a closed-loop walk-to
    // under the walk ramp overshoots and runs away (115 km, then 9.6 km with a brake). At 2 %
    // throttle the step is 0.4 m: the endpoint lands inside the hull and the hand-over fires.
    const BOARD_THROTTLE: f32 = 0.02;
    let started = Instant::now();
    let st = loop {
        if let Some(s) = poll_state(devctl_port)
            && own_pos(&s).is_some()
        {
            break s;
        }
        assert!(started.elapsed() < deadline, "board: no delivered own pose");
        std::thread::sleep(Duration::from_millis(100));
    };
    let me = own_pos(&st).expect("delivered");
    let berth = me + DVec3::new(BERTH_OFFSET_M, 0.0, 0.0);
    // Face the berth (the client's closed-loop look), then creep.
    let _ = devctl(
        devctl_port,
        &DevRequest::LookAt {
            target: berth.to_array(),
            align_epsilon: 0.02,
            max_ticks: 200,
        },
    );
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [BOARD_THROTTLE, 0.0, 0.0],
        },
    );
    let _ = p; // the foot speed is the shard's; the throttle is a fraction of it
    loop {
        if let Some(s) = poll_state(devctl_port)
            && s.location.as_deref().is_some_and(|l| l.starts_with("Ship"))
        {
            let _ = devctl(
                devctl_port,
                &DevRequest::Move {
                    axes: [0.0, 0.0, 0.0],
                },
            );
            return s;
        }
        assert!(
            started.elapsed() < deadline,
            "board: the pilot never re-homed into the hull"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The drawn box whose realm label starts with `prefix`, as (centre in the hull's frame, extent).
fn box_of(st: &DevState, prefix: &str) -> Option<(DVec3, f64)> {
    st.realm_boxes
        .iter()
        .find(|b| b.realm.starts_with(prefix))
        .map(|b| (DVec3::from_array(b.center), b.extent_m))
}

/// One tick of the hull's stick: forward push (+1 / 0 / −1 along the nose), yaw stick, pitch bits.
fn hull_stick(devctl_port: u16, forward: f32, yaw: f32, pitch: f32) {
    use vd_core::controls::{PILOT_PITCH_DOWN_INDEX, PILOT_PITCH_UP_INDEX};
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [forward, yaw, 0.0],
        },
    );
    let _ = devctl(
        devctl_port,
        &DevRequest::Action {
            bit: PILOT_PITCH_UP_INDEX,
            pressed: pitch > 0.0,
        },
    );
    let _ = devctl(
        devctl_port,
        &DevRequest::Action {
            bit: PILOT_PITCH_DOWN_INDEX,
            pressed: pitch < 0.0,
        },
    );
}

/// ★ FLY THE HULL INTO A DRAWN REALM: aim the nose at the box, push, brake on the stopping
/// distance, and hold until the client's location names the target realm (the hand-over landed).
/// `arrive_frac` is how deep inside the target's shell the hull settles, as a fraction of its
/// extent; `turn_radps2` is the hull's turn rating (the test berthed it, so the test knows it).
/// The turn is a DOUBLE INTEGRATOR (a stick makes an angular acceleration), so the aim commands
/// on the stopping angle: `err + rate·|rate| / (2·a)` — the angle the hull will still turn
/// through if it brakes now — with the rate measured over a half-second window, never one poll.
/// Returns the state at arrival. Panics past `deadline` with the last measured error.
pub fn fly_hull_to(
    devctl_port: u16,
    target_prefix: &str,
    arrive_frac: f64,
    turn_radps2: f64,
    deadline: Duration,
) -> DevState {
    const CONTROL_PERIOD: Duration = Duration::from_millis(100);
    const RATE_WINDOW: usize = 5;
    // The hull's spin is never damped by physics (no angular drag in space), so this controller
    // IS the damper: full stick authority, and the stopping angle reads the full turn rating.
    const STICK_LIMIT: f64 = 1.0;
    const STICK_PROPORTIONAL_RAD: f64 = 0.10;
    const AIM_TOLERANCE_RAD: f64 = 0.03;
    const BRAKE_MARGIN: f64 = 1.5;
    let started = Instant::now();
    let mut history: std::collections::VecDeque<(Instant, f64, f64, f64)> =
        std::collections::VecDeque::new(); // (when, dist, yaw_err, pitch_err)
    let mut last = String::new();
    let mut accel_est: f64 = 0.0;
    let mut prev_closing: Option<(Instant, f64)> = None;
    let mut last_forward: f32 = 0.0;
    let mut periods: u64 = 0;
    // The stopping-angle command for one axis: how far the hull still turns if it brakes now.
    let stopping = |err: f64, rate: f64| err + rate * rate.abs() / (2.0 * turn_radps2.max(1e-9));
    loop {
        let Some(st) = poll_state(devctl_port) else {
            std::thread::sleep(CONTROL_PERIOD);
            assert!(
                started.elapsed() < deadline,
                "fly: dev-control unreachable ({last})"
            );
            continue;
        };
        if st
            .location
            .as_deref()
            .is_some_and(|l| l.starts_with(target_prefix))
        {
            hull_stick(devctl_port, 0.0, 0.0, 0.0);
            return st;
        }
        let Some((d, extent)) = box_of(&st, target_prefix) else {
            std::thread::sleep(CONTROL_PERIOD);
            assert!(
                started.elapsed() < deadline,
                "fly: the target is not drawn ({last})"
            );
            continue;
        };
        let dist = d.length();
        let dir = d / dist;
        // Aim error in the hull's own frame: the nose is −Z.
        let yaw_err = (-dir.x).atan2(-dir.z);
        let pitch_err = dir.y.clamp(-1.0, 1.0).asin();
        let now = Instant::now();
        history.push_back((now, dist, yaw_err, pitch_err));
        while history.len() > RATE_WINDOW {
            history.pop_front();
        }
        let (closing, yaw_rate, pitch_rate) = match (history.front(), history.len() > 1) {
            (Some(&(t0, d0, y0, p0)), true) => {
                let dt = (now - t0).as_secs_f64().max(1e-3);
                (
                    (d0 - dist) / dt,
                    wrap_pi(yaw_err - y0) / dt,
                    (pitch_err - p0) / dt,
                )
            }
            _ => (0.0, 0.0, 0.0),
        };
        let yaw_u = stopping(yaw_err, yaw_rate);
        let pitch_u = stopping(pitch_err, pitch_rate);
        // Yaw: `yaw = −movement[1]`, so a positive stopping angle wants a negative stick.
        let yaw_cmd = (-(yaw_u / STICK_PROPORTIONAL_RAD).clamp(-1.0, 1.0) * STICK_LIMIT) as f32;
        // Pitch rides two bits: press the side that shrinks the stopping angle, release inside
        // the tolerance.
        let pitch_cmd: f32 = if pitch_u.abs() > AIM_TOLERANCE_RAD {
            pitch_u.signum() as f32
        } else {
            0.0
        };
        let aimed = yaw_err.abs() < AIM_TOLERANCE_RAD && pitch_err.abs() < AIM_TOLERANCE_RAD;
        let remaining = (dist - arrive_frac * extent).max(0.0);
        if let Some((t0, c0)) = prev_closing
            && last_forward != 0.0
        {
            let dt = (now - t0).as_secs_f64().max(1e-3);
            accel_est = accel_est.max(((closing - c0) / dt).abs());
        }
        prev_closing = Some((now, closing));
        let stop_dist = if accel_est > 0.0 {
            closing.abs() * closing.abs() / (2.0 * accel_est)
        } else {
            0.0
        };
        let forward: f32 = if !aimed {
            0.0
        } else if closing < 0.0 || stop_dist * BRAKE_MARGIN < remaining {
            1.0
        } else if closing > 0.0 {
            -1.0
        } else {
            0.0
        };
        hull_stick(devctl_port, forward, yaw_cmd, pitch_cmd);
        last_forward = forward;
        periods += 1;
        if periods.is_multiple_of(50) {
            eprintln!(
                "[hull] t={:.0}s dist {dist:.3e} m closing {closing:.3e} m/s accel {accel_est:.2e} yaw {yaw_err:+.3} ({yaw_rate:+.3}/s) pitch {pitch_err:+.3} ({pitch_rate:+.3}/s) fwd {forward:+.0} loc {:?}",
                started.elapsed().as_secs_f64(),
                st.location
            );
        }
        last = format!(
            "dist {dist:.3e} m (extent {extent:.3e}), closing {closing:.3e} m/s, accel {accel_est:.3e}, yaw {yaw_err:.3} rad, pitch {pitch_err:.3} rad, loc {:?}",
            st.location
        );
        assert!(started.elapsed() < deadline, "fly: never arrived — {last}");
        std::thread::sleep(CONTROL_PERIOD);
    }
}

/// Wrap an angle difference into (−π, π].
fn wrap_pi(a: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut x = (a + std::f64::consts::PI) % two_pi;
    if x < 0.0 {
        x += two_pi;
    }
    x - std::f64::consts::PI
}
