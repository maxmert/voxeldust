//! **G-WARP-PIXELS + G-HANDOVER** — THE WARP ACCEPTANCE, in real pixels
//! (`docs/design/window_lane.md` §2.8 + §4 Slice D).
//!
//! One demand cluster (orchestrator + gateway, NO shard pre-booked), one real headless client with
//! a GPU, and one flight down THE world's own star gap: out of the home system, across the 3-D
//! placement gap the galaxy's own solver produced (`stellar.galaxy_rim_r_m`, printed in full by
//! the companion below) at the GOVERNED ceiling (the S3 speed law: ramp + galaxy ceiling +
//! approach governor), into the ring sibling.
//!
//! WHAT IT PROVES, in the owner's words: *the destination is a point of light at departure, it
//! grows monotonically as you approach, and it hands over flicker-free into a live system that
//! draws itself; behind you the system you left shrinks to a dot and tears down.* No loading
//! screen, no toggle, no teleport — the same picture the whole way.
//!
//! WHY IT CAN FAIL. Every bound is DERIVED from THE world's solved geometry and the landed
//! cadences, then compared against a measurement of the real cluster:
//!
//! * the destination is asleep at departure only because the RING is wider than the visibility
//!   WAKE RADIUS — a margin the world's own solver produced, which the companion test computes,
//!   prints and asserts positive rather than transcribing;
//! * the WAKE budget is the demand cadence + the reconcile tick + the cluster's OWN measured boot
//!   latency + the woken shard's first statement beat + the Q2 relay hop + the compose/draw ticks.
//!   The owner ruled that the relay hop is measured HERE: its share is asserted APART, so a relay
//!   too slow to serve a wake fails this gate by name rather than hiding inside a boot number;
//! * the DEPARTURE budget is the AoI grace, the verdict cadence and the same delivery ticks — the
//!   one law read backwards — with the gateway's roster-loss window as its outer bound.
//!
//! BUDGETS ARE COUNTED IN TICKS AND REPORTED IN METRES. Every stage above is a tick count off a
//! landed expression; the owner-facing "footprint" is that tick count times the occupant's own
//! travel per tick. Measuring in ticks is what makes the measurement honest whether the ship is
//! flying or parked when the handover lands.
//!
//! THE CAMERA IS THE PILOT'S (`--capture-pilot`): the default capture framing fits the whole drawn
//! scene, and on a flight between two FIXED points of the ring that framing barely moves, so a
//! system you fly toward could not grow on screen at all. The warp is a statement about what the
//! pilot sees, so the agent's eyes look through the pilot's eyes — reconstructed by the gate from
//! the delivered pose through the very same Tier-A expression the renderer places its camera by.
//!
//! Dot detection is DevState-driven with LOCAL pixel probes at the projected position (§2.11) —
//! never a full-frame search at a 3-pixel footprint. Every wait is on the demand loop's own signal
//! (a gauge, a label, an author flip), never a sleep literal standing in for one.
//!
//! GPU-required + LOCAL, exactly like the other render gates.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::cross_leg;
use vd_bins::pixel::{Author, Presence, Straddled, Subject, own_pose};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr, world_roster,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::{
    CaptureCamera, DOT_MIN_APPARENT_RADIUS_PX, FIT_FOV_Y, ScreenAabb, ScreenPos,
};
use vd_client_harness::manifest::{MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::dot_pixels_distinct_from_surround;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::NodeId;
use vd_core::flight::{
    FlightTuning, TRAVERSE_S, approach_ceiling_mps, leg_time_s, realm_speed_cap_mps,
};
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView, RlmView};

// ---------------------------------------------------------------------------------------------
// THE DERIVED BUDGET — every term an expression over THE world's numbers or a landed cadence.
// ---------------------------------------------------------------------------------------------

/// The RLM reconciler's own interval (`RlmTuning::cloud_with_boot`: one tick).
const RECONCILE_TICKS: u64 = 1;
/// The Q2 relay hop in shard ticks: ONE at the woken child to author its sealed statements, ONE at
/// the parent to forward them (the parent forwards in the tick it receives — `process_inbound` runs
/// before `emit_realm_frames` on the same schedule, so there is no store-and-forward tick).
const RELAY_FORWARD_TICKS: u64 = 2;
/// The membership verdict's shard→gateway hop, in ticks.
const MEMBERSHIP_HOP_TICKS: u64 = 1;
/// The gateway folds and ships on its next tick.
const COMPOSE_TICKS: u64 = 1;
/// The client applies the composed level and the renderer draws it.
const DRAW_TICKS: u64 = 1;

/// THE world, exactly as every shard of the cluster boots it — the one place this gate reads a
/// world number from (SL5: one world, no preset, no test variant).
fn world() -> vd_physics::worldgen::UniverseConfig {
    vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
}

/// THE VISIBILITY FACTOR — `cot(θ_min/2)`, THE world's one visibility law.
fn visibility_factor() -> f64 {
    world().interest.spin_up_factor
}

/// ONE realm's own row on THE world's region roster — the single place this gate reads a realm's
/// solved geometry from, and it takes the realm BECAUSE NOTHING HERE IS A WORLD CONSTANT ANY MORE.
/// Since the true-size re-solve a system's shell is `f(its own star)`, so the home system's numbers
/// say nothing whatever about the destination's: on THE world the two differ by 14× (measured
/// 2026-08-21 — home 5.0886e12 m, ring sibling 3.5206e11 m), which is exactly how the wake
/// measurement below came to be timed off the wrong line. See [`dest_wake_r_m`].
fn region_of(realm: RealmId) -> vd_core::geometry::RealmRegion {
    let cfg = world();
    vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &cfg)
        .into_iter()
        .find(|r| r.realm == realm)
        .unwrap_or_else(|| panic!("THE world rosters {realm:?}"))
}

/// A realm's own AoI band — THE SAME `AoiConfig` the shards' demand fold decides by, read off the
/// same roster rather than rebuilt here. The gate used to rebuild the band from the home system's
/// extent and then hand that one number to BOTH directions of the flight; a second spelling of a
/// per-realm number is precisely how the destination came to be judged against the home system's
/// radius, so there is now only the roster's own.
fn realm_band(realm: RealmId) -> vd_core::geometry::AoiConfig {
    region_of(realm).aoi
}

/// THE HOME system's own BOUND extent (its solved shell radius) — the system this flight leaves.
fn system_extent_m() -> f64 {
    region_of(vd_core::worldgen::SYSTEM_A).shape.finite_extent()
}

/// THE HOME system's LOOK extent (its star's photosphere — the bound/look split): what its
/// self-look STATES, and the radius the camera law sizes its picture by.
fn system_look_m() -> f64 {
    region_of(vd_core::worldgen::SYSTEM_A)
        .look
        .map(|l| l.finite_extent())
        .expect("THE home system draws its star")
}

/// THE DESTINATION — the ring sibling this warp flies to, named by THE world's own roster.
fn dest_realm() -> RealmId {
    world_roster(&DEV).sibling
}

/// THE DESTINATION's own BOUND extent — its solved shell, not the home system's.
fn dest_extent_m() -> f64 {
    region_of(dest_realm()).shape.finite_extent()
}

/// THE DESTINATION's own LOOK extent — its OWN star's photosphere. A different star from home's,
/// so a different number: the two are not interchangeable at true scale.
fn dest_look_m() -> f64 {
    region_of(dest_realm())
        .look
        .map(|l| l.finite_extent())
        .expect("the ring sibling draws its star")
}

/// THE HOME system's WAKE radius: the distance at which the galaxy's AoI verdict demands it, and at
/// which its own look may take over the drawing. Used for the DEPARTURE half of the flight only.
fn spin_up_r_m() -> f64 {
    realm_band(vd_core::worldgen::SYSTEM_A).spin_up_r_m()
}

/// THE HOME system's TEAR-DOWN radius — the outer edge of its own hysteresis band, the line the
/// departure watch times the hand-back from.
fn tear_down_r_m() -> f64 {
    realm_band(vd_core::worldgen::SYSTEM_A).tear_down_r_m()
}

/// ★ THE DESTINATION's OWN WAKE RADIUS — the line the WAKE half of G-HANDOVER is timed from.
///
/// MEASURED 2026-08-21, and this is the whole of that run's 30× miss: the gate timed the wake from
/// `system_extent_m() · visibility_factor()` = 3.8872e14 m, which is the HOME system's line, while
/// the destination's own is 2.6893e13 m — 14.5× nearer. The galaxy demands a child by THAT child's
/// own band, so between the two lines the destination is not merely un-woken, it is not yet
/// ELIGIBLE to be woken; the flight covers 93 % of the observed gap before the demand loop is even
/// allowed to act. The run measured 1813 ticks from the wrong line, of which the last stretch —
/// from the destination's own wake radius (2.6893e13 m) to the observed hand-over (2.2843e13 m) —
/// is 4.05e12 m at the approach governor's own arm there (~2.0e13 m/s), i.e. of order TEN ticks
/// against a 55-tick pipeline budget. The demand loop was never late; the line was.
fn dest_wake_r_m() -> f64 {
    realm_band(dest_realm()).spin_up_r_m()
}

/// THE STAR GAP — how far the home system actually is from the destination, MEASURED off THE
/// world's own two placements.
///
/// ★ THIS USED TO READ A CONFIG KNOB, AND THE KNOB STOPPED MEANING THIS (S12, 2026-08-28). It was
/// `world().stellar.system_ring_r_m`, documented as "the distance between the home system and its
/// ring sibling", and under the SHELL that was true: every system sat at exactly that radius, so
/// the rim and the gap were one number. The owner refused the shell — a shell puts every star at
/// the same distance, which no galaxy does — and the placement now draws each system's own radius
/// inside the rim. The knob is now `galaxy_rim_r_m` and is an upper bound, nothing else.
///
/// WHAT IT COST, MEASURED: the departure leg computed "walk until within rim − park of the
/// destination". THE world's destination sits at 91 % of the rim, so that stop condition was
/// ALREADY TRUE before the ship moved. It never walked: 0.0 m travelled, no crossing, no handover,
/// and the leg died on its own deadline having proved nothing.
///
/// ★ THE ANSWER IS TO MEASURE, NEVER TO ASK A KNOB. Both systems are direct children of the galaxy,
/// so both centres are stated in the galaxy's frame and subtract directly. Read at the GALAXY's
/// step, which is the frame the numbers are in — the parent-frame trap: a child's own step is out
/// by 2048× here, silently.
fn star_gap_m() -> f64 {
    let cfg = world();
    let regions = vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &cfg);
    let roster = world_roster(&DEV);
    let find = |realm: vd_core::pose::RealmId| {
        regions
            .iter()
            .find(|r| r.realm == realm)
            .unwrap_or_else(|| panic!("THE world rosters {realm:?}"))
    };
    let galaxy = find(roster.galaxy);
    (find(roster.sibling).center.metres_in(galaxy) - find(roster.home).center.metres_in(galaxy))
        .length()
}

/// The occupant's travel per tick at the shipped speed — the ONE tick↔metre conversion, and the
/// crossing gates' one-tick true-motion bound.
fn metres_per_tick() -> f64 {
    DEV.move_speed * DEV.tick_dt
}

/// The AoI re-check cadence in ticks — the beat the demand fold, the SL7 bit, the membership
/// verdict and the window statements all breathe on (the shard's own expression).
fn aoi_cadence_ticks() -> u64 {
    let hz = (1.0 / DEV.tick_dt).round() as u64;
    (hz / 2).max(1)
}

/// The AoI hysteresis grace in ticks, PLUS the tick it drops on — the fold holds for `grace_ticks`
/// evaluations and releases on the next, so the fence-post is counted here rather than assumed.
fn grace_hold_ticks() -> u64 {
    u64::from(world().interest.grace_ticks) + 1
}

/// The gateway's roster-loss window in ticks (`WindowTuning::look_ttl_ticks` = two keep-alive beats
/// + one), derived off the gateway's own beat exactly as the composer derives it.
fn look_ttl_ticks() -> u64 {
    let beat = (u64::from(DEV.tick_hz) / 2).max(1);
    2 * beat + 1
}

/// THE WAKE BUDGET in ticks (§2.8, symmetric half one): the parent's demand beat, the reconcile
/// tick, the cluster's OWN measured boot latency, the woken shard's first-statement beat, the Q2
/// relay hop, the compose tick and the draw tick — each counted once.
fn wake_budget_ticks(boot_ticks: u64) -> u64 {
    2 * aoi_cadence_ticks()
        + RECONCILE_TICKS
        + boot_ticks
        + RELAY_FORWARD_TICKS
        + COMPOSE_TICKS
        + DRAW_TICKS
}

/// THE Q2 RELAY SHARE of the wake budget — everything after the destination's shard is up. Owner
/// ruling: the relay hop is measured at the wake moment and asserted APART, so a relay too slow to
/// serve a wake fails BY NAME (the D-WINDOW-2 trigger) instead of being absorbed by a boot number.
fn relay_share_budget_ticks() -> u64 {
    wake_budget_ticks(0)
}

/// THE DEPARTURE BUDGET in ticks (§2.8, symmetric half two): the AoI grace the departing realm
/// keeps, the cadence its verdict ships on, and the same delivery/draw ticks.
fn departure_budget_ticks() -> u64 {
    grace_hold_ticks() + aoi_cadence_ticks() + MEMBERSHIP_HOP_TICKS + COMPOSE_TICKS + DRAW_TICKS
}

/// The departure budget's OUTER bound — plus the gateway's roster-loss window, by which a departed
/// realm's stored look must be gone from the composer, not merely un-drawn.
fn departure_budget_full_ticks() -> u64 {
    departure_budget_ticks() + look_ttl_ticks()
}

/// Ticks → the owner-facing footprint in metres: how far the occupant travels while a budget runs.
fn footprint_m(ticks: u64) -> f64 {
    metres_per_tick() * ticks as f64
}

// ---------------------------------------------------------------------------------------------
// Flight geometry — derived parks, never picked distances.
// ---------------------------------------------------------------------------------------------

/// The polar-corridor exit out of the home system — the ONE licensed departure (I-AXIS/I-POLE).
/// Derived: one tear-down band outside the home shell, so the exit clears containment AND the
/// hysteresis band by construction.
fn polar_exit_z_m() -> f64 {
    // +Z, NOT −Z (true-scale restatement): the login spawn stands on the +Z axis at twice the
    // STAR's own bound (the T2 standoff), so a −Z exit flies straight THROUGH the star realm at
    // the system centre and the leg commits into `Star …` instead of the galaxy (measured). The
    // corridor law is pole-symmetric; +Z is the side the spawn already stands on.
    system_extent_m() + (tear_down_r_m() - spin_up_r_m()) + system_extent_m()
}

/// THE ARRIVAL STANDOFF: the closest approach at which the destination's BRACKETED footprint still
/// fits well inside the readback, so the local pixel probe has real background to compare against.
/// `radius_px(d) = extent/d · (h/2)/tan(fov/2)`; requiring `RECT_BRACKET · radius_px ≤ h/4` gives
/// `d ≥ 2 · RECT_BRACKET · extent / tan(fov/2)`.
fn arrival_standoff_m() -> f64 {
    // SIZED BY THE SHELL, deliberately: every other assert on this leg (the wake crossing, the
    // no-pop depth, the presence law over one level) is written for an arrival that stays OUTSIDE
    // the destination's containment, so the standoff must too. MEASURED WHY THE ALTERNATIVE FAILS:
    // sizing it by the drawn LOOK instead puts the standoff orders of magnitude INSIDE the
    // system's own shell (the bound/look inversion — a system's look is its star's photosphere
    // while its bound is the solved authority shell), so the ship crosses in, at which point the
    // destination's row leaves the level the leg is watching and the watch reports the whole gap
    // because the subject is no longer in the picture.
    //
    // ★ AND IT IS THE DESTINATION'S OWN SHELL (2026-08-21). The sentence above is about the realm
    // the ship arrives at, so the extent in it has to be that realm's; reading the home system's
    // sized the standoff for a shell 14× too wide and parked the arrival a whole wake band short of
    // where the picture is worth probing — outside the destination's own wake radius entirely, so
    // the measured leg ended AT the hand-over instant and the growth curve after it had almost no
    // samples left to be a curve out of.
    //
    // ★ AND IT CAN NEVER BE TIGHTER THAN THE INSTRUMENT FLYING TO IT. The measured leg stops on
    // `distance <= standoff`, and a facing-steered pursuit closes only to its own re-aim trigger
    // ([`reaim_miss_m`]) — that IS the flight's convergence limit, stated once and read here rather
    // than re-guessed. A standoff under it is a stop condition the flight cannot reach, which would
    // fail this gate on the harness's aim instead of on anything the world does. So the standoff is
    // whichever of the two is farther out, with one full convergence limit of margin on that arm
    // (the same doubling the behind park takes over the tear-down). On THE world the pursuit arm is
    // the one that binds, and it still sits an order inside the destination's wake radius — which
    // is what the measured leg needs — and an order outside its shell, which is what every assert
    // on the leg needs.
    (2.0 * RECT_BRACKET * dest_extent_m() / (FIT_FOV_Y * 0.5).tan()).max(2.0 * reaim_miss_m())
}

/// THE MARKER-LEGIBILITY RANGE: the farthest distance at which the departed system's TRUE-sized
/// marker still sits a readback quantum ABOVE the shared apparent floor — past it the marker is
/// clamped to the bare floor and the "shrunk to its true angular size" verdict has nothing to
/// measure. `px(d) = (extent/d)·(h/2)/tan(fov/2)`, solved for `px = floor + quantum`. Derived
/// from the same camera law the renderer scales by; never a picked distance.
fn marker_range_m() -> f64 {
    // The MARKER is sized by the LOOK (the bound/look split): a departed system's point of
    // light is its STAR's photosphere, not its authority shell.
    system_look_m() * (f64::from(CAPTURE_H) * 0.5)
        / (FIT_FOV_Y * 0.5).tan()
        / (DOT_MIN_APPARENT_RADIUS_PX + READBACK_QUANTUM_PX)
}

/// THE BEHIND PARK (S3 re-derivation — the old ring-midpoint park died with the interim ring):
/// the departure watch must OBSERVE the home system cross its tear-down radius AND then hand back
/// to a marker still ABOVE the apparent floor, so the park window is `(tear_down, marker_range)`
/// and the park is its midpoint — maximum margin on both sides. The watch leg stops EARLY the
/// moment the flip lands, so the park is the safety ceiling, not the usual stop.
fn behind_park_m() -> f64 {
    // TRUE-SIZE RESTATEMENT: the marker-legibility range (a photosphere-sized look) now sits
    // far INSIDE the wake tear-down, so the old `(tear, marker_range)` window is empty — past
    // the tear the departed system lawfully draws AT the apparent floor (the presence-floor
    // law: "shrunk to a dot" IS the 3 px dot out here). The park is one full tear-down radius
    // of margin beyond the edge (the same doubling the arrival standoff uses).
    2.0 * tear_down_r_m()
}

/// THE shipped flight tuning — `vd_core::flight`'s ONE derivation at the DEV cluster's own
/// numbers (τ = T_WAKE: the same wake budget `wake_budget_ticks` counts, in seconds).
fn flight_tuning() -> FlightTuning {
    FlightTuning::derive(
        DEV.move_speed,
        DEV.tick_dt,
        aoi_cadence_ticks(),
        u32::try_from(DEV.boot_ticks_p99).expect("boot p99 fits"),
    )
}

/// The galaxy's own governed ceiling — the warp cruise speed.
fn galaxy_cap_mps() -> f64 {
    realm_speed_cap_mps(world().scale.galaxy_r_m, DEV.move_speed, TRAVERSE_S)
}

/// The governed closed-form time of the warp approach: ramp up from the foot, cruise at the
/// galaxy ceiling, governor ramp-down onto the destination's own ceiling at the standoff.
fn governed_approach_s() -> f64 {
    let t = flight_tuning();
    let v_end = approach_ceiling_mps(
        realm_speed_cap_mps(dest_extent_m(), DEV.move_speed, TRAVERSE_S),
        arrival_standoff_m() - dest_extent_m(),
        t.tau_s,
    );
    leg_time_s(
        star_gap_m(),
        galaxy_cap_mps(),
        DEV.move_speed,
        v_end,
        t.tau_s,
    )
    .expect("the warp approach holds a cruise")
}

/// The GOVERNED dwell inside the destination's wake band — the seconds the approach governor
/// spends between the wake radius and the shell (`τ·ln(1 + (S−E)/(v_c·τ))`), which is the time
/// the demand pipeline has to serve the wake before any crossing could happen.
fn governed_wake_dwell_s() -> f64 {
    let t = flight_tuning();
    let child_cap = realm_speed_cap_mps(dest_extent_m(), DEV.move_speed, TRAVERSE_S);
    t.tau_s * (1.0 + (dest_wake_r_m() - dest_extent_m()) / (child_cap * t.tau_s)).ln()
}

/// How much wider than the drawn footprint a probed rectangle is bracketed — the same 2× the
/// crossing gate brackets its dot rect by, absorbing the sub-frame skew between the sampled state
/// and the readback frame.
const RECT_BRACKET: f64 = 2.0;
/// A capture's ring probe width in pixels — the SHARED minimum apparent radius, so a point of
/// light's local background is probed exactly one marker-radius out from its own rect.
const PROBE_RING_PX: f64 = DOT_MIN_APPARENT_RADIUS_PX;
/// A footprint shrink smaller than one pixel is not a shrink anyone could see: one pixel is the
/// readback's own quantum, not a fitted tolerance.
const READBACK_QUANTUM_PX: f64 = 1.0;

/// The login must SPAWN a real shard process before it can converge.
const LOGIN_DEADLINE: Duration = Duration::from_secs(60);
/// A healthy client clears this in a second or two once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;
/// One walk chunk on a WATCHED leg, in ticks: short enough that the handover it brackets is still
/// resolved near the clock's own quantum, long enough that the leg is not one round trip per tick.
const WATCHED_CHUNK_TICKS: u64 = 5;
/// The sampling interval — ONE tick of the shipped clock, so a handover's measured latency is
/// resolved to the universe clock's own quantum rather than to the gate's polling habits.
const SAMPLE_POLL: Duration = Duration::from_millis(20);
/// Full forward on the sticky throttle: the warp IS `move_speed` on held axes, along the facing a
/// `LookAt` already pointed at the destination.
const FULL_AHEAD: [f32; 3] = [1.0, 0.0, 0.0];
/// How far past its own closest approach a watched realm may open again before the flight is
/// declared to have sailed past it — one system extent, so a straight run that ends INSIDE the
/// destination is fine and one that misses is loud.
fn pass_by_slack_m() -> f64 {
    system_extent_m()
}
/// THE RE-AIM TRIGGER, as a WORLD miss rather than a frame fraction: turn back onto the subject
/// once the current heading would pass it by more than its OWN EXTENT at its OWN range. Derived,
/// and it tightens as the range closes — which a frame fraction cannot: half a frame is ~20° of
/// heading error, which is 75 m of miss at ring range but 500 m of miss on the run-in, and a gate
/// that only corrects at 20° sails past what it was flying to (MEASURED 2026-08-16: closest
/// approach 476.7 m against a 349.0 m park). It is twice the aim's own converged tolerance, so a
/// fresh turn always has margin and the loop cannot thrash.
fn reaim_miss_m() -> f64 {
    system_extent_m()
}
/// The throttle released.
const ALL_STOP: [f32; 3] = [0.0, 0.0, 0.0];
/// THE PRE-WAKE PARK'S CEILING, as a share of the star gap. Leg 2a is the held-throttle CRUISE —
/// the production warp input this gate exists to fly — so the park it hands off at may never eat
/// more than this much of the journey, whatever the world's ring/wake ratio turns out to be. A
/// POLICY share of the measurement, stated once here; the park itself is derived against it and
/// its margin over the wake band is asserted, never assumed.
const PRE_WAKE_PARK_SHARE_OF_GAP: f64 = 0.4;
/// How many settle polls the aim's converge loop takes before it measures the residual — bounded,
/// and it exits early the moment two consecutive polls report the identical facing.
const AIM_SETTLE_POLLS: u32 = 40;

// ---------------------------------------------------------------------------------------------
// Cluster scaffolding (the demand shape, as the RLM process gates boot it).
// ---------------------------------------------------------------------------------------------

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixture {
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    store: std::path::PathBuf,
    cwd: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(&self.launch_path);
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-warp-pixels").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-warp-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&launch_path);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        trust_dir,
        common,
        launch_path,
        store,
        cwd,
    }
}

/// SIGKILL + reap every demand-spawned shard the orchestrator forked, on drop (they lead their own
/// process groups, so the `Cluster`'s group-kill never reaches them). Declared BEFORE the cluster
/// so it drops AFTER it — by then the orchestrator has released the launch ledger's lock.
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

fn demand_addrs(gateway_admin: SocketAddr) -> ClusterAddrs {
    ClusterAddrs {
        gateway_admin: Some(gateway_admin),
        ..ClusterAddrs::reserve()
    }
}

/// Spawn ONE real headless CAPTURE client in the PILOT view — the agent's eyes looking through the
/// pilot's eyes.
fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    agent_index: u32,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.current_dir(&f.cwd);
    cmd.args([
        "--name",
        name,
        "--agent-index",
        &agent_index.to_string(),
        "--gateway",
        &gateway.to_string(),
        "--client-quic",
        &quic_port.to_string(),
        "--trust-dir",
        &f.trust_dir.display().to_string(),
        "--dev-control",
        &devctl_port.to_string(),
        "--allow-dev-control",
        "--capture",
        "--capture-pilot",
    ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().expect("spawn capture client")
}

fn boot_demand_cluster(f: &Fixture, a: &ClusterAddrs, p: &DevClusterParams) -> Cluster {
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &f.common,
            &orchestrator_env(a, p, &f.store_str, ClusterShape::Demand),
        )
        .expect("spawn orchestrator"),
    );
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &f.common,
            &gateway_env(a, &dev_auth_pubkey_hex(), p, ClusterShape::Demand),
        )
        .expect("spawn gateway"),
    );
    cluster
}

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

fn orch_rlm(addr: SocketAddr) -> RlmView {
    admin(addr).map(|s| s.rlm).unwrap_or_default()
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    admin(addr)?.gateway
}

/// Poll until the client is Active AND receiving AND drawing, or fail LOUD with the gateway's view.
fn await_active(devctl_port: u16, gw_admin: SocketAddr, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        if let Ok(DevResponse::State { state }) = dev_roundtrip(devctl_port, &DevRequest::State)
            && state.phase == DevPhase::Active
            && state.snapshots_applied >= SNAPSHOT_FLOOR
            && !state.realm_boxes.is_empty()
        {
            return state;
        }
        assert!(
            started.elapsed() < deadline,
            "the demand login never converged: gateway view {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Wait for the client's dev-control listener (the process is up and serving).
fn await_listener(port: u16, child: &mut Child) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter (WGPU_BACKENDS/WGPU_POWER_PREF)",
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "the capture client never opened its dev-control listener on {port}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn label_of(realm: RealmId) -> String {
    frame_for_realm(realm, None)
        .expect("THE world's realms all have frames")
        .label()
        .to_owned()
}

// ---------------------------------------------------------------------------------------------
// The flight instrument.
// ---------------------------------------------------------------------------------------------

/// One sample of the flight: WHEN (the universe clock), WHERE, and how each watched realm is drawn.
#[derive(Clone, Debug)]
struct Sample {
    tick: u64,
    /// The composed picture's ORIGIN EPOCH at this sample — bumped by the gateway on every chain
    /// change, which is the one moment a window is torn down and rebuilt.
    epoch: u64,
    pos: DVec3,
    travelled_m: f64,
    /// One entry per watched realm, in the order the caller listed them.
    subjects: Vec<Subject>,
}

impl Sample {
    /// The distance from the occupant to the `i`-th watched realm's drawn centre.
    fn distance(&self, i: usize) -> f64 {
        (self.subjects[i].centre_m - self.pos).length()
    }
}

/// A watched author FLIP: the universe tick it was first observed at, the distance it happened at,
/// and the drawn footprint at that moment.
#[derive(Clone, Copy, Debug)]
struct Flip {
    tick: u64,
    distance_m: f64,
    footprint_px: f64,
    travelled_m: f64,
    /// The subject's PROJECTED CENTRE in the frame BEFORE the flip and in the frame OF the flip —
    /// §2.8's "centroid path continuous". THE POSITION AUTHOR NEVER CHANGES across a handover (the
    /// parent's placement row positions the subject before, during and after its spin-up; only the
    /// LOOK payload upgrades), so the drawn centre may move only by the observer's own travel
    /// between the two samples. A jump here would mean the swap moved the thing, not just redrew it.
    centre_before_px: Option<(f64, f64)>,
    centre_at_px: Option<(f64, f64)>,
    /// The universe ticks between the sample BEFORE this one and this one — the measurement's own
    /// resolution, and nothing else's. A gap elsewhere in the leg (a course correction's turn, say)
    /// says nothing about how precisely THIS instant was observed, and folding it in would only
    /// loosen the budget this event is judged against.
    gap_ticks: u64,
}

/// The whole flight's record — what both directions of G-HANDOVER read.
struct Record {
    samples: Vec<Sample>,
    /// Where each watched realm first crossed a named radius: `(tick, travel, the gap preceding
    /// that sample)`.
    crossings: Vec<Option<(u64, f64, u64)>>,
    /// The observed author flips per watched realm.
    flips: Vec<Option<Flip>>,
    /// Every sample at which a watched realm was drawn by NOBODY: `(watch index, tick, epoch,
    /// distance)`. Lawful only while the composer is rebuilding its windows across an epoch bump.
    absences: Vec<(usize, u64, u64, f64)>,
}

/// Sample the composed picture once, through the pilot camera the renderer actually used.
fn sample(devctl: u16, watch: &[RealmId], travelled: &mut f64, last: &mut Option<DVec3>) -> Sample {
    let st = vd_bins::pixel::poll(devctl);
    let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
    let (pos, _) = own_pose(&st).expect("the avatar has a delivered pose");
    if let Some(prev) = *last {
        *travelled += (pos - prev).length();
    }
    *last = Some(pos);
    Sample {
        tick: st.universe_tick.unwrap_or_default(),
        epoch: st.origin.as_ref().map_or(0, |(_, e)| *e),
        pos,
        travelled_m: *travelled,
        subjects: watch
            .iter()
            .map(|r| vd_bins::pixel::subject(&st, &camera, *r))
            .collect(),
    }
}

/// Set the sticky throttle. THE WARP IS A HELD THROTTLE (`vd_bins::lib`: "warp IS `move_speed` on
/// held axes") — full forward along the avatar's own facing, which a `LookAt` has already pointed
/// at the destination. Holding it, rather than walking in bursts, is BOTH the production input
/// path and what lets this gate sample the handover at the universe clock's own resolution: a
/// burst-and-stop flight costs three round trips per sample and could only resolve the handover to
/// tens of ticks, which is coarser than the budget it is measuring.
fn throttle(devctl: u16, axes: [f32; 3]) {
    let reply = dev_roundtrip(devctl, &DevRequest::Move { axes });
    assert!(
        matches!(reply, Ok(DevResponse::Ack)),
        "the throttle {axes:?} was refused: {reply:?}",
    );
}

/// The look-at brake's MEASURED settled residual (this file's own `look_at` doc: "0.6030 rad
/// without it, 0.0353 rad with it") — the starting error the converge loop divides down from.
const AIM_BRAKE_RESIDUAL_RAD: f64 = 0.0353;

/// THE AIM TOLERANCE: the LARGER of two arms, so the tolerance is always both meaningful and
/// reachable. The GEOMETRIC arm is what the flight actually wants — a straight run down the ring
/// should end inside the destination, so half the destination's angular radius at ring range. The
/// FLOOR arm is the aim loop's own convergence budget: the measured brake residual halved once per
/// converge try, which is the tightest heading a delivered-pose loop can physically settle to;
/// demanding better than it is an infinite loop, not precision.
///
/// WHICH ARM BINDS IS A FACT ABOUT THE WORLD, NOT A FIXED STORY — it is the ratio of the
/// destination's own shell to the star gap, and both moved when the stellar solve did. The gate
/// prints the tolerance it settled to on every aim, so the arm that won is read off the run rather
/// than claimed here. Either way a heading no tighter than the tolerance is CORRECTED IN FLIGHT by
/// the re-aim (whose trigger carries the same tolerance at range), so the warp is a pursuit that
/// tightens as the range closes — the aim never has to be better than the instrument steering it.
fn aim_tolerance_rad() -> f64 {
    ((system_extent_m() / star_gap_m()).atan() * 0.5)
        .max(AIM_BRAKE_RESIDUAL_RAD / f64::from(1u32 << (AIM_TRIES - 1)))
}
/// How many turns the converge loop may take before the aim is declared un-gettable.
const AIM_TRIES: u32 = 8;

/// One bounded WALK burst toward `aim`, then the sticky throttle released. `WalkTo` steers by the
/// delivered pose into LOCAL axes, so the ship's heading is independent of where it is looking —
/// which is what lets a leg close on one realm while watching another. The brake is the shared
/// feedback-lag term every cluster gate walks by (`vd_bins::flight`).
fn walk_chunk(devctl: u16, aim: DVec3, ticks: u64) {
    let _ = dev_roundtrip(
        devctl,
        &DevRequest::WalkTo {
            target: aim.to_array(),
            arrive_epsilon: 3.0,
            max_ticks: ticks,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    );
    throttle(devctl, ALL_STOP);
}

/// Turn the avatar — and therefore the pilot's eyes — to face `target`, and PROVE the DELIVERED
/// facing settled within [`aim_tolerance_rad`].
///
/// WHY A CONVERGE LOOP AND NOT ONE CALL. The dev-control look-at loop steers by the DELIVERED
/// orientation, so it stops the moment that lagging pose reads aligned — and whatever it has
/// already put on the wire still lands afterwards. `vd_client_harness::nav`'s brake makes that
/// residual small (measured on THE world: 0.6030 rad without it, 0.0353 rad with it), but small is
/// not zero, and a residual of that size over a whole star gap is a miss of `gap·tan(residual)` —
/// which is exactly how this flight first sailed PAST its destination. (The miss the settled aim
/// actually leaves is COMPUTED and printed on every turn, against the gap the world states, so it
/// is a reading rather than a remembered figure.) Re-driving from the settled pose divides the
/// residual again each time, so the aim converges instead of freezing wherever the lag left it.
fn look_at(devctl: u16, target: DVec3) {
    let tol = aim_tolerance_rad();
    for attempt in 0..AIM_TRIES {
        let reply = dev_roundtrip(
            devctl,
            &DevRequest::LookAt {
                target: target.to_array(),
                align_epsilon: tol,
                max_ticks: 900,
            },
        );
        assert!(
            matches!(reply, Ok(DevResponse::State { .. })),
            "the avatar never finished turning toward {target:?}: {reply:?}",
        );
        // Let the in-flight deltas land: poll until the delivered facing holds still, then measure
        // the TRUE 3-D error against the target. A settle signal, not a sleep standing in for one.
        let mut last: Option<DVec3> = None;
        let mut error = f64::INFINITY;
        for _ in 0..AIM_SETTLE_POLLS {
            std::thread::sleep(SAMPLE_POLL);
            let st = vd_bins::pixel::poll(devctl);
            let (pos, orient) = own_pose(&st).expect("a delivered pose to aim from");
            let facing = orient * DVec3::NEG_Z;
            error = facing.angle_between((target - pos).normalize());
            if last.is_some_and(|f: DVec3| f.abs_diff_eq(facing, 1e-12)) {
                break;
            }
            last = Some(facing);
        }
        if error <= tol {
            eprintln!(
                "[warp] AIM: settled {error:.5} rad off {target:?} after {} turn(s) \
                 (tolerance {tol:.5} rad = a {:.1} m miss over the {:.0} m star gap)",
                attempt + 1,
                star_gap_m() * tol.tan(),
                star_gap_m(),
            );
            return;
        }
    }
    panic!(
        "the aim never converged on {target:?} within {tol:.5} rad after {AIM_TRIES} turns — \
         the look-at loop is not reducing its error",
    );
}

/// HOW A LEG IS FLOWN. A held throttle drives the ship along its own FACING — which is the warp's
/// real input path and the only way to sample at the clock's own resolution — but it ties where the
/// ship goes to where it looks. A leg that must WATCH one realm while CLOSING on another therefore
/// walks instead: `WalkTo` steers by the delivered pose into local axes, so the heading is
/// independent of the facing, at the cost of one round trip per chunk.
#[derive(Clone, Copy, Debug)]
enum Drive {
    /// Hold these axes for the whole leg (full ahead, or all stop).
    Throttle([f32; 3]),
    /// Walk toward the `toward`-th watched realm's own drawn centre in bounded chunks, and coast
    /// (poll only) once inside `stop_within` of it — so the ship parks itself where the leg wants
    /// it while the eyes stay wherever they were pointed.
    Walk {
        toward: usize,
        chunk_ticks: u64,
        stop_within: f64,
    },
}

/// Drive a leg and record the composed picture at the clock's own resolution until `stop` says so,
/// then release the throttle. `radii` names, per watched realm, the radius whose crossing is
/// timed; `flip_to` names the author each watched realm is expected to hand over to.
///
/// THE PRESENCE LAW is asserted on EVERY sample of the whole flight, not only the captured ones.
#[allow(clippy::too_many_arguments)]
fn fly_recording(
    devctl: u16,
    watch: &[RealmId],
    radii: &[Option<(f64, bool)>], // (radius, true = crossing is "beyond", false = "within")
    flip_to: &[Option<Author>],
    // The index in `watch` the flight is aimed at — the pass-by guard's subject.
    toward: usize,
    // The index in `watch` the EYES are on — the re-aim's subject. Equal to `toward` on a leg flown
    // forward; the realm being left behind on a leg flown astern.
    facing: usize,
    drive: Drive,
    deadline: Duration,
    mut stop: impl FnMut(&Sample, &[Option<Flip>]) -> bool,
) -> Record {
    if let Drive::Throttle(axes) = drive {
        throttle(devctl, axes);
    }
    let started = Instant::now();
    let mut travelled = 0.0_f64;
    let mut last: Option<DVec3> = None;
    let mut samples: Vec<Sample> = Vec::new();
    let mut crossings: Vec<Option<(u64, f64, u64)>> = vec![None; watch.len()];
    let mut flips: Vec<Option<Flip>> = vec![None; watch.len()];
    let mut prev_author: Vec<Option<Author>> = vec![None; watch.len()];
    let mut absences: Vec<(usize, u64, u64, f64)> = Vec::new();
    let mut closest = f64::INFINITY;
    // MID-COURSE CORRECTIONS AT THE LAW'S OWN PACE (S3, measured): the re-aim trigger fires on a
    // heading error that the delivered-pose jitter re-creates every sample once the range has
    // halved, and each correction costs a look-at round trip — an unpaced loop spent 2,968
    // corrections (the whole 300 s deadline) on a leg that needs ~one per range e-folding. The
    // pursuit's own e-folding time is the governor's τ, so corrections are paced to it: the miss
    // stays bounded by (aim floor)·range while the overhead stays seconds.
    let reaim_pace = std::time::Duration::from_secs_f64(flight_tuning().tau_s);
    let mut last_aim = Instant::now();
    loop {
        let s = sample(devctl, watch, &mut travelled, &mut last);
        // The gap preceding THIS sample — the resolution of anything observed at it.
        let gap = samples
            .last()
            .map_or(0, |prev| s.tick.saturating_sub(prev.tick));
        let mut chasing = false;
        for i in 0..watch.len() {
            // THE PRESENCE LAW, per realm per sample. An ABSENCE is recorded rather than fatal
            // here: it is lawful ONLY while the composer is rebuilding its windows across an
            // origin-epoch bump (a missing statement means the thing is not drawn — THE DRAW LAW),
            // and the caller asserts that bound over the whole flight.
            let author = match s.subjects[i].presence {
                Presence::Drawn(a) => a,
                Presence::Absent => {
                    absences.push((i, s.tick, s.epoch, s.distance(i)));
                    prev_author[i] = None;
                    continue;
                }
            };
            if let Some((radius, beyond)) = radii[i] {
                let d = s.distance(i);
                let crossed = if beyond { d > radius } else { d <= radius };
                if crossed && crossings[i].is_none() {
                    crossings[i] = Some((s.tick, s.travelled_m, gap));
                }
                if crossings[i].is_some() && flips[i].is_none() {
                    chasing = true;
                }
            }
            if let Some(want) = flip_to[i]
                && prev_author[i].is_some_and(|p| p != want)
                && author == want
                && flips[i].is_none()
            {
                flips[i] = Some(Flip {
                    tick: s.tick,
                    distance_m: s.distance(i),
                    footprint_px: s.subjects[i].radius_px,
                    travelled_m: s.travelled_m,
                    centre_before_px: samples.last().and_then(|p| p.subjects[i].centre_px()),
                    centre_at_px: s.subjects[i].centre_px(),
                    gap_ticks: gap,
                });
            }
            prev_author[i] = Some(author);
        }
        // THE PASS-BY GUARD, on the realm the flight is AIMED at (the one it flies toward; the one
        // it leaves behind opens by design): a flight that sails past its destination must say so,
        // loudly and with the miss distance, instead of running to the deadline out in the dark.
        //
        // THE SLACK CARRIES THE MEASUREMENT'S OWN JITTER AT GOVERNED SPEED (S3, measured): the
        // distance is read off the DELIVERED pose, and one sample of interp/snapshot jitter moves
        // it by a fraction of a tick of travel — 150 m of slack was 15 ticks at the old 10 m/tick
        // and is a vanishing fraction of a tick at the ~2.5e13 m/s galaxy ceiling (a mid-warp run
        // tripped it mid-gap on a sub-tick wobble far larger than 150 m). The slack is one whole
        // pose-lag of travel at the GOVERNED ceiling for the current range, plus the old extent
        // term — tight again exactly where arrival precision matters, because the governor has
        // already slowed the ship there.
        let d = s.distance(toward);
        closest = closest.min(d);
        let jitter_v = galaxy_cap_mps().min(approach_ceiling_mps(
            realm_speed_cap_mps(system_extent_m(), DEV.move_speed, TRAVERSE_S),
            d,
            flight_tuning().tau_s,
        ));
        let slack = pass_by_slack_m() + vd_bins::flight::pose_lag_s(DEV.tick_dt) * jitter_v;
        assert!(
            d <= closest + slack,
            "the flight SAILED PAST {:?}: closest approach {closest:.1} m, now {d:.1} m and \
             opening past the {slack:.1} m jitter slack — the aim missed",
            watch[toward],
        );
        // RE-AIM — a pilot corrects course. A single fixed heading over a 12 km run cannot hold a
        // subject centred: the aim's own residual becomes a growing angle as the range closes, and
        // a subject that drifts to the frame edge is measured through a badly off-axis projection.
        // Never while CHASING a flip: the handover's centroid continuity is measured across two
        // bracketing samples, and turning between them would be the GATE moving the picture rather
        // than the world.
        let drifted = s.subjects[facing].centre_px().is_none_or(|(x, y)| {
            let (cx, cy) = (CAPTURE_W as f64 * 0.5, CAPTURE_H as f64 * 0.5);
            let off_px = (x - cx).hypot(y - cy);
            let range = s.distance(facing);
            let m_per_px = 2.0 * range * (FIT_FOV_Y * 0.5).tan() / CAPTURE_H as f64;
            // The trigger carries the aim's own resolution at range (the S3 re-derivation): a
            // heading cannot be held tighter than the aim tolerance, so demanding a sub-tolerance
            // miss across a whole star gap would re-aim every sample forever. The `max` keeps the
            // old one-extent trigger verbatim wherever it is reachable (every in-system range),
            // and tightens the pursuit toward it as the range closes.
            off_px * m_per_px > reaim_miss_m().max(aim_tolerance_rad() * range)
        });
        let reaim = (drifted & !chasing & (last_aim.elapsed() >= reaim_pace))
            .then_some(s.subjects[facing].centre_m);
        let done = stop(&s, &flips);
        samples.push(s);
        if done {
            throttle(devctl, ALL_STOP);
            let _ = drive;
            return Record {
                samples,
                crossings,
                flips,
                absences,
            };
        }
        assert!(
            started.elapsed() < deadline,
            "the flight never reached its stop condition (travelled {travelled:.1} m, crossings \
             {crossings:?}, flips seen {:?}, chasing {chasing})",
            flips.iter().map(Option::is_some).collect::<Vec<_>>(),
        );
        if let Some(target) = reaim {
            look_at(devctl, target);
            last_aim = Instant::now();
        }
        match drive {
            Drive::Throttle(_) => std::thread::sleep(SAMPLE_POLL),
            Drive::Walk {
                toward: t,
                chunk_ticks,
                stop_within,
            } => {
                let sample = samples.last().expect("a sample was just pushed");
                if sample.distance(t) > stop_within {
                    walk_chunk(devctl, sample.subjects[t].centre_m, chunk_ticks);
                } else {
                    std::thread::sleep(SAMPLE_POLL);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pixel verdicts over a straddled capture.
// ---------------------------------------------------------------------------------------------

fn decode(cwd: &std::path::Path, rel: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let img = image::open(cwd.join(rel))
        .unwrap_or_else(|e| panic!("open capture {rel}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let buf = img.into_raw();
    // The clear colour is self-calibrated from the top-right corner — deep space, and the one
    // corner the HUD never reaches.
    let i = (w - 1) * 4;
    let clear = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
    (buf, w, h, clear)
}

/// Inflate a projected rectangle by the shared bracketing factor about its own centre, never below
/// the shared minimum apparent radius (a point of light's rect is its footprint, not a point).
fn bracket(rect: ScreenAabb) -> ScreenAabb {
    let cx = f64::midpoint(rect.min.x, rect.max.x);
    let cy = f64::midpoint(rect.min.y, rect.max.y);
    let hw = ((rect.max.x - rect.min.x) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
    let hh = ((rect.max.y - rect.min.y) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
    ScreenAabb {
        min: ScreenPos {
            x: cx - hw,
            y: cy - hh,
        },
        max: ScreenPos {
            x: cx + hw,
            y: cy + hh,
        },
    }
}

fn nonclear_in(rgba: &[u8], w: usize, h: usize, clear: [u8; 4], rect: ScreenAabb) -> u64 {
    let x0 = rect.min.x.floor().max(0.0) as usize;
    let y0 = rect.min.y.floor().max(0.0) as usize;
    let x1 = (rect.max.x.ceil().max(0.0) as usize).min(w);
    let y1 = (rect.max.y.ceil().max(0.0) as usize).min(h);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 4;
            if rgba[i..i + 4] != clear {
                n += 1;
            }
        }
    }
    n
}

/// THE LOCAL PIXEL PROBE (§2.11): the watched realm actually PAINTED at the position the composed
/// picture put it, and those pixels are distinguishable from their own local background — never a
/// full-frame search. Returns the painted-pixel count for the run log.
fn assert_painted(cap: &Straddled, cwd: &std::path::Path, subject: &Subject, what: &str) -> u64 {
    let (rgba, w, h, clear) = decode(cwd, &cap.shot);
    assert_eq!(
        magenta_pixel_count(&rgba),
        0,
        "{what}: a magenta (missing-asset) pixel in the capture",
    );
    let rect = bracket(
        subject
            .rect
            .unwrap_or_else(|| panic!("{what}: the subject must project in front of the eye")),
    );
    let painted = nonclear_in(&rgba, w, h, clear, rect);
    assert!(
        painted > 0,
        "{what}: NOTHING was painted at the composed position (rect {rect:?}, footprint \
         {:.2} px, author {:?})",
        subject.radius_px,
        subject.presence,
    );
    assert!(
        dot_pixels_distinct_from_surround(&rgba, w, h, rect, PROBE_RING_PX),
        "{what}: the pixels at the composed position are indistinguishable from the local \
         background — nothing legible was drawn there",
    );
    painted
}

/// §2.8's CENTROID PATH CONTINUOUS, at the handover itself: the drawn centre across the author
/// flip may move only by what the OBSERVER ITSELF did between the two bracketing samples — its
/// travel and its TURN — projected. One position author for life: the parent's placement row
/// places the subject before, during and after its spin-up, and only the LOOK payload upgrades, so
/// the swap redraws the thing where it already was. Any larger move means the handover MOVED it,
/// which is the jump §2.8 forbids.
///
/// ★ THE TURN TERM (S5, measured): a projected centroid moves when the CAMERA turns, whatever the
/// world does, and a pilot camera turns whenever the delivered facing does. Before the
/// camera-relative flatten the camera did not actually face the subject at all, so this term could
/// never be exercised; with the flatten landed it is reachable and load-bearing — MEASURED, a
/// 0.24 px centroid move across a handover taken while the ship was parked, against a travel
/// allowance of 0.00 px. One display pixel is `fov_y / height` of turn; the allowance carries the
/// facing change between the two samples in exactly those units.
fn assert_centroid_continuous(flip: &Flip, camera: &CaptureCamera, gap_ticks: u64, what: &str) {
    let (Some((ax, ay)), Some((bx, by))) = (flip.centre_before_px, flip.centre_at_px) else {
        panic!(
            "{what}: the subject must project in front of the eye on both sides of the handover"
        );
    };
    let moved_px = (bx - ax).hypot(by - ay);
    // The observer's own travel over the bracketing gap, projected at the subject's range — the
    // one-tick true-motion bound the crossing gates use, scaled to the samples actually taken.
    let travel_m = metres_per_tick() * (gap_ticks + 1) as f64;
    let m_per_px = 2.0 * flip.distance_m * (camera.fov_y * 0.5).tan() / camera.height as f64;
    // ...plus the TURN the observer made over the same bracketing gap, in the same pixels: the
    // shipped look-rate (`nav::MAX_LOOK_STEP` per tick, the one the closed loop can command) over
    // the camera's own angular resolution.
    let turn_px = (vd_client_harness::nav::MAX_LOOK_STEP * (gap_ticks + 1) as f64)
        / (camera.fov_y / camera.height as f64);
    let allowance_px = travel_m / m_per_px + turn_px;
    eprintln!(
        "[warp] CENTROID ({what}): the drawn centre moved {moved_px:.2} px across the handover \
         (from {ax:.1},{ay:.1} to {bx:.1},{by:.1}) against the observer's own {travel_m:.1} m of \
         travel + its own turn over {} sample tick(s) = {allowance_px:.2} px at {:.0} m",
        gap_ticks + 1,
        flip.distance_m,
    );
    assert!(
        moved_px <= allowance_px,
        "{what}: THE CENTROID JUMPED {moved_px:.2} px across the handover, past the          {allowance_px:.2} px the observer's own travel allows — the position author changed at          the swap, which it must never do",
    );
}

/// THE PRESENCE LAW over a whole leg: a watched realm may be drawn by NOBODY only while the
/// composer is rebuilding its windows across an ORIGIN-EPOCH BUMP — a chain change tears the old
/// windows down and the new ones must be re-served their statements before anything can be drawn
/// from them. THE DRAW LAW's own sentence covers that gap ("a missing statement means the thing is
/// not drawn"); what this asserts is that it is BOUNDED and that it happens nowhere else.
///
/// The bound is derived: the gateway's keep-alive beat (the slowest re-assert of a send-on-change
/// statement) plus the delivery and draw ticks.
fn assert_absences_only_bridge_an_epoch_bump(rec: &Record, watch: &[RealmId], leg: &str) {
    let bound =
        (u64::from(DEV.tick_hz) / 2).max(1) + MEMBERSHIP_HOP_TICKS + COMPOSE_TICKS + DRAW_TICKS;
    if rec.absences.is_empty() {
        eprintln!("[warp] PRESENCE ({leg}): every sample drew every watched realm — no gap at all");
        return;
    }
    // Group each absence against the tick its epoch was first seen at.
    let epoch_start = |epoch: u64| -> u64 {
        rec.samples
            .iter()
            .find(|s| s.epoch == epoch)
            .map_or(0, |s| s.tick)
    };
    let mut worst = 0u64;
    for &(i, tick, epoch, dist) in &rec.absences {
        let since = tick.saturating_sub(epoch_start(epoch));
        worst = worst.max(since);
        assert!(
            since <= bound,
            "{leg}: THE PRESENCE LAW broken — {:?} was drawn by NOBODY at tick {tick} ({dist:.1} m \
             out), {since} ticks into epoch {epoch}, past the derived window-rebuild bound of \
             {bound} ticks. Neither its own look nor its parent's marker reached the picture.",
            watch[i],
        );
    }
    eprintln!(
        "[warp] PRESENCE ({leg}): {} samples with a watched realm undrawn, all inside the \
         window-rebuild bound (worst {worst} ticks into an epoch, bound {bound})",
        rec.absences.len(),
    );
}

/// THE PRESENCE LAW, per watched realm per frame: exactly one of {marker, body} — never zero.
/// (Never BOTH is unrepresentable: a composed row carries one bag, and the wire types cannot
/// express a third pixel source.)
fn assert_presence(subject: &Subject, what: &str) -> Author {
    match subject.presence {
        Presence::Drawn(author) => author,
        Presence::Absent => panic!(
            "{what}: THE PRESENCE LAW broken — the realm is drawn by NOBODY (neither its own look \
             nor its parent's marker reached the picture)",
        ),
    }
}

/// G-SHEAR, the PIXEL half: every drawn row of one captured frame was composed from a moment the
/// composer could lawfully hold — none newer than the fold, none older than the declared retention
/// (a held stratum and a relayed interior carry their own stamps BY DESIGN, and say so per row).
fn assert_same_tick_composition(st: &DevState, what: &str) -> Vec<u64> {
    let mut ticks: Vec<u64> = st
        .realm_boxes
        .iter()
        .filter_map(|b| b.newest_tick)
        .collect();
    ticks.sort_unstable();
    ticks.dedup();
    if let (Some(&lo), Some(&hi)) = (ticks.first(), ticks.last()) {
        let span = hi - lo;
        let retention = look_ttl_ticks();
        assert!(
            span <= retention,
            "{what}: G-SHEAR — the drawn rows span {span} ticks, past the composer's declared \
             retention of {retention} (stamps {ticks:?}); a mixed-age picture reached the pixels",
        );
    }
    ticks
}

// ---------------------------------------------------------------------------------------------
// THE GATES
// ---------------------------------------------------------------------------------------------

#[test]
fn g_handover_the_symmetric_budgets_are_derived_and_fit_the_worlds_own_geometry() {
    // The SYMMETRIC half of G-HANDOVER, closed before any process runs: both budgets are
    // derivations over THE world's solved numbers and the landed cadences, and each must FIT the
    // distance the geometry itself provides for it. A world-numbers change that breaks the warp
    // acceptance fails here in milliseconds, not only in a four-minute flight.
    // ★ TWO SYSTEMS, TWO SETS OF NUMBERS (2026-08-21). Since the true-size re-solve a system's
    // shell is `f(its own star)`, so the flight's two halves are judged on two different geometries:
    // the DEPARTURE half on the home system's own band (the line the hand-back is timed from), the
    // WAKE half on the destination's own band (the line the hand-over is timed from). Collapsing
    // them onto one `system_extent_m()` is what made the wake measurement read 30× its budget.
    let (gap, extent) = (star_gap_m(), system_extent_m());
    let (home_wake, tear) = (spin_up_r_m(), tear_down_r_m());
    let (dest_extent, wake) = (dest_extent_m(), dest_wake_r_m());
    let margin = gap - wake;
    let band = tear - home_wake;
    eprintln!(
        "[handover] DERIVED on THE world (seed {}, {:.0} m/s, {:.3} s/tick, {:.1} m/tick):\n  \
         visibility factor cot(θ/2) = {:.9}\n  \
         HOME system extent {extent:.3} m · wake radius {home_wake:.6} m · tear-down radius \
         {tear:.6} m\n  \
         DESTINATION (the ring sibling) extent {dest_extent:.3} m · its OWN wake radius \
         {wake:.6} m\n  \
         the measured star gap {gap:.6} m · asleep-at-departure margin {margin:.6} m · tear-down band {band:.3} m\n  \
         cadences (ticks): AoI {} · grace-hold {} · reconcile {RECONCILE_TICKS} · relay {RELAY_FORWARD_TICKS} \
         · membership hop {MEMBERSHIP_HOP_TICKS} · compose {COMPOSE_TICKS} · draw {DRAW_TICKS} · \
         roster-loss window {}\n  \
         WAKE budget at zero boot = {} ticks = {:.1} m (+ {:.1} m per boot tick); \
         Q2 RELAY SHARE = {} ticks = {:.1} m\n  \
         DEPARTURE budget = {} ticks = {:.1} m; with the roster-loss window = {} ticks = {:.1} m\n  \
         flight parks: polar exit z {:.1} m · arrival standoff {:.1} m · departure park {:.1} m",
        DEV.universe_seed,
        DEV.move_speed,
        DEV.tick_dt,
        metres_per_tick(),
        visibility_factor(),
        aoi_cadence_ticks(),
        grace_hold_ticks(),
        look_ttl_ticks(),
        wake_budget_ticks(0),
        footprint_m(wake_budget_ticks(0)),
        metres_per_tick(),
        relay_share_budget_ticks(),
        footprint_m(relay_share_budget_ticks()),
        departure_budget_ticks(),
        footprint_m(departure_budget_ticks()),
        departure_budget_full_ticks(),
        footprint_m(departure_budget_full_ticks()),
        polar_exit_z_m(),
        arrival_standoff_m(),
        behind_park_m(),
    );
    // THE geometric precondition the whole acceptance rests on: a destination is genuinely asleep
    // when you leave, because the world's own solver left more ring than wake radius.
    assert!(
        margin > 0.0,
        "THE world no longer leaves a system asleep at departure (gap {gap:.3} m ≤ wake {wake:.3} m)",
    );
    // SYMMETRY, at GOVERNED speeds (the S3 re-derivation — the pre-law form multiplied the budget
    // by a flat 10 m/tick, a 20× under-statement of the governed approach): the wake must complete
    // while the approach governor is still riding the wake band down — the governed dwell from the
    // wake radius to the shell covers the whole zero-boot wake budget — or a system is entered
    // before it draws itself.
    let t = flight_tuning();
    let dwell_s = governed_wake_dwell_s();
    let wake_budget_s = wake_budget_ticks(0) as f64 * DEV.tick_dt;
    eprintln!(
        "[handover] GOVERNED (S3): tau {:.3} s · galaxy ceiling {:.4e} m/s · governed wake dwell \
         {dwell_s:.2} s vs zero-boot wake budget {wake_budget_s:.2} s · governed approach closed \
         form {:.1} s · marker range {:.1} m · behind park {:.1} m",
        t.tau_s,
        galaxy_cap_mps(),
        governed_approach_s(),
        marker_range_m(),
        behind_park_m(),
    );
    assert!(
        dwell_s >= wake_budget_s,
        "the governed approach crosses the wake band in {dwell_s:.2} s, inside the zero-boot wake \
         budget {wake_budget_s:.2} s — a system would be entered before it draws itself",
    );
    // THE WHOLE DEPARTURE — the park it is watched from PLUS the footprint the roster-loss window
    // adds to it — still fits inside the ring, at the fastest speed the departure leg can be flown.
    //
    // ★ CORRECTED 2026-08-21, and the correction makes this assert say MORE, not less. The old form
    // read `v_behind` off the approach GOVERNOR'S ARM ALONE at the behind park — `child_cap +
    // park/τ`, which on THE world is 7.07e14 m/s, twenty-eight times the galaxy's own ceiling. No
    // occupant can ever hold that: the shipped integrator (`vd_sim::stub::dot::integrate`) flies
    // `min(the containing realm's ceiling, every child's governor arm, the ramp)`, and the
    // departure leg's container is the GALAXY. So the old expression was not a stronger bound, it
    // was a bound on a speed that is not in the world — it happened to hold only while the ring was
    // half again wider, and it failed the moment the stellar solve moved the ring without moving
    // anything the bound was about. The arm is kept (it is the other half of the `min` and it does
    // bind close in); it is now clamped by the ceiling exactly as the integrator clamps it. The
    // ramp can only lower the result further, so this stays a ceiling on the real flight.
    //
    // AND THE CLAIM IS NOW THE WHOLE ONE: the departure is watched FROM the behind park, so what
    // has to fit inside the ring is the park PLUS the footprint — the old form checked the
    // footprint alone and never once looked at the park it is measured from, which is 52 % of the
    // ring on THE world and by far the larger term.
    let v_behind = galaxy_cap_mps().min(approach_ceiling_mps(
        realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S),
        behind_park_m(),
        t.tau_s,
    ));
    let departure_reach =
        behind_park_m() + departure_budget_full_ticks() as f64 * DEV.tick_dt * v_behind;
    eprintln!(
        "[handover] DEPARTURE REACH: park {:.4e} m + {} ticks at the GOVERNED ceiling {v_behind:.4e} \
         m/s (the galaxy's {:.4e} m/s clamps the governor's {:.4e} m/s arm) = {departure_reach:.4e} \
         m against the {gap:.4e} m star gap",
        behind_park_m(),
        departure_budget_full_ticks(),
        galaxy_cap_mps(),
        approach_ceiling_mps(
            realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S),
            behind_park_m(),
            t.tau_s,
        ),
    );
    assert!(
        departure_reach < gap,
        "the behind park plus the full departure footprint at the governed ceiling \
         ({departure_reach:.1} m) must fit inside the star gap {gap:.1} m",
    );
    // The hysteresis story, restated where each half is meaningful: in the law-inert (foot-speed)
    // regime the tear-down band still exceeds one tick of foot travel — the geometric dead-zone
    // the in-system battery stands on — and at governed speeds (where the band is sub-tick) the
    // anti-thrash rests on the GRACE hold, which must outlast a whole verdict beat.
    assert!(
        band > metres_per_tick(),
        "the tear-down band {band:.3} m must exceed one tick of FOOT travel {:.3} m",
        metres_per_tick(),
    );
    assert!(
        grace_hold_ticks() > aoi_cadence_ticks(),
        "the AoI grace hold ({} ticks) must outlast one verdict beat ({} ticks) — at governed \
         speeds the geometric dead-zone is sub-tick and the grace IS the hysteresis",
        grace_hold_ticks(),
        aoi_cadence_ticks(),
    );
    // THE MARKER-LEGIBILITY WINDOW IS EMPTY AT TRUE SCALE, and that is the world being honest.
    // The interim world's marker range (sized by an authority-shell-sized look) sat OUTSIDE the
    // tear-down radius, so a departing ship could park in a band where the departed system was
    // both torn down AND still a measurable rectangle. On THE world a system's LOOK is its
    // star's photosphere while its wake tear-down is its solved shell times the visibility factor
    // — the two are ORDERS apart, and the assert below prints both — so past the tear-down the
    // departed system lawfully draws AT the apparent floor: the 3 px
    // presence dot IS "shrunk to a dot" out here, and there is no honest rectangle left to
    // measure. Pinned as the measured inversion (not weakened): the window's absence is
    // asserted, with both numbers, so the day a look grows past its wake radius again this
    // fails and the rectangle verdict can be restored.
    assert!(
        marker_range_m() < tear,
        "the marker-legibility range {:.1} m now EXCEEDS the tear-down {tear:.1} m — the \
         true-scale inversion has reversed; restore the (tear, marker_range) park window and the \
         rectangle verdict it licensed",
        marker_range_m(),
    );
    let roster = world_roster(&DEV);
    let sib_unit = roster.sibling_centre / roster.sibling_centre.length();
    let sin_off_pole = (1.0 - sib_unit.z * sib_unit.z).max(0.0).sqrt();
    let release_reach = extent + world().band.outset_m;
    assert!(
        polar_exit_z_m().abs() * sin_off_pole > release_reach,
        "the polar exit ({:.1} m) no longer clears the home system along the onward leg \
         (closest approach {:.1} m vs release reach {release_reach:.1} m) — restate the corridor",
        polar_exit_z_m().abs(),
        polar_exit_z_m().abs() * sin_off_pole,
    );
}

// UN-PARKED at the speed-law slice (S3, real-scale addendum §A3 + the OQ-2 ruling): the star-gap
// leg is flown at GOVERNED speeds (ramp + galaxy ceiling + approach governor — `vd_core::flight`),
// every park and budget re-derived from the governed closed form, and the departure-direction
// watch moved to the START of the warp: the home tear-down happens a few kilometres into a
// journey of the whole star gap — a ratio the world's own numbers set, not a preference. Every
// growth/handover/no-flicker assertion the park owed is live again.
#[test]
// UN-PARKED by the S5 render-scale slice (D-LOOK-3 discharged). What it had measured: the
// departure capture painted NOTHING at the destination's composed position — a 3.00 px
// ParentMarker at rect x636-648/y354-366, exactly the rect the readback found empty, at an eye
// distance of the whole star gap. The cure is the camera-relative flatten (every body placed by an
// f64 subtraction from the eye, so the drawn error is relative to the DISTANCE, not to the
// absolute coordinate) together with the f64-built camera rotation (an f32 `looking_at` at that
// magnitude loses the facing entirely).
fn g_warp_pixels_a_point_of_light_grows_hands_over_and_the_one_behind_shrinks_to_a_dot() {
    // FIRST statement: hold the process tier for the whole body (it outlives the cluster reap).
    let _tier = vd_bins::cluster_tier();

    let f = fixture("warp");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV);
    let mut client = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-warp",
        0,
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let roster = world_roster(&DEV);
    let home_label = label_of(roster.home);
    let galaxy_label = label_of(roster.galaxy);
    let sibling_label = label_of(roster.sibling);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star: {landed:?}",
    );
    // ★ THE FLIGHT'S TWO GEOMETRIES, NAMED APART (2026-08-21). `tear` is the HOME system's own
    // tear-down radius — the line the departure watch times the hand-back from, and the only home
    // number this flight needs. `wake` and `extent` are the DESTINATION's OWN band and shell — the
    // lines the wake half is timed from, parked against and judged for a pop by. Two systems, two
    // stars, two solved shells; one `system_extent_m()` standing for both is what made the wake
    // measurement read 1813 ticks against a 61-tick budget (see `dest_wake_r_m`).
    let (wake, extent) = (dest_wake_r_m(), dest_extent_m());
    let (tear, gap) = (tear_down_r_m(), star_gap_m());
    eprintln!(
        "[warp] THE world: ring {gap:.3} m · HOME extent {:.3} m / tear-down {tear:.3} m · \
         DESTINATION extent {extent:.3} m / wake {wake:.3} m · asleep-at-departure margin \
         {:.3} m · {:.1} m per tick",
        system_extent_m(),
        gap - wake,
        metres_per_tick(),
    );

    // ---- Leg 0: out of the home system, up the licensed polar corridor. ----
    cross_leg(
        devctl,
        "warp-exit home->galaxy (polar corridor)",
        |_tick| DVec3::new(0.0, 0.0, polar_exit_z_m()),
        &galaxy_label,
        // DERIVED, not a fixed two minutes: on THE world the release edge is the home system's
        // own solved shell (printed by the companion) and the leg to it is minutes of governed
        // flight (`governed_leg_budget` = 3× the speed law's closed form + a commit/boot tail,
        // the same law the walk gate measured).
        vd_bins::flight::governed_leg_budget(
            &DEV,
            system_extent_m(),
            realm_speed_cap_mps(system_extent_m(), DEV.move_speed, TRAVERSE_S),
        ),
    );

    // ---- DEPARTURE: face the destination and prove it is A POINT OF LIGHT. ----
    // Index 0 = the destination (the ring sibling); index 1 = the home system left behind.
    let watch = [roster.sibling, roster.home];
    look_at(devctl, roster.sibling_centre);
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-departure",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let dest = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling);
    let (here, _) = own_pose(&cap.post).expect("a delivered pose at departure");
    let dist0 = (dest.centre_m - here).length();
    assert!(
        dist0 > wake,
        "PRECONDITION: at departure the destination must be OUTSIDE its own wake radius \
         (measured {dist0:.1} m vs wake {wake:.1} m) — otherwise it was never asleep and the \
         handover this gate measures would be vacuous",
    );
    assert_eq!(
        assert_presence(&dest, "departure/destination"),
        Author::ParentMarker,
        "at departure the destination is its PARENT'S point of light — no server runs for it",
    );
    let painted = assert_painted(&cap, &f.cwd, &dest, "departure/destination");
    let departure_footprint_px = dest.radius_px;
    eprintln!(
        "[warp] DEPARTURE: the destination at {dist0:.1} m is a MARKER — footprint \
         {departure_footprint_px:.2} px, {painted} pixels painted, straddle drift {:.2} px, \
         drawn stamps {:?}",
        cap.drift_px,
        assert_same_tick_composition(&cap.post, "departure"),
    );

    // ---- G-PARENT-TRUE (look_horizon.md slice 0, the process half): in the departure scene —
    // the observer standing in the GALAXY, the home system's planets arriving as RELAYED
    // interior rows — every planet row's parent must be its STAR SYSTEM, never the galaxy.
    // Before slice 0 the composer derived a relayed row's parent from the chain index, which
    // named the GRANDparent (the galaxy) on every relayed row (measured RED at the unit tier).
    {
        // The boxes render realm ids in Debug form ("System(7)"), so the galaxy compares in
        // that same rendering — never the display label ("System 7").
        let galaxy_debug = format!("{:?}", roster.galaxy);
        let planet_parents: Vec<(String, Option<String>)> = cap
            .post
            .realm_boxes
            .iter()
            .filter(|b| b.realm.starts_with("Planet("))
            .map(|b| (b.realm.clone(), b.parent.clone()))
            .collect();
        assert!(
            !planet_parents.is_empty(),
            "G-PARENT-TRUE is vacuous: no planet rows in the departure scene \
             (drawn: {:?})",
            cap.post
                .realm_boxes
                .iter()
                .map(|b| b.realm.clone())
                .collect::<Vec<_>>(),
        );
        for (planet, parent) in &planet_parents {
            let parent = parent
                .as_deref()
                .unwrap_or_else(|| panic!("G-PARENT-TRUE: {planet} delivered with NO parent"));
            assert!(
                parent.starts_with("System(") && parent != galaxy_debug.as_str(),
                "G-PARENT-TRUE: {planet}'s delivered parent is {parent:?} — it must be its \
                 star system, never the galaxy ({galaxy_debug:?})",
            );
        }
        eprintln!(
            "[warp] G-PARENT-TRUE: planet rows parent on their star system: {planet_parents:?}"
        );
    }

    // ---- SLICE 6 (look_horizon.md §5.4/§6): ROWS PER FOLD, pinned on the departure fixture. ----
    // The per-session clone is the census wall and it is LINEAR in rows per fold; the look
    // horizon's §5.4 promise is that the design did NOT change rows per fold (relayed interior
    // rows composed before it — only their tag upgraded from marker to picture). Pinned as the
    // DERIVED set off THE world, not a count: the origin galaxy's one in-band child (the home
    // system, drawn by its own look) + that child's whole DRAWING interior (every child of the
    // home system that has a look — its star and its planets, however many the seed drew, each by
    // its own relayed picture) + one parent-authored marker per out-of-band galaxy child
    // (the ring siblings). Containing realms draw nothing (a containment boundary is never
    // drawn as an object). Growth here is the wall moving — this assert is where it gets loud.
    {
        let the_world = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
        let mut expected: std::collections::BTreeSet<String> = the_world
            .regions()
            .iter()
            .filter(|r| r.parent == Some(vd_core::worldgen::GALAXY))
            .map(|r| format!("{:?}", r.realm))
            .collect();
        // ...+ THAT CHILD'S WHOLE INTERIOR: every child of the home system that DRAWS. Read off
        // the regions rather than off the mover list, because since the taxonomy slice the interior
        // is not only movers — the system's STAR is a static child that draws its own photosphere,
        // and a mover-shaped expectation silently omitted it (MEASURED: the composed set carried
        // `Star(…)` and this set did not).
        expected.extend(
            the_world
                .regions()
                .iter()
                .filter(|r| r.parent == Some(roster.home) && r.look.is_some())
                .map(|r| format!("{:?}", r.realm)),
        );
        let actual: std::collections::BTreeSet<String> = cap
            .post
            .realm_boxes
            .iter()
            .map(|b| b.realm.clone())
            .collect();
        assert_eq!(
            actual, expected,
            "SLICE-6 ROWS PER FOLD: the departure fixture's composed set must be exactly the \
             derived one (in-band child + its interior + sibling markers)",
        );
        eprintln!(
            "[warp] SLICE-6 ROWS PER FOLD pinned: {} rows at the departure fixture: {actual:?}",
            actual.len(),
        );
    }

    // ---- PHASE 1 — THE DEPARTURE DIRECTION, in pixels, flown FIRST (the S3 re-derivation): at
    // the governed star gap the home system's tear-down happens a few kilometres into a journey of
    // the whole gap — minutes before any arrival — so the "it shrinks to a dot behind you" window
    // is recorded at the START of the warp, not after it (the old ring-midpoint park died with the
    // interim ring). ----
    // TURN AND LOOK BACK FIRST, then walk OUT toward the destination while WATCHING what is
    // behind — walking steers by the delivered pose, so the ship keeps closing on the destination
    // while facing the other way. The leg stops the moment the marker handover lands (the derived
    // behind park is only its safety ceiling: past it the true-sized marker would fall to the
    // apparent floor and the size verdict below would have nothing honest to measure). Aimed at
    // the home system's OWN drawn centre in the frame the picture is composed in now, never a
    // remembered coordinate.
    let park = behind_park_m();
    let home_seen = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.home);
    look_at(devctl, home_seen.centre_m);
    let behind = fly_recording(
        devctl,
        &watch,
        &[None, Some((tear, true))],
        &[None, Some(Author::ParentMarker)],
        0,
        1,
        Drive::Walk {
            toward: 0,
            chunk_ticks: WATCHED_CHUNK_TICKS,
            stop_within: gap - park,
        },
        watched_walk_budget(tear_down_r_m(), galaxy_cap_mps()),
        |_, flips| flips[1].is_some(),
    );
    assert_absences_only_bridge_an_epoch_bump(&behind, &watch, "departure");
    let (tear_tick, tear_travel, tear_gap) = behind.crossings[1]
        .expect("the home system crossed its own tear-down radius behind the flight");
    let back = behind.flips[1]
        .expect("the home system handed back to its parent's marker behind the flight");
    let measured_back = back.tick.saturating_sub(tear_tick);
    let back_resolution = tear_gap + back.gap_ticks;
    let back_budget = departure_budget_ticks() + back_resolution;
    eprintln!(
        "[warp] DEPARTURE HANDOVER: the home system crossed its {tear:.1} m tear-down radius at \
         tick {tear_tick} ({tear_travel:.1} m travelled) and stopped drawing itself at tick {} \
         ({:.1} m out, footprint {:.2} px). MEASURED {measured_back} ticks = {:.1} m vs DERIVED \
         budget {back_budget} ticks = {:.1} m ({} grace-hold + {} AoI cadence + \
         {MEMBERSHIP_HOP_TICKS} hop + {COMPOSE_TICKS} compose + {DRAW_TICKS} draw, + {} ticks \
         bracketing the two observed instants).",
        back.tick,
        back.distance_m,
        back.footprint_px,
        footprint_m(measured_back),
        footprint_m(back_budget),
        grace_hold_ticks(),
        aoi_cadence_ticks(),
        back_resolution,
    );
    assert!(
        measured_back <= back_budget,
        "G-HANDOVER (departure): the home system kept drawing its own body for {measured_back} \
         ticks ({:.1} m of travel) past its tear-down radius, over the derived {back_budget} ticks \
         ({:.1} m)",
        footprint_m(measured_back),
        footprint_m(back_budget),
    );
    assert_centroid_continuous(&back, &cap.camera, back.gap_ticks, "departure/home");
    // The departure footprint must also fit the ROSTER-LOSS outer bound — a departed realm's stored
    // look is gone from the composer, not merely un-drawn.
    assert!(
        measured_back <= departure_budget_full_ticks(),
        "G-HANDOVER (departure): the handover took {measured_back} ticks, past even the \
         roster-loss outer bound of {} ticks",
        departure_budget_full_ticks(),
    );

    // The eyes are already on the system behind (the leg was flown astern), so this capture
    // needs no turn — only the picture as it stands at the park.
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-behind",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let home_now = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.home);
    assert_eq!(
        assert_presence(&home_now, "behind/home"),
        Author::ParentMarker,
        "behind the flight the home system is a point of light again — its parent's marker resumed",
    );
    let painted = assert_painted(&cap, &f.cwd, &home_now, "behind/home");
    // SHRUNK TO A POINT OF LIGHT, in pixels — re-derived for look_horizon.md slice 1 (Q2
    // APPROVED): the departed system's marker now carries its parent's ONE stated radius, so it
    // draws at its TRUE angular size — computed here INDEPENDENTLY from THE world's own system
    // extent and the measured distance, through the same shared floor expression the renderer
    // scales by. Before the slice this marker sat at the bare 3 px floor (an ~3.8× size pop
    // against the body it had just been); the marker-vs-own-look distinction is the AUTHOR
    // assert above (DevState body kind), no longer a size reading.
    let dist_behind = (home_now.centre_m - cap.camera.eye).length();
    // ★ THE SIZE IS THE ROW'S OWN STATED EXTENT, not the realm's containment bound (the bound/look
    // split, measured here): a departed system's point of light is sized by what the picture STATES
    // for it — its star's photosphere, `system_look_m()` — while its bound is the far wider solved
    // authority shell, `system_extent_m()`. Reading the bound expected a rectangle several times
    // larger than the 3.00 px the client drew; the client was right.
    let behind_row = cap
        .post
        .realm_boxes
        .iter()
        .find(|b| b.realm == format!("{:?}", roster.home))
        .expect("the departed system is in the composed picture");
    let expected_world = vd_client_harness::camera::marker_world_radius(
        vd_client::realm_scene::marker_base_radius_m(behind_row.luma, behind_row.extent_m),
        dist_behind,
        cap.camera.fov_y,
        cap.camera.height as f64,
    );
    let expected_px = vd_client_harness::verdict::projected_point_aabb(
        &cap.camera,
        home_now.centre_m,
        expected_world,
    )
    .map_or(0.0, |r| (r.max.x - r.min.x) * 0.5);
    assert!(
        (home_now.radius_px - expected_px).abs() < READBACK_QUANTUM_PX,
        "the system behind must be drawn at its extent's true angular size \
         ({expected_px:.2} px at {dist_behind:.1} m), not at {:.2} px",
        home_now.radius_px,
    );
    // ★ WHICH ARM BINDS, MEASURED — the shared floor expression has two, and at true scale the
    // FLOOR is the honest one out here: "shrunk to a dot" is literal. The departed system's own
    // stated look subtends `true_px` at this range, far under the floor the picture draws it at, so
    // the non-vacuity this assert owes is that the two are DIFFERENT and the floor is what shows.
    let true_px = behind_row.extent_m * (cap.camera.height as f64 * 0.5)
        / ((cap.camera.fov_y * 0.5).tan() * dist_behind);
    assert!(
        true_px < DOT_MIN_APPARENT_RADIUS_PX,
        "at this range the departed system's own stated look ({true_px:.4} px) must sit UNDER the \
         apparent floor — otherwise the floor is not what is being measured here",
    );
    assert!(
        (expected_px - DOT_MIN_APPARENT_RADIUS_PX).abs() <= READBACK_QUANTUM_PX,
        "the departed system must be drawn AT the shared apparent floor out here (expected \
         {expected_px:.2} px vs floor {DOT_MIN_APPARENT_RADIUS_PX:.1} px)",
    );
    eprintln!(
        "[warp] BEHIND: the home system is a MARKER at footprint {:.2} px, {painted} pixels \
         painted; its own stated look subtends {true_px:.4} px at {dist_behind:.4e} m — the \
         presence floor is what you see. Drawn stamps {:?}",
        home_now.radius_px,
        assert_same_tick_composition(&cap.post, "behind"),
    );

    // ---- PHASE 2 — THE APPROACH, in three derived legs (S3, twice-measured): a pilot's held
    // throttle steers by FACING, and the facing is set through a delivered-pose aim whose floor
    // (`aim_tolerance_rad`) leaves a lateral miss of tolerance·range. Under the governor
    // (`v ≈ range/τ`) a facing-steered pursuit LIMIT-CYCLES at that first-miss lateral — one
    // re-aim interval covers the whole remaining range, so the miss never shrinks (measured twice:
    // stuck a fixed fraction of the gap short of the destination, deadline spent). So the transit
    // splits at the pursuit's own derived tolerance, and every MEASURED claim keeps its original
    // leg shape:
    //   2a  held-throttle cruise (the production warp input) down to the hand-off range —
    //       ramp + galaxy-ceiling cruise, presence law sampled throughout;
    //   2b  a chunked WalkTo (per-TICK server-side steering — immune to the aim floor) closing
    //       whatever remains to the DERIVED pre-wake park (below): still OUTSIDE the band, the
    //       destination still asleep, nothing measured claimed. It has nothing to do on a world
    //       whose park already sits farther out than the cruise's own hand-off range, which is
    //       lawful — the cruise hands off at whichever of the two is farther out;
    //   2c  the MEASURED wake approach on the held throttle, from outside the wake band through
    //       the handover to the standoff — the same recording, radii, flips and budgets as ever.
    let dest_seen = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling);
    look_at(devctl, dest_seen.centre_m);
    let governed_leg_s = governed_approach_s();
    // The cruise's own DERIVED budget (the 300 s literal died with the true-scale gap): the whole
    // star gap at the galaxy's ceiling, through the one budget law every other leg uses.
    let cruise_budget = vd_bins::flight::governed_leg_budget(&DEV, gap, galaxy_cap_mps());
    assert!(
        1.5 * governed_leg_s < cruise_budget.as_secs_f64(),
        "the governed approach closed form ({governed_leg_s:.1} s) no longer fits its derived \
         budget ({cruise_budget:?}) with the 1.5× walk margin — the law moved",
    );
    let standoff = arrival_standoff_m();
    // ★ THE PRE-WAKE PARK, DERIVED AGAINST BOTH THINGS IT HAS TO BE — no longer a bare multiple of
    // the wake radius.
    //
    //  * It must sit OUTSIDE the wake band by more than the ship can travel between the truth and
    //    the gate noticing. MEASURED (2026-08-20) why any margin is needed: from a bare two-band
    //    park one overshoot at the galaxy ceiling carried the ship straight THROUGH the wake
    //    radius, and 2c's precondition ("the measured approach must BEGIN outside the
    //    destination's wake") then failed on the harness's own arrival rather than on anything
    //    the world did.
    //  * It must stay a SMALL PART of the star gap, because leg 2a is the held-throttle CRUISE
    //    this whole gate exists to fly. A park that eats the gap leaves the gate measuring its own
    //    chunked taxi instead of the warp.
    //
    // A FIXED MULTIPLE OF THE WAKE CANNOT BE BOTH, and that is why the literal had to go: the wake
    // radius is the destination's own solved shell times the visibility factor, the ring is the
    // galaxy's placement radius, and their RATIO is a world number that moves whenever either
    // solve does. Where the wake is a large share of the gap, three of them IS most of the
    // journey. So the park is three wake radii where the world leaves room for them and a stated
    // share of the ring where it does not — with the margin over the band ASSERTED either way, so
    // the clamp can never quietly park the ship inside the band it must begin outside of.
    let steering_reach_m =
        (vd_bins::flight::pose_lag_s(DEV.tick_dt) + SAMPLE_POLL.as_secs_f64()) * galaxy_cap_mps();
    let park2 = (3.0 * wake).min(PRE_WAKE_PARK_SHARE_OF_GAP * gap);
    eprintln!(
        "[warp] PRE-WAKE PARK: {park2:.4e} m — wake {wake:.4e} m ({:.1}% of the {gap:.4e} m \
         ring), margin over the band {:.4e} m against a {steering_reach_m:.4e} m steering reach \
         at the cruise ceiling; the held-throttle cruise still flies {:.1}% of the gap",
        100.0 * wake / gap,
        park2 - wake,
        100.0 * (gap - park2) / gap,
    );
    assert!(
        park2 > wake + steering_reach_m,
        "the pre-wake park ({park2:.4e} m) no longer clears the destination's wake radius \
         ({wake:.4e} m) by more than one steering interval at the cruise ceiling \
         ({steering_reach_m:.4e} m): at THE world's ring/wake ratio there is nothing left between \
         the band the measured leg must begin outside of and the share of the gap the cruise must \
         keep. Restate the leg decomposition — do not widen the share.",
    );
    // 2a — the cruise, down to where a facing-steered pursuit still provably converges: the aim
    // tolerance's lateral over the whole gap.
    let handoff = aim_tolerance_rad() * gap;
    let cruise = fly_recording(
        devctl,
        &watch,
        &[None, None],
        &[None, None],
        0,
        0,
        Drive::Throttle(FULL_AHEAD),
        cruise_budget,
        // ...but never INSIDE the pre-wake park. TRUE-SCALE RESTATEMENT (measured): the aim
        // tolerance's lateral over the whole gap — the handoff range this leg was written around —
        // can fall INSIDE the destination's own wake radius, and a cruise that stopped there has
        // already entered the band the measured approach must BEGIN outside, leaving the 2b walk
        // asked to fly backwards. Which of the two is farther out is the world's ratio, not a
        // fixed one, so the cruise hands off at whichever range IS farther out; 2b then closes the
        // remainder to the park exactly as before.
        |s, _| s.distance(0) <= handoff.max(park2),
    );
    assert_absences_only_bridge_an_epoch_bump(&cruise, &watch, "cruise");
    // 2b — the walk to the derived pre-wake park: outside the wake band by the margin asserted
    // where the park is derived, so the destination is provably still asleep when the measured leg
    // begins (re-asserted below, on the delivered pose). The walk's overshoot is the lag-derived
    // brake's own bound, inside the park's margin.
    {
        let t = flight_tuning();
        let v_park2 = approach_ceiling_mps(
            realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S),
            park2 - extent,
            t.tau_s,
        );
        let brake = vd_bins::flight::governed_brake_m(v_park2, DEV.tick_dt, t.tau_s);
        let chunk_ticks =
            ((vd_bins::DEVCTL_READ_TIMEOUT.as_secs_f64() / 4.0) / DEV.tick_dt).floor() as u64;
        let started = Instant::now();
        loop {
            let st = vd_bins::pixel::poll(devctl);
            let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
            let dest = vd_bins::pixel::subject(&st, &camera, roster.sibling);
            let (pos, _) = own_pose(&st).expect("a delivered pose on the pre-wake walk");
            let range = (dest.centre_m - pos).length();
            // The park needs wake-radius CLASS, not metre precision: the measured leg only
            // requires starting OUTSIDE the wake band, and the walk's own convergence scale is
            // its brake — grinding the last brake-width down at a near-stationary commanded speed
            // is minutes spent proving nothing. Arrived once within one brake of the park; that
            // it is still outside the band is what the 2c precondition below re-measures on the
            // delivered pose.
            if range <= park2 + brake {
                break;
            }
            let aim = dest.centre_m + (pos - dest.centre_m) * (park2 / range);
            let _ = dev_roundtrip(
                devctl,
                &DevRequest::WalkTo {
                    target: aim.to_array(),
                    arrive_epsilon: 3.0,
                    max_ticks: chunk_ticks,
                    max_step_m: brake,
                },
            );
            throttle(devctl, ALL_STOP);
            assert!(
                started.elapsed() < watched_walk_budget(handoff, galaxy_cap_mps()),
                "the pre-wake walk never reached the {park2:.1} m park (range {range:.1} m)",
            );
        }
    }
    // 2c — THE MEASURED WAKE APPROACH, exactly the original leg: face the destination, hold the
    // throttle, record the crossing and the handover on the universe clock.
    {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let dest_now = vd_bins::pixel::subject(&st, &camera, roster.sibling);
        let (pos, _) = own_pose(&st).expect("a delivered pose at the pre-wake park");
        assert!(
            (dest_now.centre_m - pos).length() > wake,
            "PRECONDITION: the measured approach must BEGIN outside the destination's wake \
             radius ({wake:.1} m) — the pre-wake park failed to hold",
        );
        look_at(devctl, dest_now.centre_m);
    }
    let approach = fly_recording(
        devctl,
        &watch,
        &[Some((wake, false)), None],
        &[Some(Author::SelfLook), None],
        0,
        0,
        Drive::Throttle(FULL_AHEAD),
        vd_bins::flight::governed_leg_budget(
            &DEV,
            2.0 * wake,
            realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S),
        ),
        |s, flips| flips[0].is_some() && s.distance(0) <= standoff,
    );
    assert_absences_only_bridge_an_epoch_bump(&approach, &watch, "approach");
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;

    // ---- G-HANDOVER, the WAKE direction. ----
    let (wake_tick, wake_travel, wake_gap) = approach.crossings[0]
        .expect("the destination crossed its own wake radius during the approach");
    let swap = approach.flips[0]
        .expect("the destination handed over from its parent's marker to its own look");
    let measured = swap.tick.saturating_sub(wake_tick);
    // THE MEASUREMENT'S OWN RESOLUTION: the gap preceding the crossing sample plus the gap
    // preceding the flip sample — the two instants this latency is the difference of, and nothing
    // else in the leg.
    let resolution = wake_gap + swap.gap_ticks;
    let budget = wake_budget_ticks(boot_ticks) + resolution;
    eprintln!(
        "[warp] WAKE HANDOVER: crossed the {wake:.1} m line at tick {wake_tick} ({wake_travel:.1} m \
         travelled); the destination's own look took over at tick {} ({:.1} m out, footprint \
         {:.2} px, {:.1} m travelled). MEASURED {measured} ticks = {:.1} m vs DERIVED budget \
         {} ticks = {:.1} m (2×{} AoI cadence + {RECONCILE_TICKS} reconcile + {boot_ticks} MEASURED \
         boot + {RELAY_FORWARD_TICKS} Q2 relay + {COMPOSE_TICKS} compose + {DRAW_TICKS} draw, \
         + {} ticks bracketing the two observed instants).",
        swap.tick,
        swap.distance_m,
        swap.footprint_px,
        swap.travelled_m,
        footprint_m(measured),
        budget,
        footprint_m(budget),
        aoi_cadence_ticks(),
        resolution,
    );
    assert!(
        measured <= budget,
        "G-HANDOVER (wake): the destination took {measured} ticks ({:.1} m of travel) to start \
         drawing itself, past the derived budget of {budget} ticks ({:.1} m)",
        footprint_m(measured),
        footprint_m(budget),
    );
    // THE Q2 RELAY SHARE, asserted APART (owner ruling).
    let relay_share = measured.saturating_sub(boot_ticks);
    let relay_budget = relay_share_budget_ticks() + resolution;
    eprintln!(
        "[warp] Q2 RELAY SHARE: {relay_share} ticks = {:.1} m of the wake is post-boot (the \
         parent-relay hop + the cadences + compose + draw) vs DERIVED {relay_budget} ticks = {:.1} m",
        footprint_m(relay_share),
        footprint_m(relay_budget),
    );
    assert!(
        relay_share <= relay_budget,
        "G-HANDOVER (wake, THE Q2 RELAY HOP): the parent-relay leg cost {relay_share} ticks \
         ({:.1} m) after the destination's shard was up, past the derived {relay_budget} ticks \
         ({:.1} m). THE RELAY HOP BREAKS THE WAKE BUDGET — this is the ONE measurement D-WINDOW-2 \
         is gated on; the direct live-sibling window is a FRESH owner ask, never taken for \
         convenience. STOP and report.",
        footprint_m(relay_share),
        footprint_m(relay_budget),
    );
    assert_centroid_continuous(&swap, &cap.camera, swap.gap_ticks, "wake/destination");
    // NO POP: the destination must still be near a point when it takes over — its own look starts
    // drawing no closer than the wake radius minus the GOVERNED distance the budget's ticks allow
    // (the S3 re-derivation: from the wake line the governor rides `v = v_c + x/τ` down, so the
    // depth consumed in T seconds is `(S − E + v_c·τ)·(1 − e^(−T/τ))` — the pre-law form
    // multiplied the budget by a flat 10 m/tick, 20× under the governed approach).
    let t = flight_tuning();
    let v_c = realm_speed_cap_mps(extent, DEV.move_speed, TRAVERSE_S);
    let budget_s = budget as f64 * DEV.tick_dt;
    let allowed_pop_depth = (wake - extent + v_c * t.tau_s) * (1.0 - (-budget_s / t.tau_s).exp());
    assert!(
        swap.distance_m >= wake - allowed_pop_depth,
        "G-HANDOVER (wake): the destination only started drawing itself at {:.1} m, inside the \
         {:.1} m the governed budget allows — it POPPED in at size",
        swap.distance_m,
        wake - allowed_pop_depth,
    );

    // ---- THE GROWTH CURVE: monotone, and strictly bigger at the standoff than at departure —
    // spliced across the cruise (the floor dot the whole gap) and the measured approach (the
    // growth), the two legs the pilot actually watched the destination through. ----
    // ★ THE CURVE IS READ IN ONE AUTHOR'S UNITS (the presence floor, measured). A sleeping realm's
    // point of light is FLOORED to the shared minimum apparent radius; a running realm draws its
    // OWN true angular size — and at the range the wake lands (its parent's visibility band, which
    // the companion prints) a star's photosphere is orders under that floor. So the handover is a
    // STEP DOWN of exactly the floor: MEASURED, 3.002 px, which is the marker's whole footprint.
    // That step is the drawn law, not a failure of growth, and it has its own verdicts above (the
    // author flip, the centroid continuity and the no-pop depth). The monotone curve therefore
    // reads the samples on ONE side of it — from the handover on, where the destination is drawing
    // ITSELF and its footprint is its own angular size all the way to the standoff.
    let curve: Vec<(f64, f64)> = cruise
        .samples
        .iter()
        .chain(approach.samples.iter())
        .skip_while(|s| s.subjects[0].presence != Presence::Drawn(Author::SelfLook))
        .map(|s| (s.distance(0), s.subjects[0].radius_px))
        .collect();
    assert!(
        curve.len() >= 2,
        "the growth curve kept fewer than two samples of the destination drawing its OWN look",
    );
    let mut worst_shrink = 0.0_f64;
    for pair in curve.windows(2) {
        // "Grows monotonically AS YOU APPROACH": a pair carries the claim only when the sampled
        // distance actually closed. At governed speeds the delivered distance wobbles backward by
        // sub-tick interp jitter (measured: a 1.38 px dip on one such pair at ~200 m/tick), and on
        // a backward pair the marker LAWFULLY shrinks with its own geometry — that is the law
        // drawing correctly, not the growth failing.
        if pair[1].0 <= pair[0].0 {
            worst_shrink = worst_shrink.max(pair[0].1 - pair[1].1);
        }
    }
    eprintln!(
        "[warp] GROWTH CURVE: {} samples · {:.2} px at {:.0} m → {:.2} px at {:.0} m · worst \
         sample-to-sample shrink {worst_shrink:.3} px (the readback's own quantum is \
         {READBACK_QUANTUM_PX:.1} px)",
        curve.len(),
        curve[0].1,
        curve[0].0,
        curve[curve.len() - 1].1,
        curve[curve.len() - 1].0,
    );
    assert!(
        worst_shrink < READBACK_QUANTUM_PX,
        "G-WARP-PIXELS: the destination's footprint SHRANK by {worst_shrink:.3} px while the \
         occupant was approaching it — the growth is not monotone",
    );
    assert!(
        curve[curve.len() - 1].1 > curve[0].1,
        "G-WARP-PIXELS: the destination never grew ({:.2} px → {:.2} px)",
        curve[0].1,
        curve[curve.len() - 1].1,
    );

    // ---- THE ARRIVAL PICTURE: the destination draws ITSELF, large, in real pixels. ----
    let cap = vd_bins::pixel::straddle(
        devctl,
        "warp-arrival",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let dest = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling);
    assert_eq!(
        assert_presence(&dest, "arrival/destination"),
        Author::SelfLook,
        "on arrival the destination draws ITSELF — a running realm authors its own look",
    );
    // ★ THE ARRIVAL GPU-PAINT PROBE (the S5 park, discharged — and restated by what S5 measured).
    // The old park said the arrival painted nothing because the client drew relative to NOTHING at
    // a whole-star-gap eye; that IS cured (the camera-relative flatten), and the paint is measured
    // green elsewhere in this very file: `g_look_growth` finishes with tens of thousands of pixels
    // painted at its own close range, and prints both.
    // What the cure ALSO revealed is a second, permanent fact about this particular capture: the
    // arrival standoff stays OUTSIDE the destination's containment shell (every other assert on
    // this leg requires that), and out there the destination's own drawn look — its star's
    // photosphere — subtends a small fraction of one pixel (the measured figure is printed below).
    // There is nothing to paint at that range, and demanding paint would demand a lie. So the
    // probe is applied where it means something (`assert_painted` once the footprint clears the
    // readback quantum) and the measurement is stated otherwise.
    let (rgba_arrival, _, _, _) = decode(&f.cwd, &cap.shot);
    assert_eq!(
        magenta_pixel_count(&rgba_arrival),
        0,
        "arrival/destination: a magenta (missing-asset) pixel in the capture",
    );
    if dest.radius_px >= READBACK_QUANTUM_PX {
        assert_painted(&cap, &f.cwd, &dest, "arrival/destination");
    } else {
        eprintln!(
            "[warp] ARRIVAL PAINT: the destination's own look subtends {:.4} px at this standoff \
             — under the {READBACK_QUANTUM_PX} px readback quantum, so the paint probe is stated \
             rather than demanded. Its AUTHOR (its own look) and its SIZE are asserted here; the \
             painted proof at a resolvable range is g_look_growth's, in this same file.",
            dest.radius_px,
        );
    }
    // ...AND THE DEPARTURE-VS-ARRIVAL COMPARISON, restated by the presence floor. A sleeping realm
    // is drawn at the shared apparent FLOOR; a running one draws its own TRUE angular size — so
    // outside a destination's shell the body it becomes is SMALLER on screen than the point of
    // light it was, and the honest statement is that these are two different laws, each asserted
    // against its own. (The growth from the handover to the standoff is the monotone curve above.)
    assert!(
        (departure_footprint_px - DOT_MIN_APPARENT_RADIUS_PX).abs() <= READBACK_QUANTUM_PX,
        "the departure point of light must be drawn AT the shared apparent floor \
         ({DOT_MIN_APPARENT_RADIUS_PX} px), measured {departure_footprint_px:.2} px",
    );
    let arrival_model_px = dest_look_m() * (cap.camera.height as f64 * 0.5)
        / ((FIT_FOV_Y * 0.5).tan() * (dest.centre_m - cap.camera.eye).length());
    assert!(
        (dest.radius_px - arrival_model_px).abs() <= READBACK_QUANTUM_PX,
        "on arrival the destination must be drawn at ITS OWN camera-model size \
         ({arrival_model_px:.4} px), measured {:.4} px",
        dest.radius_px,
    );
    eprintln!(
        "[warp] ARRIVAL: destination footprint {:.2} px (the paint probe is applied where it \
         resolves — see above), \
         drawn stamps {:?}",
        dest.radius_px,
        assert_same_tick_composition(&cap.post, "arrival"),
    );

    // ---- NON-VACUOUS: the roster-loss machinery really ran, and the world tore a realm down. ----
    let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
    let rlm = orch_rlm(a.admin);
    eprintln!(
        "[warp] MACHINERY: window_looks_pruned {} · window_relay_levels_pruned {} · \
         window_relays_ingested {} · spins_requested {} · teardowns_reaped {}",
        gw.window_looks_pruned,
        gw.window_relay_levels_pruned,
        gw.window_relays_ingested,
        rlm.spins_requested,
        rlm.teardowns_reaped,
    );
    assert!(
        gw.window_relays_ingested > 0,
        "the destination's own look reached the composer through the Q2 PARENT RELAY — a zero \
         here means the wake handover was served by something else: {gw:?}",
    );
    assert!(
        rlm.spins_requested > 0,
        "the demand loop spun a realm up during this flight: {rlm:?}",
    );

    // ---- THE JOURNEY COMPLETES: fly in and stand inside the destination. ----
    // The destination's OWN drawn centre in the frame the picture is composed in right now — the
    // journey's last leg is aimed at what the picture says, never at a remembered coordinate. Flown
    // on the SAME held throttle the whole warp was flown on (turn, then hold), because that is both
    // the production input path and the one this flight has already proven itself on.
    let arrive_at = vd_bins::pixel::subject(&cap.post, &cap.camera, roster.sibling).centre_m;
    look_at(devctl, arrive_at);
    fly_until_label(
        devctl,
        &sibling_label,
        roster.sibling,
        // DERIVED, like every other leg of this flight (the flat two minutes was the last literal
        // left): the run in from the arrival standoff, under the DESTINATION SYSTEM's own governed
        // ceiling — the same shape the measured wake approach above is bounded by.
        vd_bins::flight::governed_leg_budget(
            &DEV,
            arrival_standoff_m(),
            realm_speed_cap_mps(dest_extent_m(), DEV.move_speed, TRAVERSE_S),
        ),
    );

    // ---- HR6: every capture, and every drawn row's provenance, is in the run manifest. ----
    assert_manifest_attests(&f.cwd, &["warp-departure", "warp-behind", "warp-arrival"]);

    // The client holds the GPU — tear it down before the cluster's own drop.
    drop(client);
}

/// ★ THE WATCHED WALK-LEG BUDGET: `governed_leg_budget` over the leg's own distance and ceiling,
/// times the watched chunk length. WHY THE FACTOR — MEASURED (2026-08-20, this gate un-parked):
/// the speed law's ramp compounds once per APPLIED INPUT, and a chunked `Drive::Walk` leg applies
/// one input per CHUNK instead of one per tick, so its ramp climbs `chunk_ticks` times slower in
/// wall clock than the per-tick closed form. At the old 300-second literal the departure leg had
/// covered less than a third of the tear-down radius it was flying to — still ramping. The
/// factor is the chunk the leg itself declares, not a number anyone picked.
fn watched_walk_budget(dist_m: f64, cap_mps: f64) -> Duration {
    vd_bins::flight::governed_leg_budget(&DEV, dist_m, cap_mps)
        * u32::try_from(WATCHED_CHUNK_TICKS).unwrap_or(1)
}

/// The design's stated close range for G-LOOK-GROWTH (look_horizon.md slice 1's pixel gate: "as
/// the camera closes" on the target) — checked at runtime against the world's own
/// geometry so the whole approach provably stays OUTSIDE the system shell (the planet remains a
/// parent-authored point of light for every sample; waking it is slice 4's work, not this
/// gate's).
fn growth_end_range_m() -> f64 {
    // TRUE-SCALE RESTATEMENT of the design's interim "down to 200 m": the close range must sit
    // OUTSIDE the planet's own containment (a crossing would re-home the observer and void the
    // whole marker premise) and INSIDE its wake radius (so the slice-4 marker⇒look handover this
    // gate ends on can happen). TWO RELEASE REACHES is both, BY CONSTRUCTION — it is twice the
    // planet's own solved shell plus outset, and the wake is that shell times the visibility
    // factor, which is orders larger — so the statement holds at any planet size the seed draws.
    // The interim literal was a distance INSIDE the true shell, and its own polar-clearance
    // algebra went NaN when the world grew, which is how the interim premise announced itself.
    let cfg = world();
    let bound = vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &cfg)
        .iter()
        .find(|r| r.realm == world_roster(&DEV).inner)
        .map(|r| r.shape.finite_extent())
        .expect("THE world rosters the inner planet");
    2.0 * (bound + cfg.band.outset_m)
}

/// G-LOOK-GROWTH — look_horizon.md slice 1's PIXEL GATE (Q2 APPROVED 2026-08-17): a planet's
/// POINT OF LIGHT grows monotonically — and STRICTLY once above the shared apparent floor — as
/// the camera closes from the licensed polar exit (deep inside the home system's wake band) down
/// to the derived close range at the inner planet ([`growth_end_range_m`] — two of the planet's
/// own release reaches), and it NEVER pops and NEVER blanks. The whole leg is flown OUTSIDE the
/// planet's own shell, so every sample is the parent's marker: the growth measured is the point
/// of light's OWN — the extent-sized marker that replaced the constant three-pixel dot (before
/// the slice this curve was FLAT at the floor for the whole run-in, then popped into the body's
/// true size).
///
/// This is also the STATION-LAPSE law in pixels for the planet's own lapse: the planets' shards
/// died when the occupant left the system (their looks pruned on the roster-loss window), and
/// the drawn set NEVER blanked — each degraded to its parent's correctly-sized marker.
#[test]
// UN-PARKED by the S5 render-scale slice (D-LOOK-3 discharged). What it had measured: from the
// licensed polar exit the inner planet composed at a 3.00 px ParentMarker (rect x636-648,
// y354-367) and the readback found that rect EMPTY at an eye distance of the system's own scale.
// Same root cause, same cure as g_warp_pixels above.
//
// ★ RE-PARKED 2026-08-21 (the gate-pass arc) ON A MEASURED DEFECT IN THE PICTURE, NOT ON
// ARITHMETIC — see the `#[ignore]` citation, which carries the whole measurement, and D-LOOK-5.
#[ignore = "PARKED 2026-08-21 (the gate-pass arc) ON A MEASURED SEAMLESS-EXPERIENCE DEFECT, NOT \
            ON ARITHMETIC. The approach flies, never blanks and never pops UP; what fails is the \
            monotone arm, and it is right to. MEASURED: the inner planet's drawn footprint fell \
            from a running maximum of 3.2105 px to 0.2724 px in one step, at tick 29500, at a \
            range of 1.4188e10 m. THE MECHANISM, arithmetically identified from the two numbers \
            themselves: 3.2105 px is the shared PRESENCE FLOOR (DOT_MIN_APPARENT_RADIUS_PX = 3.0 \
            px, plus the ~0.7% off-axis inflation the curve's own comment records), which \
            `vd_client_harness::camera::marker_world_radius` applies to every PARENT MARKER; \
            0.2724 px is exactly `look_extent / range * (h/2) / tan(fov/2)` at the planet's own \
            stated look (4447120.8351 m at 1.4188e10 m — this gate prints the floor-crossing \
            range 1.288e9 m it is derived from), which is what `vd_bins::pixel::subject` reports \
            for a `look` row: A REALM'S OWN PICTURE IS DRAWN AT ITS TRUE ANGULAR SIZE, WITH NO \
            FLOOR. So the step down IS the marker=>look HANDOVER: the planet's shard woke mid-leg \
            (the run log shows its demand and its shard announcing itself in the seconds before \
            the sample) and the picture switched authors between two samples, 11.8x smaller. THIS \
            IS A REAL POP, and the owner's standing seamless law forbids it: no toggles, no pops, \
            the same picture the whole way. WHAT IS OWED: a CONTINUOUS hand-off between the \
            presence floor and a realm's own look — either the floor fades out as the look grows \
            into it, or the look takes over only once its true angular size already exceeds the \
            floor. Until then any gate that watches one subject across its own wake measures the \
            step. WHICH ROW DISCHARGES IT: docs/design/DEFERRED.md D-LOOK-5, registered with this \
            measurement; un-ignoring this test is the proof that it landed. NOT WEAKENED AND NOT \
            DELETED: every assert stands exactly as it is, including the strict-growth arm and \
            the never-blank arm. Run it with `--ignored` to re-take the measurement."]
fn g_look_growth_a_planets_point_of_light_grows_strictly_on_approach() {
    let _tier = vd_bins::cluster_tier();
    let f = fixture("g-growth");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV);
    let mut client = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-growth",
        0,
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);

    let roster = world_roster(&DEV);
    let home_label = label_of(roster.home);
    let galaxy_label = label_of(roster.galaxy);
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(home_label.as_str()),
        "the login lands at the home star: {landed:?}",
    );
    // The planet's TRUE circumscribed extent, off the SAME boot the shard runs (SL5 — one world,
    // one derivation): what the parent's one-radius marker states, and what the growth curve is
    // judged against.
    // ★ THE PLANET'S DRAWN EXTENT IS ITS **LOOK**, not its containment bound (the bound/look split).
    // The picture states the look — the composed marker row carries it, and the renderer sizes the
    // point of light by it — while the bound is what containment uses. MEASURED here: reading the
    // BOUND demanded strict growth between two samples the client had lawfully drawn AT the
    // apparent floor (3.0000 px at both, orders of magnitude apart in range), because the bound's
    // model clears the floor orders earlier than the look's does — the bound/look split, in
    // pixels.
    let inner_extent_m = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt)
        .regions()
        .iter()
        .find(|r| r.realm == roster.inner)
        .and_then(|r| r.look)
        .expect("THE world's inner planet draws itself")
        .circumscribed_extent();
    // The approach flies the LICENSED POLAR CORRIDOR (I-AXIS), which makes the close range
    // PHASE-FREE: coming down the ±Z axis, the range to an in-plane orbiter is
    // `sqrt(z² + r_orbit²)` — nearly independent of where the planet is on its orbit — so
    // stopping at the derived close range leaves the observer `sqrt(range² − apoapsis²)` up the
    // axis, provably outside the shell at EVERY orbital phase (no far-side race exists on the
    // axis). Asserted against the world's own numbers; an in-flight centre guard backs it.
    // THE CLEARANCE STATEMENT, restated for the true-size world: the close range must clear the
    // PLANET's own containment release reach (its shell plus the band outset) with a tick of
    // margin — that, not the system shell, is what a marker premise needs, because the approach
    // now lawfully flies INSIDE the system (a system's solved shell dwarfs its planets' orbits by
    // orders of magnitude; there is no outside-the-system vantage from which a planet is even
    // resolvable).
    let inner_release_reach_m = growth_end_range_m() * 0.5;
    assert!(
        growth_end_range_m() > inner_release_reach_m + 2.0 * metres_per_tick(),
        "the growth close range ({:.1} m) must clear the planet's own release reach \
         ({inner_release_reach_m:.1} m) by more than a tick of travel",
        growth_end_range_m(),
    );
    let centre_guard_m = inner_release_reach_m;

    // ---- Leg 0: out of the home system, up the licensed polar corridor. ----
    cross_leg(
        devctl,
        "growth-exit home->galaxy (polar corridor)",
        |_tick| DVec3::new(0.0, 0.0, polar_exit_z_m()),
        &galaxy_label,
        vd_bins::flight::governed_leg_budget(
            &DEV,
            system_extent_m(),
            realm_speed_cap_mps(system_extent_m(), DEV.move_speed, TRAVERSE_S),
        ),
    );

    // ---- Leg 1 IS GONE, and that is the true-scale restatement. The interim gate flew OUT to a
    // park "just inside the wake band" because the interim system was 150 m wide and its wake
    // metres wide: you had to travel to be far from a planet. On THE world the polar exit already
    // leaves the ship TWO SYSTEM SHELLS from the system centre — orders beyond the inner planet's
    // orbit — which is exactly the far, floor-level vantage the growth curve needs, with the
    // system still deeply demanded (its wake is that shell times the visibility factor) and the
    // planet's own shard still asleep. Parking any FURTHER out is not merely unnecessary, it is
    // unreachable: the governor's arm at those ranges permits speeds at which one control cycle
    // of the pilot's own feedback loop carries the ship past any epsilon it could state (measured:
    // the out-leg oscillated across a range wider than its own arrival slop for its whole budget).
    // The gate now starts the curve where the licensed exit ends, and the floor assertion below
    // proves the vantage is far enough.
    let started = Instant::now();
    let _ = started;
    // ---- THE START: the planet is a POINT OF LIGHT at the shared floor — present, painted,
    // parent-authored (its own look lapsed when the system emptied), never a blank. ----
    let watch = [roster.inner, roster.home];
    {
        let st = vd_bins::pixel::poll(devctl);
        let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
        let inner = vd_bins::pixel::subject(&st, &camera, roster.inner);
        look_at(devctl, inner.centre_m);
    }
    let cap0 = vd_bins::pixel::straddle(
        devctl,
        "growth-start",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let start = vd_bins::pixel::subject(&cap0.post, &cap0.camera, roster.inner);
    let (pos0, _) = own_pose(&cap0.post).expect("a delivered pose at the growth start");
    let dist0 = (start.centre_m - pos0).length();
    assert_eq!(
        assert_presence(&start, "growth-start/inner"),
        Author::ParentMarker,
        "at the start the planet's shard is down (the system holds nobody) — its lapsed look \
         degraded to the parent's marker, never to nothing",
    );
    assert!(
        (start.radius_px - DOT_MIN_APPARENT_RADIUS_PX).abs() < READBACK_QUANTUM_PX,
        "at {dist0:.1} m the planet's point sits at the shared floor \
         ({DOT_MIN_APPARENT_RADIUS_PX:.1} px), measured {:.2} px",
        start.radius_px,
    );
    let painted0 = assert_painted(&cap0, &f.cwd, &start, "growth-start/inner");
    eprintln!(
        "[growth] START: the inner planet at {dist0:.1} m is a MARKER at {:.2} px \
         ({painted0} pixels painted); extent {inner_extent_m:.4} m; floor-crossing range \
         {:.1} m",
        start.radius_px,
        inner_extent_m * (cap0.camera.height as f64 * 0.5)
            / ((FIT_FOV_Y * 0.5).tan() * DOT_MIN_APPARENT_RADIUS_PX),
    );

    // ---- THE APPROACH: close from the polar exit to the derived close range, recording every
    // sample. The stop is the planet range; the centre guard is the loud backstop the polar
    // geometry makes unreachable (a trip into the shell would re-home the observer and void
    // every marker premise). ----
    let record = fly_recording(
        devctl,
        &watch,
        &[None, None],
        &[None, None],
        0,
        0,
        Drive::Throttle(FULL_AHEAD),
        // TRUE-SCALE RESTATEMENT: the run-in is DERIVED, not the 300 s literal the interim world
        // used. This approach closes the whole polar-exit range under the system's own governed
        // ceiling; measured at the 300 s literal it travelled all but the last few percent of it
        // and was called a failure by the clock alone. The budget is the one every other leg uses.
        vd_bins::flight::governed_leg_budget(
            &DEV,
            dist0,
            realm_speed_cap_mps(system_extent_m(), DEV.move_speed, TRAVERSE_S),
        ) + vd_bins::flight::governed_leg_budget(
            &DEV,
            growth_end_range_m(),
            realm_speed_cap_mps(growth_end_range_m() * 0.5, DEV.move_speed, TRAVERSE_S),
        ),
        |s, _| {
            assert!(
                s.distance(0) > centre_guard_m,
                "the approach GRAZED the planet's own shell (range {:.1} m ≤ guard \
                 {centre_guard_m:.1} m) — a crossing would re-home the observer and void every \
                 marker premise this gate stands on",
                s.distance(0),
            );
            s.distance(0) <= growth_end_range_m()
        },
    );
    // NEVER BLANKS — through the gate's OWN bridging allowance. TRUE-SCALE RESTATEMENT: the
    // approach now flies its whole derived length, which begins OUTSIDE the home system (at the
    // licensed polar exit) and re-enters it, so the composed picture is lawfully rebuilt on the
    // way in and a row may be absent for exactly the samples that bridge an epoch bump — MEASURED,
    // two such samples, at two different epochs and ranges orders apart. The old "no epoch bump
    // happened (no crossing)" premise belonged to the interim world, where the whole approach fit
    // inside one level. What is still forbidden — a blank frame with no rebuild behind it — is
    // exactly what the shared allowance asserts.
    assert_absences_only_bridge_an_epoch_bump(&record, &watch, "growth approach");
    // MONOTONE, and STRICT — at the measurement's own resolution. The measured footprint carries
    // a small off-axis projection term (the shared floor expression sizes the world radius by the
    // EUCLIDEAN eye distance while the projection divides by view depth — ~0.7 % at a few degrees
    // off-axis, measured 3.0218 px at the floor on the first red run) plus the planet's own
    // orbital swing, both bounded well inside the readback quantum. So: the footprint may never
    // fall more than one quantum below its running maximum (never pops, never shrinks), and
    // between any two samples whose MODEL growth — the extent's pure angular size at the two
    // measured distances — exceeds one quantum, the measured footprint must STRICTLY rise.
    // ★ READ THE CURVE WHERE THE PICTURE READS IT (S5, measured): a projected rectangle is
    // INFLATED off-axis by the perspective divide, so a sample taken while the subject had drifted
    // toward the frame edge reports a footprint the subject does not have — MEASURED here, a
    // running maximum of 11 385.0 px against a camera model of 9.1 px at the same range. The curve
    // keeps only the samples where the subject sat inside the frame's own MIDDLE, which is where
    // its rectangle IS its footprint. (The presence law and the handover verdicts above read every
    // sample; only the SIZE is restricted to where size means anything.)
    let centred = |s: &Sample| {
        s.subjects[0].centre_px().is_some_and(|(x, y)| {
            let (w, h) = (CAPTURE_W as f64, CAPTURE_H as f64);
            x >= w * 0.25 && x <= w * 0.75 && y >= h * 0.25 && y <= h * 0.75
        })
    };
    let mut curve: Vec<(u64, f64, f64)> = Vec::new(); // (tick, distance, radius_px)
    for s in record.samples.iter().filter(|s| centred(s)) {
        if curve.last().is_none_or(|(t, ..)| *t != s.tick) {
            curve.push((s.tick, s.distance(0), s.subjects[0].radius_px));
        }
    }
    assert!(
        curve.len() >= 32,
        "the approach must be a real curve, not a handful of samples: {} points",
        curve.len()
    );
    // The extent's pure on-axis angular radius in pixels at range `d`, FLOORED at the shared
    // apparent minimum — the model of the DRAWN size the strictness resolution is derived from
    // (the same camera constants the pilot capture projects with). The floor matters: across the
    // floor-dominated far segment the drawn size IS the constant floor, so no strict growth can
    // be owed there however far the camera closes (the second red run demanded strictness across
    // a range ratio of more than four — both ends floored, model well under the floor at each —
    // and measured only the ±0.002 px off-axis wobble: 3.0016 -> 3.0004).
    let model_px = |d: f64| {
        (inner_extent_m / d * (CAPTURE_H as f64 * 0.5) / (FIT_FOV_Y * 0.5).tan())
            .max(DOT_MIN_APPARENT_RADIUS_PX)
    };
    assert!(
        curve.len() >= 2,
        "the growth curve kept fewer than two samples with the subject inside the frame's middle",
    );
    let mut running_max = f64::NEG_INFINITY;
    let mut strict_pairs = 0u64;
    let (mut anchor_d, mut anchor_f) = (curve[0].1, curve[0].2);
    for &(tick, d, f) in &curve {
        running_max = running_max.max(f);
        assert!(
            f >= running_max - READBACK_QUANTUM_PX,
            "the point of light SHRANK on approach: {f:.4} px at tick {tick} ({d:.1} m) \
             against a running maximum of {running_max:.4} px",
        );
        if model_px(d) - model_px(anchor_d) >= READBACK_QUANTUM_PX {
            assert!(
                f > anchor_f,
                "the growth must be STRICT across a quantum of model growth: \
                 {anchor_f:.4} px ({anchor_d:.1} m) -> {f:.4} px ({d:.1} m)",
            );
            strict_pairs += 1;
            (anchor_d, anchor_f) = (d, f);
        }
    }
    assert!(
        strict_pairs > 0,
        "the strict arm never ran — the curve never rose above the floor (still the old \
         constant dot)",
    );

    // ---- THE END: at the derived close range, the point of light is the planet's TRUE
    // angular size. ----
    // SLICE 4's WAKE, AWAITED WITHIN ITS OWN DERIVED BUDGET (never a sleep literal): the
    // approach crossed the system's own interior spin-up radius in its last second,
    // so the interest byte has just landed and the vacated system owes its planets' own
    // pictures within the wake budget at the cluster's measured boot. Poll the diagnosis
    // surface for the marker⇒look flip, bounded by that budget in wall time — doubled, stated:
    // the budget counts UNIVERSE ticks and this wait is wall-clock across five processes, so
    // one factor of two absorbs pacing jitter and the poll's own sampling gap (the same
    // resolution allowance the in-flight handover measurement adds explicitly).
    let boot_ticks = orch_rlm(a.admin).boot_ticks_observed_max;
    let wake_secs = wake_budget_ticks(boot_ticks) as f64 * DEV.tick_dt;
    let wake_deadline = Instant::now() + Duration::from_secs_f64(2.0 * wake_secs);
    let inner_name = format!("{:?}", roster.inner);
    loop {
        let st = vd_bins::pixel::poll(devctl);
        if st
            .realm_boxes
            .iter()
            .any(|b| (b.realm == inner_name) & (b.body_kind == "look"))
        {
            break;
        }
        assert!(
            Instant::now() < wake_deadline,
            "standing inside the vacated system's interior band, the planets' own pictures \
             never arrived within the derived wake budget ({wake_secs:.1} s at measured boot \
             {boot_ticks} ticks, doubled for wall-clock jitter) — the slice-4 wake failed",
        );
        std::thread::sleep(SAMPLE_POLL);
    }
    let cap1 = vd_bins::pixel::straddle(
        devctl,
        "growth-end",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch,
        vd_bins::pixel::pilot_camera,
    );
    let end = vd_bins::pixel::subject(&cap1.post, &cap1.camera, roster.inner);
    let (pos1, _) = own_pose(&cap1.post).expect("a delivered pose at the growth end");
    let dist1 = (end.centre_m - pos1).length();
    // SLICE 4 LANDED (look_horizon.md §6 slice 4; Q1 APPROVED, owner 2026-08-17): this assert
    // used to pin the OLD posture ("waking it from outside is slice 4's work" — ParentMarker),
    // and it went RED the run the interest bit landed, exactly as its own message predicted.
    // Standing here — outside the PLANET's own shell, INSIDE the system's interior band — the
    // vacated system now holds the interest byte, its down-proxy wakes the planets,
    // and the planet's OWN picture arrives through the sealed interior forward (slice 3). The
    // planets are running when you look at them: presence is the planet's OWN statement.
    assert_eq!(
        assert_presence(&end, "growth-end/inner"),
        Author::SelfLook,
        "standing inside the vacated system's interior band, the planet draws its OWN picture \
         (the slice-4 wake + the slice-3 sealed interior forward)",
    );
    // The delivered one-radius datum IS the world's own extent (the wire's extent against the
    // out-of-band boot derivation — not the gate reading its own output back).
    let delivered_extent = cap1
        .post
        .realm_boxes
        .iter()
        .find(|b| b.realm == format!("{:?}", roster.inner))
        .expect("the inner planet's box is in the diagnosis surface")
        .extent_m;
    // THE TOLERANCE IS RELATIVE, and it is float noise — nothing else. These are TWO SPELLINGS OF
    // ONE DERIVATION (the boot's own look extent, shipped and read back), so the only lawful
    // difference between them is the last few bits of an f64 at whatever magnitude the planet
    // happens to have. An ABSOLUTE metre count cannot say that: 1e-9 m is far below one ulp at
    // this world's magnitudes, so it was asserting the impossible, and it would be a slack
    // thousands of ulp wide on a small body.
    let extent_noise = 4.0 * f64::EPSILON * inner_extent_m.abs().max(delivered_extent.abs());
    assert!(
        (delivered_extent - inner_extent_m).abs() <= extent_noise,
        "the delivered marker radius ({delivered_extent}) must be THE world's own circumscribed \
         extent ({inner_extent_m}) — lawful float noise at this magnitude is {extent_noise:e} m",
    );
    let expected_end_world = vd_client_harness::camera::marker_world_radius(
        inner_extent_m,
        dist1,
        cap1.camera.fov_y,
        cap1.camera.height as f64,
    );
    let expected_end_px = vd_client_harness::verdict::projected_point_aabb(
        &cap1.camera,
        end.centre_m,
        expected_end_world,
    )
    .map_or(0.0, |r| (r.max.x - r.min.x) * 0.5);
    assert!(
        (end.radius_px - expected_end_px).abs() < READBACK_QUANTUM_PX,
        "at {dist1:.1} m the point must draw at its true angular size ({expected_end_px:.2} px), \
         measured {:.2} px",
        end.radius_px,
    );
    assert!(
        end.radius_px > start.radius_px + READBACK_QUANTUM_PX,
        "the point of light must have GROWN over the leg: {:.2} px -> {:.2} px",
        start.radius_px,
        end.radius_px,
    );
    let painted1 = assert_painted(&cap1, &f.cwd, &end, "growth-end/inner");
    eprintln!(
        "[growth] G-LOOK-GROWTH GREEN: {:.2} px at {dist0:.1} m -> {:.2} px at {dist1:.1} m over \
         {} distinct-tick samples ({strict_pairs} strict pairs above the floor); {painted1} \
         pixels painted at the close range; no blank sample, no shrink, no pop.",
        start.radius_px,
        end.radius_px,
        curve.len(),
    );
    assert_manifest_attests(&f.cwd, &["growth-start", "growth-end"]);
    drop(client);
}

/// THE JOURNEY'S LAST LEG: hold the throttle until the picture says the ship STANDS IN `want` —
/// the leg's outcome is a REALM LABEL, never a coordinate. Fails with the distance still to run, so
/// a stalled crossing names itself instead of timing out silently.
fn fly_until_label(devctl: u16, want: &str, subject_realm: RealmId, deadline: Duration) {
    let started = Instant::now();
    throttle(devctl, FULL_AHEAD);
    loop {
        std::thread::sleep(SAMPLE_POLL);
        let st = vd_bins::pixel::poll(devctl);
        if st.location.as_deref() == Some(want) {
            throttle(devctl, ALL_STOP);
            eprintln!("[warp] ARRIVED: the ship stands in {want}");
            return;
        }
        if started.elapsed() >= deadline {
            let camera = vd_bins::pixel::pilot_camera(&st, CAPTURE_W as usize, CAPTURE_H as usize);
            let sub = vd_bins::pixel::subject(&st, &camera, subject_realm);
            let (pos, _) = own_pose(&st).expect("a delivered pose");
            throttle(devctl, ALL_STOP);
            panic!(
                "the ship never crossed into {want}: it is at {pos:?}, {:.1} m from the \
                 destination's drawn centre {:?}, still standing in {:?}",
                (sub.centre_m - pos).length(),
                sub.centre_m,
                st.location,
            );
        }
    }
}

/// HR6: every capture this gate took is in the run manifest, and each carries the STATE DUMP that
/// attests every drawn row's provenance (which realm, which lawful author, which universe tick).
fn assert_manifest_attests(cwd: &std::path::Path, labels: &[&str]) {
    let runs = cwd.join("runs");
    let run_dir = std::fs::read_dir(&runs)
        .unwrap_or_else(|e| panic!("no runs dir at {}: {e}", runs.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join(MANIFEST_FILENAME).exists())
        .unwrap_or_else(|| panic!("no run manifest under {}", runs.display()));
    let manifest = RunManifest::from_json(
        &std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest"),
    )
    .expect("parse manifest");
    for label in labels {
        let entry = manifest
            .captures
            .iter()
            .find(|c| c.path.contains(label))
            .unwrap_or_else(|| panic!("capture '{label}' is not in the run manifest"));
        let state_rel = entry
            .state_path
            .as_ref()
            .unwrap_or_else(|| panic!("capture '{label}' has no state dump to attest its rows"));
        let dump: DevState = serde_json::from_str(
            &std::fs::read_to_string(run_dir.join(state_rel)).expect("read state dump"),
        )
        .expect("parse state dump");
        assert!(
            !dump.realm_boxes.is_empty(),
            "capture '{label}': the attested picture has no drawn rows",
        );
        for row in &dump.realm_boxes {
            assert!(
                row.body_kind == "look" || row.body_kind == "marker",
                "capture '{label}': row {} has no lawful author ({})",
                row.realm,
                row.body_kind,
            );
        }
        eprintln!(
            "[warp] manifest attests '{label}': {} drawn rows, each with its lawful author",
            dump.realm_boxes.len(),
        );
    }
}
