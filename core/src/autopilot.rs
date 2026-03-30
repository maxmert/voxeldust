//! Brachistochrone trajectory planner with gravity-aware continuous guidance.
//!
//! Shared between client (HUD preview) and system shard (authoritative execution).
//! Computes intercept trajectories to orbiting targets with accel→flip→brake phases.

use glam::DVec3;

use crate::system::{
    compute_gravity_acceleration, compute_planet_position, compute_soi_radius, PlanetParams,
    StarParams, SystemParams,
};

pub const SHIP_MASS: f64 = 10_000.0;

/// Engine tier definition. Two categories: maneuver (no dampening, low g)
/// and high-thrust (inertial dampener active, extreme acceleration).
#[derive(Debug, Clone, Copy)]
pub struct EngineTier {
    /// Display name for HUD.
    pub name: &'static str,
    /// True acceleration in m/s² (thrust_force / SHIP_MASS).
    pub acceleration: f64,
    /// G-force felt by crew after inertial dampening.
    pub felt_g: f64,
    /// Whether inertial dampener is active at this tier.
    pub dampened: bool,
}

impl EngineTier {
    /// Thrust force in Newtons for this tier.
    pub const fn thrust_force(&self) -> f64 {
        self.acceleration * SHIP_MASS
    }
}

/// Engine tiers from maneuver to emergency high-thrust.
///
/// | Tier | Name        | Accel      | Felt | ~1e11 m | ~4e11 m |
/// |------|-------------|------------|------|---------|---------|
/// | 0    | Maneuver    | 4.9 m/s²   | 0.5g | —       | —       |
/// | 1    | Impulse     | 29.4 m/s²  | 3.0g | —       | —       |
/// | 2    | Cruise      | 49 km/s²   | 1.0g | 47 min  | 94 min  |
/// | 3    | Long Range  | 490 km/s²  | 2.0g | 15 min  | 30 min  |
/// | 4    | Emergency   | 2450 km/s² | 5.0g | 6.7 min | 13 min  |
pub const ENGINE_TIERS: [EngineTier; 5] = [
    EngineTier { name: "Maneuver",   acceleration: 4.905,       felt_g: 0.5, dampened: false },
    EngineTier { name: "Impulse",    acceleration: 29.43,       felt_g: 3.0, dampened: false },
    EngineTier { name: "Cruise",     acceleration: 49_050.0,    felt_g: 1.0, dampened: true },
    EngineTier { name: "Long Range", acceleration: 490_500.0,   felt_g: 2.0, dampened: true },
    EngineTier { name: "Emergency",  acceleration: 2_452_500.0, felt_g: 5.0, dampened: true },
];

/// Flight phase of the autopilot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightPhase {
    Accelerate,
    Flip,
    Brake,
    Arrived,
}

/// A sampled point along the planned trajectory (for HUD rendering).
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPoint {
    pub position: DVec3,
    pub velocity: DVec3,
    pub phase: FlightPhase,
    /// Celestial time at this point.
    pub time: f64,
}

/// Result of trajectory planning (for HUD visualization).
#[derive(Debug, Clone)]
pub struct TrajectoryPlan {
    /// Sampled points along the trajectory.
    pub points: Vec<TrajectoryPoint>,
    /// Current flight phase at the ship's position.
    pub current_phase: FlightPhase,
    /// Estimated time of arrival in real seconds.
    pub eta_real_seconds: f64,
    /// Where the target planet will be when we arrive.
    pub intercept_position: DVec3,
    /// Current thrust direction (world-space unit vector).
    pub thrust_direction: DVec3,
    /// Current thrust magnitude (Newtons).
    pub thrust_magnitude: f64,
    /// Index in `points` where flip occurs.
    pub flip_index: usize,
    /// Target planet index (0-based into system.planets).
    pub target_planet_index: usize,
    /// SOI radius of the target planet.
    pub target_soi_radius: f64,
    /// G-force felt by crew (dampened).
    pub felt_g: f64,
    /// Whether inertial dampener is active.
    pub dampener_active: bool,
    /// Engine tier name for HUD display.
    pub engine_tier_name: &'static str,
}

/// Output of per-tick guidance computation.
#[derive(Debug, Clone)]
pub struct GuidanceCommand {
    /// Thrust direction in world space (unit vector). Ship should orient to face this.
    pub thrust_direction: DVec3,
    /// Thrust force magnitude (Newtons), before ramp factor.
    pub thrust_magnitude: f64,
    /// Current flight phase.
    pub phase: FlightPhase,
    /// Whether autopilot has completed (arrived at SOI).
    pub completed: bool,
    /// ETA in real seconds.
    pub eta_real_seconds: f64,
    /// G-force felt by crew (after dampening, before ramp).
    pub felt_g: f64,
    /// Whether inertial dampener is active.
    pub dampener_active: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get engine tier, clamped to valid range.
pub fn engine_tier(tier: u8) -> &'static EngineTier {
    &ENGINE_TIERS[(tier as usize).min(ENGINE_TIERS.len() - 1)]
}

/// Progressive thrust ramp factor (0.0–1.0).
///
/// Ramps thrust linearly from 0 to full over `ramp_duration` at the start
/// of acceleration, and from full to 0 over the last `ramp_duration` before
/// arrival. The flip transition is handled naturally by the alignment-based
/// thrust scaling (cos(misalignment) → 0 during rotation).
///
/// `elapsed` = real seconds since autopilot engage.
/// `tof` = estimated total time of flight.
pub fn thrust_ramp_factor(elapsed: f64, tof: f64, phase: FlightPhase) -> f64 {
    if tof <= 0.0 {
        return 1.0;
    }
    let ramp_duration = (tof / 8.0).clamp(5.0, 30.0);
    match phase {
        // Smooth ramp-up at start of acceleration only.
        // Braking always uses full thrust to match stopping distance formula v²/(2a).
        FlightPhase::Accelerate => (elapsed / ramp_duration).clamp(0.0, 1.0),
        _ => 1.0,
    }
}

/// Brachistochrone time-of-flight: accelerate to midpoint, flip, decelerate to rest.
/// Handles any initial velocity (positive = toward target, negative = away).
///
/// Formula: `tof = (2*sqrt(a*d + v²/2) - v) / a`
fn brachistochrone_tof(distance: f64, v_along: f64, accel: f64) -> f64 {
    if accel <= 0.0 || distance <= 0.0 {
        return 0.0;
    }
    let inner = accel * distance + 0.5 * v_along * v_along;
    if inner < 0.0 {
        return 0.0;
    }
    let tof = (2.0 * inner.sqrt() - v_along) / accel;
    tof.max(0.0)
}

// ---------------------------------------------------------------------------
// Intercept solver
// ---------------------------------------------------------------------------

/// Solve for the intercept point: where will the target planet be when we arrive?
///
/// Returns `(intercept_position, tof_real_seconds)` or None if unsolvable.
/// `intercept_position` is on the SOI boundary (not planet center).
pub fn solve_intercept(
    ship_pos: DVec3,
    ship_vel: DVec3,
    target_planet: &PlanetParams,
    star: &StarParams,
    celestial_time: f64,
    time_scale: f64,
    thrust_tier: u8,
) -> Option<(DVec3, f64)> {
    let accel = engine_tier(thrust_tier).acceleration;
    let soi = compute_soi_radius(target_planet, star);

    // Initial estimate using current target position.
    let target_now = compute_planet_position(target_planet, celestial_time);
    let d0 = (target_now - ship_pos).length();
    if d0 < soi {
        // Already inside SOI — autopilot considers this arrived.
        return Some((target_now, 0.0));
    }

    // Seed TOF: brachistochrone (accel half, decel half) with initial velocity.
    // Unified formula valid for any v_along sign:
    //   tof = (2*sqrt(a*d + v²/2) - v) / a
    let dir0 = (target_now - ship_pos).normalize();
    let v_along = ship_vel.dot(dir0);
    let mut tof_real = brachistochrone_tof(d0, v_along, accel);

    // Iteratively refine: planet moves while we travel.
    for _ in 0..10 {
        let arrival_celestial = celestial_time + tof_real * time_scale;
        let planet_future = compute_planet_position(target_planet, arrival_celestial);

        // Aim for SOI boundary, on the side facing the ship.
        let approach_dir = (planet_future - ship_pos).normalize();
        let intercept_pos = planet_future - approach_dir * soi;
        let d = (intercept_pos - ship_pos).length();

        let dir = (intercept_pos - ship_pos).normalize();
        let v_along = ship_vel.dot(dir);
        let tof_new = brachistochrone_tof(d, v_along, accel);
        if tof_new <= 0.0 {
            return None;
        }

        if (tof_new - tof_real).abs() < tof_real.max(1.0) * 0.01 {
            tof_real = tof_new;
            break;
        }
        tof_real = tof_new;
    }

    let arrival_celestial = celestial_time + tof_real * time_scale;
    let planet_future = compute_planet_position(target_planet, arrival_celestial);
    let approach_dir = (planet_future - ship_pos).normalize();
    let intercept_pos = planet_future - approach_dir * soi;

    Some((intercept_pos, tof_real))
}

// ---------------------------------------------------------------------------
// Per-tick guidance (gravity-aware)
// ---------------------------------------------------------------------------

/// Compute the thrust command for the current tick.
///
/// Leans into helpful gravity (along the desired direction) and only
/// counteracts the perpendicular component. Produces naturally curved paths.
pub fn compute_guidance(
    ship_pos: DVec3,
    ship_vel: DVec3,
    intercept_pos: DVec3,
    target_planet_index: usize,
    system: &SystemParams,
    planet_positions: &[DVec3],
    celestial_time: f64,
    thrust_tier: u8,
) -> GuidanceCommand {
    let et = engine_tier(thrust_tier);
    let thrust_force = et.acceleration * SHIP_MASS;
    let accel = et.acceleration;

    let planet = &system.planets[target_planet_index];
    let star = &system.star;
    let soi = compute_soi_radius(planet, star);
    let planet_pos = planet_positions[target_planet_index];

    // Check if arrived at SOI.
    let dist_to_planet = (ship_pos - planet_pos).length();
    if dist_to_planet <= soi {
        return GuidanceCommand {
            thrust_direction: DVec3::NEG_Z,
            thrust_magnitude: 0.0,
            phase: FlightPhase::Arrived,
            completed: true,
            eta_real_seconds: 0.0,
            felt_g: 0.0,
            dampener_active: et.dampened,
        };
    }

    // Direction and distance to intercept point.
    let to_intercept = intercept_pos - ship_pos;
    let dist_to_intercept = to_intercept.length();
    if dist_to_intercept < 1.0 {
        return GuidanceCommand {
            thrust_direction: DVec3::NEG_Z,
            thrust_magnitude: 0.0,
            phase: FlightPhase::Arrived,
            completed: true,
            eta_real_seconds: 0.0,
            felt_g: 0.0,
            dampener_active: et.dampened,
        };
    }
    let desired_dir = to_intercept / dist_to_intercept;

    // Gravity at current position.
    let gravity = compute_gravity_acceleration(
        ship_pos,
        star,
        &system.planets,
        planet_positions,
        celestial_time,
    );

    // Decompose gravity into helpful (along desired dir) and perpendicular.
    let grav_along = gravity.dot(desired_dir);
    let grav_perp = gravity - desired_dir * grav_along;

    // Velocity components.
    let speed = ship_vel.length();
    let v_along = ship_vel.dot(desired_dir);
    let v_perp = ship_vel - desired_dir * v_along;

    // Stopping distance at current prograde speed.
    // Conservative: don't count on gravity helping during braking.
    let stopping_dist = if v_along > 0.0 {
        v_along * v_along / (2.0 * accel)
    } else {
        0.0
    };

    // ETA estimate (brachistochrone).
    let eta_real = brachistochrone_tof(dist_to_intercept, v_along, accel);

    // Phase determination: brake when stopping distance reaches remaining distance.
    if stopping_dist >= dist_to_intercept * 0.95 && v_along > 0.0 {
        // BRAKE phase: thrust retrograde.
        let mut thrust_dir = if speed > 0.1 {
            -ship_vel.normalize()
        } else {
            -desired_dir
        };

        // Correct lateral drift during braking.
        if v_perp.length() > 1.0 {
            let correction = -v_perp.normalize() * 0.3;
            thrust_dir = (thrust_dir + correction).normalize();
        }

        // If gravity is helping decelerate, reduce thrust proportionally.
        let grav_decel = -gravity.dot(ship_vel.normalize_or_zero());
        let needed_thrust = (thrust_force - grav_decel * SHIP_MASS).max(0.0);

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: needed_thrust.min(thrust_force),
            phase: FlightPhase::Brake,
            completed: false,
            eta_real_seconds: eta_real,
            felt_g: et.felt_g,
            dampener_active: et.dampened,
        }
    } else {
        // ACCELERATE phase: thrust toward intercept, lean into helpful gravity.
        // Only counteract perpendicular gravity component — let along-track gravity help.
        let desired_accel = desired_dir * accel;
        let gravity_correction = -grav_perp;
        let mut thrust_dir = (desired_accel + gravity_correction).normalize();

        // Correct accumulated perpendicular velocity.
        if v_perp.length() > 10.0 {
            let correction = -v_perp.normalize() * 0.2;
            thrust_dir = (thrust_dir + correction).normalize();
        }

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: thrust_force,
            phase: FlightPhase::Accelerate,
            completed: false,
            eta_real_seconds: eta_real,
            felt_g: et.felt_g,
            dampener_active: et.dampened,
        }
    }
}

// ---------------------------------------------------------------------------
// Trajectory planning (for HUD visualization)
// ---------------------------------------------------------------------------

/// Plan a full trajectory from current state to target planet SOI.
///
/// Forward-simulates with gravity-aware guidance to produce sampled points
/// for HUD rendering via Catmull-Rom spline.
pub fn plan_trajectory(
    ship_pos: DVec3,
    ship_vel: DVec3,
    target_planet_index: usize,
    system: &SystemParams,
    celestial_time: f64,
    thrust_tier: u8,
    sample_count: usize,
) -> Option<TrajectoryPlan> {
    if target_planet_index >= system.planets.len() {
        return None;
    }

    let planet = &system.planets[target_planet_index];
    let star = &system.star;
    let soi = compute_soi_radius(planet, star);

    // Solve intercept.
    let (intercept_pos, eta_real) = solve_intercept(
        ship_pos,
        ship_vel,
        planet,
        star,
        celestial_time,
        system.scale.time_scale,
        thrust_tier,
    )?;

    if eta_real <= 0.0 {
        // Already inside SOI — return an immediate Arrived plan.
        let et = engine_tier(thrust_tier);
        return Some(TrajectoryPlan {
            points: vec![],
            current_phase: FlightPhase::Arrived,
            eta_real_seconds: 0.0,
            intercept_position: intercept_pos,
            thrust_direction: DVec3::NEG_Z,
            thrust_magnitude: 0.0,
            flip_index: 0,
            target_planet_index,
            target_soi_radius: soi,
            felt_g: 0.0,
            dampener_active: et.dampened,
            engine_tier_name: et.name,
        });
    }

    // Forward simulate.
    // Simulate slightly beyond ETA to ensure we capture arrival.
    let sim_dt_real = (eta_real * 1.2 / sample_count as f64).max(0.1);
    let sim_dt_celestial = sim_dt_real * system.scale.time_scale;

    let mut pos = ship_pos;
    let mut vel = ship_vel;
    let mut ct = celestial_time;
    let mut points = Vec::with_capacity(sample_count);
    let mut flip_index = 0;
    let mut found_flip = false;

    for i in 0..sample_count {
        // Planet positions at this simulation time.
        let planet_positions: Vec<DVec3> = system
            .planets
            .iter()
            .map(|p| compute_planet_position(p, ct))
            .collect();

        let planet_pos = planet_positions[target_planet_index];
        let dist_to_planet = (pos - planet_pos).length();

        // Check SOI arrival.
        if dist_to_planet <= soi {
            points.push(TrajectoryPoint {
                position: pos,
                velocity: vel,
                phase: FlightPhase::Arrived,
                time: ct,
            });
            break;
        }

        // Re-solve intercept periodically during simulation for accuracy.
        let current_intercept = if i % 20 == 0 {
            solve_intercept(
                pos,
                vel,
                planet,
                star,
                ct,
                system.scale.time_scale,
                thrust_tier,
            )
            .map(|(p, _)| p)
            .unwrap_or(intercept_pos)
        } else {
            intercept_pos
        };

        // Guidance at this point.
        let guidance = compute_guidance(
            pos,
            vel,
            current_intercept,
            target_planet_index,
            system,
            &planet_positions,
            ct,
            thrust_tier,
        );

        if !found_flip && guidance.phase == FlightPhase::Brake {
            flip_index = i;
            found_flip = true;
        }

        points.push(TrajectoryPoint {
            position: pos,
            velocity: vel,
            phase: guidance.phase,
            time: ct,
        });

        if guidance.completed {
            break;
        }

        // Integrate (true Velocity Verlet / Störmer-Verlet).
        // Two-evaluation for symplectic energy conservation.
        let gravity_old = compute_gravity_acceleration(
            pos, star, &system.planets, &planet_positions, ct,
        );
        let thrust_accel = guidance.thrust_direction * (guidance.thrust_magnitude / SHIP_MASS);
        let accel_old = gravity_old + thrust_accel;

        // Advance position.
        pos += vel * sim_dt_real + 0.5 * accel_old * sim_dt_real * sim_dt_real;
        ct += sim_dt_celestial;

        // Recompute gravity at new position and new time (planets have moved).
        let planet_positions_new: Vec<DVec3> = system
            .planets
            .iter()
            .map(|p| compute_planet_position(p, ct))
            .collect();
        let gravity_new = compute_gravity_acceleration(
            pos, star, &system.planets, &planet_positions_new, ct,
        );
        let accel_new = gravity_new + thrust_accel;

        // Advance velocity with average acceleration.
        vel += 0.5 * (accel_old + accel_new) * sim_dt_real;
    }

    if points.is_empty() {
        return None;
    }

    // Current guidance for the ship's actual position.
    let planet_positions_now: Vec<DVec3> = system
        .planets
        .iter()
        .map(|p| compute_planet_position(p, celestial_time))
        .collect();
    let initial_guidance = compute_guidance(
        ship_pos,
        ship_vel,
        intercept_pos,
        target_planet_index,
        system,
        &planet_positions_now,
        celestial_time,
        thrust_tier,
    );

    let et = engine_tier(thrust_tier);
    Some(TrajectoryPlan {
        points,
        current_phase: initial_guidance.phase,
        eta_real_seconds: eta_real,
        intercept_position: intercept_pos,
        thrust_direction: initial_guidance.thrust_direction,
        thrust_magnitude: initial_guidance.thrust_magnitude,
        flip_index,
        target_planet_index,
        target_soi_radius: soi,
        felt_g: et.felt_g,
        dampener_active: et.dampened,
        engine_tier_name: et.name,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::SystemParams;

    fn test_system() -> SystemParams {
        SystemParams::from_seed(42)
    }

    #[test]
    fn engine_tiers_increasing() {
        for i in 1..ENGINE_TIERS.len() {
            assert!(
                ENGINE_TIERS[i].acceleration > ENGINE_TIERS[i - 1].acceleration,
                "tier {} accel not greater than tier {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn engine_tier_accelerations() {
        assert!((ENGINE_TIERS[0].acceleration - 4.905).abs() < 0.01, "tier 0 should be ~4.9 m/s²");
        assert!((ENGINE_TIERS[1].acceleration - 29.43).abs() < 0.01, "tier 1 should be ~29.4 m/s²");
        assert!((ENGINE_TIERS[2].acceleration - 49_050.0).abs() < 1.0, "tier 2 should be ~49050 m/s²");
        assert!((ENGINE_TIERS[3].acceleration - 490_500.0).abs() < 1.0, "tier 3 should be ~490500 m/s²");
        assert!((ENGINE_TIERS[4].acceleration - 2_452_500.0).abs() < 1.0, "tier 4 should be ~2452500 m/s²");
    }

    #[test]
    fn engine_tier_dampening() {
        assert!(!ENGINE_TIERS[0].dampened, "maneuver should not be dampened");
        assert!(!ENGINE_TIERS[1].dampened, "impulse should not be dampened");
        assert!(ENGINE_TIERS[2].dampened, "cruise should be dampened");
        assert!(ENGINE_TIERS[3].dampened, "long range should be dampened");
        assert!(ENGINE_TIERS[4].dampened, "emergency should be dampened");
    }

    #[test]
    fn thrust_ramp_factor_values() {
        let tof = 100.0;
        let ramp = tof / 8.0; // 12.5s
        // Start of accel: ramp up
        assert!((thrust_ramp_factor(0.0, tof, FlightPhase::Accelerate)).abs() < 0.01);
        assert!((thrust_ramp_factor(ramp / 2.0, tof, FlightPhase::Accelerate) - 0.5).abs() < 0.01);
        assert!((thrust_ramp_factor(ramp, tof, FlightPhase::Accelerate) - 1.0).abs() < 0.01);
        // Mid-accel: full
        assert!((thrust_ramp_factor(50.0, tof, FlightPhase::Accelerate) - 1.0).abs() < 0.01);
        // Brake: always full thrust (no ramp-down — needed for stopping distance accuracy)
        assert!((thrust_ramp_factor(60.0, tof, FlightPhase::Brake) - 1.0).abs() < 0.01);
        assert!((thrust_ramp_factor(tof, tof, FlightPhase::Brake) - 1.0).abs() < 0.01);
        assert!((thrust_ramp_factor(tof - 1.0, tof, FlightPhase::Brake) - 1.0).abs() < 0.01);
    }

    #[test]
    fn solve_intercept_converges() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;

        // Ship at spawn offset from planet 0.
        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);
        let ship_vel = DVec3::ZERO;

        let result = solve_intercept(
            ship_pos,
            ship_vel,
            planet,
            star,
            0.0,
            sys.scale.time_scale,
            1, // 1g
        );

        assert!(result.is_some(), "intercept should converge");
        let (intercept_pos, tof) = result.unwrap();
        assert!(tof > 0.0, "TOF should be positive");
        assert!(intercept_pos.length() > 0.0, "intercept should be non-zero");

        // Intercept should be near where the planet will be.
        let arrival_celestial = 0.0 + tof * sys.scale.time_scale;
        let planet_future = compute_planet_position(planet, arrival_celestial);
        let soi = compute_soi_radius(planet, star);
        let dist_to_planet = (intercept_pos - planet_future).length();
        assert!(
            (dist_to_planet - soi).abs() < soi * 0.1,
            "intercept should be near SOI boundary: dist={:.2e}, soi={:.2e}",
            dist_to_planet,
            soi
        );
    }

    #[test]
    fn solve_intercept_already_at_target() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;

        // Ship inside planet SOI.
        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(100.0, 0.0, 0.0);

        let result = solve_intercept(
            ship_pos,
            DVec3::ZERO,
            planet,
            star,
            0.0,
            sys.scale.time_scale,
            1,
        );

        assert!(result.is_some());
        let (_, tof) = result.unwrap();
        assert!(tof < 1.0, "TOF should be near zero when already at target");
    }

    #[test]
    fn guidance_accelerate_phase() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;

        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);
        let ship_vel = DVec3::ZERO;

        let (intercept_pos, _) = solve_intercept(
            ship_pos,
            ship_vel,
            planet,
            star,
            0.0,
            sys.scale.time_scale,
            1,
        )
        .unwrap();

        let planet_positions: Vec<DVec3> = sys
            .planets
            .iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();

        let cmd = compute_guidance(
            ship_pos,
            ship_vel,
            intercept_pos,
            0,
            &sys,
            &planet_positions,
            0.0,
            1,
        );

        assert_eq!(cmd.phase, FlightPhase::Accelerate);
        assert!(!cmd.completed);
        assert!(cmd.thrust_magnitude > 0.0);
        assert!(cmd.thrust_direction.length() > 0.99);
        assert!(cmd.eta_real_seconds > 0.0);
    }

    #[test]
    fn guidance_brake_phase_at_high_speed() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;

        let planet_pos = compute_planet_position(planet, 0.0);
        let soi = compute_soi_radius(planet, star);
        // Ship close to target at high speed. Tier 1 (Impulse) = 29.43 m/s².
        // Stopping from v at accel a: d_stop = v²/(2*a).
        // At v=1000 m/s, a=29.43: d_stop = 1e6/58.86 = 16,989 m.
        // Place ship at soi + 20,000 m with 20,000 m remaining to intercept.
        let remaining = 20_000.0;
        let ship_pos = planet_pos + DVec3::new(soi + remaining, 0.0, 0.0);
        let ship_vel = DVec3::new(-1_000.0, 0.0, 0.0); // 1 km/s toward planet

        let intercept_pos = planet_pos + DVec3::new(soi, 0.0, 0.0);

        let planet_positions: Vec<DVec3> = sys
            .planets
            .iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();

        let cmd = compute_guidance(
            ship_pos,
            ship_vel,
            intercept_pos,
            0,
            &sys,
            &planet_positions,
            0.0,
            1, // Impulse tier
        );

        // stopping_dist = 1000²/(2*29.43) ≈ 16,989 m ≥ 20,000 * 0.95 = 19,000? No.
        // Need higher speed: v=1100 → d_stop = 1.21e6/58.86 = 20,557 ≥ 19,000 → braking.
        // Let's use v=1100.
        let ship_vel_fast = DVec3::new(-1_100.0, 0.0, 0.0);
        let cmd = compute_guidance(
            ship_pos,
            ship_vel_fast,
            intercept_pos,
            0,
            &sys,
            &planet_positions,
            0.0,
            1,
        );

        assert_eq!(cmd.phase, FlightPhase::Brake);
    }

    #[test]
    fn guidance_arrived_inside_soi() {
        let sys = test_system();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let soi = compute_soi_radius(&sys.planets[0], &sys.star);

        // Ship inside SOI.
        let ship_pos = planet_pos + DVec3::new(soi * 0.5, 0.0, 0.0);

        let planet_positions: Vec<DVec3> = sys
            .planets
            .iter()
            .map(|p| compute_planet_position(p, 0.0))
            .collect();

        let cmd = compute_guidance(
            ship_pos,
            DVec3::ZERO,
            planet_pos,
            0,
            &sys,
            &planet_positions,
            0.0,
            1,
        );

        assert_eq!(cmd.phase, FlightPhase::Arrived);
        assert!(cmd.completed);
    }

    #[test]
    fn plan_trajectory_produces_points() {
        let sys = test_system();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);

        let plan = plan_trajectory(ship_pos, DVec3::ZERO, 0, &sys, 0.0, 1, 200);

        assert!(plan.is_some(), "trajectory plan should succeed");
        let plan = plan.unwrap();
        assert!(!plan.points.is_empty(), "should have trajectory points");
        assert!(plan.eta_real_seconds > 0.0, "ETA should be positive");
        assert!(plan.target_soi_radius > 0.0);

        // Should have both accel and brake phases.
        let has_accel = plan
            .points
            .iter()
            .any(|p| p.phase == FlightPhase::Accelerate);
        let has_brake = plan.points.iter().any(|p| p.phase == FlightPhase::Brake);
        assert!(has_accel, "should have acceleration phase");
        // Brake phase may or may not appear depending on simulation resolution.
        // At large distances with 200 samples, it should appear.
        if plan.points.len() > 10 {
            assert!(has_brake, "should have braking phase for long trajectories");
        }
    }

    #[test]
    fn plan_trajectory_invalid_target() {
        let sys = test_system();
        let plan = plan_trajectory(DVec3::ZERO, DVec3::ZERO, 99, &sys, 0.0, 1, 200);
        assert!(plan.is_none(), "invalid target should return None");
    }

    #[test]
    fn higher_thrust_tier_faster_eta() {
        let sys = test_system();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);

        let eta_impulse = solve_intercept(
            ship_pos,
            DVec3::ZERO,
            &sys.planets[0],
            &sys.star,
            0.0,
            sys.scale.time_scale,
            1, // Impulse (29.4 m/s²)
        )
        .unwrap()
        .1;

        let eta_cruise = solve_intercept(
            ship_pos,
            DVec3::ZERO,
            &sys.planets[0],
            &sys.star,
            0.0,
            sys.scale.time_scale,
            2, // Cruise (49050 m/s²)
        )
        .unwrap()
        .1;

        assert!(
            eta_cruise < eta_impulse,
            "Cruise ({:.0}s) should be faster than Impulse ({:.0}s)",
            eta_cruise,
            eta_impulse
        );
    }
}
