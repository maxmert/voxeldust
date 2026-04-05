//! Brachistochrone trajectory planner with gravity-aware continuous guidance.
//!
//! Shared between client (HUD preview) and system shard (authoritative execution).
//! Computes intercept trajectories to orbiting targets with accel→flip→brake phases.

use glam::DVec3;

use crate::system::{
    compute_gravity_acceleration, compute_planet_position, compute_planet_velocity,
    compute_soi_radius, PlanetParams, StarParams, SystemParams,
};

/// Physics tick interval in seconds (20 Hz).
pub const PHYSICS_DT: f64 = 0.05;

/// Warp drive acceleration in galaxy units per second squared.
/// 0.5 GU/s² = 500,000 blocks/s². Gives ~3.3 min for a 200 GU trip.
pub const WARP_ACCELERATION_GU: f64 = 0.5;

/// Maximum warp cruise speed in galaxy units per second.
pub const WARP_MAX_SPEED_GU: f64 = 50.0;

/// Engine tier definition. Two categories: maneuver (no dampening, low g)
/// and high-thrust (inertial dampener active, extreme acceleration).
#[derive(Debug, Clone, Copy)]
pub struct EngineTier {
    /// Display name for HUD.
    pub name: &'static str,
    /// Thrust force in Newtons. Ship acceleration = thrust_force_n / ship_mass_kg.
    pub thrust_force_n: f64,
    /// G-force felt by crew after inertial dampening.
    pub felt_g: f64,
    /// Whether inertial dampener is active at this tier.
    pub dampened: bool,
}

impl EngineTier {
    /// Acceleration for a ship of given mass and thrust multiplier (m/s²).
    pub fn acceleration(&self, mass_kg: f64, thrust_multiplier: f64) -> f64 {
        self.thrust_force_n * thrust_multiplier / mass_kg
    }
}

/// Engine tiers from maneuver to emergency high-thrust.
///
/// | Tier | Name        | Thrust       | Felt | ~1e11 m (10t) | ~4e11 m (10t) |
/// |------|-------------|------------- |------|---------------|---------------|
/// | 0    | Maneuver    | 49 kN        | 0.5g | —             | —             |
/// | 1    | Impulse     | 294 kN       | 3.0g | —             | —             |
/// | 2    | Cruise      | 490 MN       | 1.0g | 47 min        | 94 min        |
/// | 3    | Long Range  | 4.9 GN       | 2.0g | 15 min        | 30 min        |
/// | 4    | Emergency   | 24.5 GN      | 5.0g | 6.7 min       | 13 min        |
pub const ENGINE_TIERS: [EngineTier; 5] = [
    EngineTier { name: "Maneuver",   thrust_force_n: 49_050.0,           felt_g: 0.5, dampened: false },
    EngineTier { name: "Impulse",    thrust_force_n: 294_300.0,          felt_g: 3.0, dampened: false },
    EngineTier { name: "Cruise",     thrust_force_n: 490_500_000.0,      felt_g: 1.0, dampened: true },
    EngineTier { name: "Long Range", thrust_force_n: 4_905_000_000.0,    felt_g: 2.0, dampened: true },
    EngineTier { name: "Emergency",  thrust_force_n: 24_525_000_000.0,   felt_g: 5.0, dampened: true },
];

/// Physical properties of a ship, derived from its block composition.
/// Future: recomputed dynamically when blocks are added/removed.
#[derive(Debug, Clone)]
pub struct ShipPhysicalProperties {
    /// Total mass in kg (hull + cargo + fuel).
    pub mass_kg: f64,
    /// Per-axis cross-sectional areas in m² for orientation-dependent drag.
    /// Front/back (along Z, nose-first): width × height.
    pub cross_section_front: f64,
    /// Side (along X, broadside): length × height.
    pub cross_section_side: f64,
    /// Top/bottom (along Y, belly): width × length.
    pub cross_section_top: f64,
    /// Drag coefficient when flow is along Z (nose-first, streamlined end).
    pub cd_front: f64,
    /// Drag coefficient when flow is along X (broadside, flat plate).
    pub cd_side: f64,
    /// Drag coefficient when flow is along Y (belly, flat plate).
    pub cd_top: f64,
    /// Engine power scaling (1.0 = standard, based on thruster block count).
    pub thrust_multiplier: f64,
    /// Ship bounding box dimensions in meters (width_x, height_y, length_z).
    /// Used for moment of inertia computation.
    pub dimensions: (f64, f64, f64),
    /// Center of pressure offset behind center of mass along ship Z axis (meters).
    /// Positive = CoP behind CoM = aerodynamically stable (weathercock effect).
    pub cop_offset_z: f64,
    /// Thermal capacity in joules before structural damage begins.
    pub thermal_capacity_j: f64,
    /// Thermal emissivity for radiative cooling (0-1). Steel: ~0.8.
    pub thermal_emissivity: f64,
    /// Effective nose radius for Sutton-Graves re-entry heating (meters).
    pub nose_radius_m: f64,
    /// Landing gear height above ship bottom (meters). Defines ground contact margin.
    pub landing_gear_height: f64,
    /// Hull structural strength (arbitrary units). Determines impact damage thresholds.
    pub hull_strength: f64,
    /// Maximum thrust force on the main drive axis, forward (Newtons).
    /// Derived from thruster block count × per-block thrust. Scales with tier selection.
    pub max_thrust_forward_n: f64,
    /// Maximum thrust force on the reverse axis (Newtons).
    /// Asymmetric ships may have weaker reverse thrust than forward.
    pub max_thrust_reverse_n: f64,
    /// Maximum torque the RCS/maneuvering thrusters can apply (N·m).
    /// Derived from maneuvering thruster placement × lever arm.
    pub max_torque_nm: f64,
    /// Available engine tiers as a bitmask. Bit 0 = Maneuver, bit 4 = Emergency.
    pub available_tiers: u8,
}

impl ShipPhysicalProperties {
    /// Default properties for the starter ship (4m × 8m × 3m box, 10 tonnes).
    pub fn starter_ship() -> Self {
        let (w, h, l) = (4.0_f64, 3.0_f64, 8.0_f64);
        Self {
            mass_kg: 10_000.0,
            cross_section_front: w * h,  // 12 m²
            cross_section_side: l * h,   // 24 m²
            cross_section_top: w * l,    // 32 m²
            cd_front: 1.2,               // streamlined nose
            cd_side: 2.0,                // flat plate broadside
            cd_top: 2.0,                 // flat plate belly
            thrust_multiplier: 1.0,
            dimensions: (w, h, l),
            cop_offset_z: 1.0,           // CoP 1m behind CoM (stable)
            thermal_capacity_j: 500.0e6, // 500 MJ
            thermal_emissivity: 0.8,     // steel hull
            nose_radius_m: (w * h / std::f64::consts::PI).sqrt(), // ~1.95m
            landing_gear_height: 1.5,
            hull_strength: 100.0,
            max_thrust_forward_n: 490_500_000.0, // Cruise tier baseline
            max_thrust_reverse_n: 490_500_000.0, // symmetric
            max_torque_nm: 60_833.0, // gives ~1.0 rad/s² angular acceleration for this ship
            available_tiers: 0b11111, // all 5 tiers
        }
    }

    /// Moment of inertia for a uniform-density box about each principal axis (kg*m²).
    /// Returns (I_pitch about X, I_yaw about Y, I_roll about Z).
    pub fn moment_of_inertia(&self) -> (f64, f64, f64) {
        let (w, h, l) = self.dimensions;
        let m = self.mass_kg;
        let i_x = m * (h * h + l * l) / 12.0; // pitch
        let i_y = m * (w * w + l * l) / 12.0;  // yaw
        let i_z = m * (w * w + h * h) / 12.0;  // roll
        (i_x, i_y, i_z)
    }

    /// Maximum angular velocity (rad/s) — derived from angular acceleration and a
    /// gameplay time constant representing how long the ship accelerates before
    /// reaching terminal angular velocity (PD controller + damping equilibrium).
    /// Capped to prevent unreasonably fast rotation.
    pub fn max_angular_velocity(&self) -> f64 {
        let ang_accel = self.angular_acceleration(); // rad/s²
        // Terminal angular velocity: the PD controller reaches equilibrium at roughly
        // angular_acceleration × time_constant. The time constant represents the
        // damping response of the rotation system (~0.5s for responsive ships).
        let time_constant = 0.5; // seconds to reach terminal angular velocity
        let raw = ang_accel * time_constant;
        // Cap: 0.1 rad/s (~6 deg/s) to PI/3 rad/s (~60 deg/s)
        raw.clamp(0.1, std::f64::consts::PI / 3.0)
    }

    /// Angular acceleration rate (rad/s²) — torque / moment of inertia.
    pub fn angular_acceleration(&self) -> f64 {
        let (ix, iy, _iz) = self.moment_of_inertia();
        let i_flip = ix.min(iy);
        self.max_torque_nm / i_flip
    }

    /// Acceleration for a given engine tier (m/s²), accounting for ship mass and thrust multiplier.
    pub fn engine_acceleration(&self, tier: u8) -> f64 {
        let et = engine_tier(tier);
        et.thrust_force_n * self.thrust_multiplier / self.mass_kg
    }

    /// Whether a specific engine tier is available on this ship.
    pub fn has_tier(&self, tier: u8) -> bool {
        self.available_tiers & (1 << tier) != 0
    }

    /// Reverse acceleration for a given engine tier (m/s²).
    /// May differ from forward if the ship has asymmetric thrust.
    pub fn engine_acceleration_reverse(&self, tier: u8) -> f64 {
        let et = engine_tier(tier);
        let ratio = self.max_thrust_reverse_n / self.max_thrust_forward_n.max(1.0);
        et.thrust_force_n * self.thrust_multiplier * ratio / self.mass_kg
    }

    /// Time to rotate 180° to retrograde, simulating the actual PD controller dynamics.
    ///
    /// The PD rotation controller (P=3.0, D=1.5) saturates to max_angular_velocity for
    /// large angles, then decelerates proportionally as it approaches the target.
    /// The Flip→Brake transition fires when `dot_align > cos(15°) = 0.966` (165° rotated).
    ///
    /// This simulation matches the system-shard's actual PD controller and angular velocity
    /// integration — no magic multipliers.
    pub fn flip_duration(&self) -> f64 {
        let max_ang_vel = self.max_angular_velocity();
        let ang_accel_rate = self.angular_acceleration();
        let p_gain: f64 = 3.0;
        let d_gain: f64 = 1.5;
        let target_angle = std::f64::consts::PI; // rotate 180°
        let transition_dot = 0.966; // cos(15°) — Flip→Brake threshold
        let dt = PHYSICS_DT;

        let mut angle_remaining = target_angle; // radians still to rotate
        let mut omega: f64 = 0.0; // angular velocity (rad/s)
        let mut t: f64 = 0.0;

        // Simulate PD controller + angular velocity integration.
        // Mirrors the system-shard's rotation code (p_gain=3, d_gain=1.5,
        // torque clamp=2.0, angular accel clamped per tick).
        for _ in 0..10000 {
            // PD controller: target_omega = min(angle * P, max_ang_vel)
            let target_omega = (angle_remaining * p_gain).min(max_ang_vel);
            // Torque = (target_omega - omega) * D, clamped to 2.0
            let torque_raw = (target_omega - omega) * d_gain;
            let torque = torque_raw.clamp(-2.0, 2.0);
            // Angular velocity integration: target = torque.clamp(-1,1) * max_ang_vel
            let target_ang_vel = torque.clamp(-1.0, 1.0) * max_ang_vel;
            let diff = target_ang_vel - omega;
            let max_change = ang_accel_rate * dt;
            omega += diff.clamp(-max_change, max_change);
            // Integrate angle
            angle_remaining -= omega * dt;
            t += dt;

            // Check if we've reached the Flip→Brake threshold.
            // dot_align = cos(angle_remaining). Threshold: cos(angle_remaining) > 0.966.
            if angle_remaining.cos() > transition_dot {
                return t;
            }
            // Safety: if rotation took too long, return conservative estimate.
            if t > 60.0 {
                return t;
            }
        }
        t
    }
}

/// Autopilot engagement mode — determines behavior after reaching SOI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutopilotMode {
    /// Single-tap T: brachistochrone to SOI boundary, then disengage for manual control.
    DirectApproach,
    /// Double-tap T: brachistochrone to SOI, then circularize into orbit.
    OrbitInsertion,
    /// From orbit: deorbit burn + atmospheric entry + powered landing.
    LandingSequence,
    /// From surface: vertical ascent + gravity turn + orbit insertion.
    TakeoffSequence,
    /// From orbit: escape burn + resume interplanetary travel.
    DepartureSequence,
    /// Interstellar warp to another star system.
    WarpTravel,
}

impl AutopilotMode {
    /// Convert from wire format (u8).
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::OrbitInsertion,
            2 => Self::LandingSequence,
            3 => Self::TakeoffSequence,
            4 => Self::DepartureSequence,
            5 => Self::WarpTravel,
            _ => Self::DirectApproach,
        }
    }

    /// Convert to wire format (u8).
    pub fn to_u8(self) -> u8 {
        match self {
            Self::DirectApproach => 0,
            Self::OrbitInsertion => 1,
            Self::LandingSequence => 2,
            Self::TakeoffSequence => 3,
            Self::DepartureSequence => 4,
            Self::WarpTravel => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Orbital mechanics (planet-relative)
// ---------------------------------------------------------------------------

/// Compute Keplerian orbital elements from a ship's Cartesian state relative to a planet.
///
/// `rel_pos`: position relative to planet center (meters).
/// `rel_vel`: velocity relative to planet (meters/second).
/// `gm`: gravitational parameter of the planet (m³/s²).
/// `planet_radius`: surface radius (meters), for altitude computation.
pub fn cartesian_to_orbital_elements(
    rel_pos: DVec3,
    rel_vel: DVec3,
    gm: f64,
    planet_radius: f64,
) -> ShipOrbitalElements {
    let r = rel_pos.length();
    let v = rel_vel.length();

    if r < 1.0 || gm <= 0.0 {
        return ShipOrbitalElements {
            sma: 0.0, eccentricity: 0.0, inclination: 0.0,
            raan: 0.0, arg_periapsis: 0.0, true_anomaly: 0.0,
            periapsis_altitude: 0.0, apoapsis_altitude: 0.0,
            period: 0.0, speed: v, is_bound: false,
        };
    }

    // Specific angular momentum: h = r × v
    let h = rel_pos.cross(rel_vel);
    let h_mag = h.length();

    // Node vector: n = k × h (k = Z-axis unit vector)
    let k = DVec3::Z;
    let n = k.cross(h);
    let n_mag = n.length();

    // Eccentricity vector: e = ((v² - GM/r) * r - (r·v) * v) / GM
    let r_dot_v = rel_pos.dot(rel_vel);
    let e_vec = ((v * v - gm / r) * rel_pos - r_dot_v * rel_vel) / gm;
    let e = e_vec.length();

    // Specific orbital energy: ε = v²/2 - GM/r
    let energy = v * v / 2.0 - gm / r;

    // Semi-major axis from vis-viva: a = -GM / (2ε)
    let sma = if energy.abs() > 1e-10 {
        -gm / (2.0 * energy)
    } else {
        f64::INFINITY // parabolic
    };

    // Inclination: i = acos(h_z / |h|)
    let inclination = if h_mag > 1e-10 {
        (h.z / h_mag).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };

    // Right ascension of ascending node
    let raan = if n_mag > 1e-10 {
        let raw = (n.x / n_mag).clamp(-1.0, 1.0).acos();
        if n.y >= 0.0 { raw } else { std::f64::consts::TAU - raw }
    } else {
        0.0
    };

    // Argument of periapsis
    let arg_periapsis = if n_mag > 1e-10 && e > 1e-10 {
        let cos_omega = n.dot(e_vec) / (n_mag * e);
        let raw = cos_omega.clamp(-1.0, 1.0).acos();
        if e_vec.z >= 0.0 { raw } else { std::f64::consts::TAU - raw }
    } else {
        0.0
    };

    // True anomaly
    let true_anomaly = if e > 1e-10 {
        let cos_nu = e_vec.dot(rel_pos) / (e * r);
        let raw = cos_nu.clamp(-1.0, 1.0).acos();
        if r_dot_v >= 0.0 { raw } else { std::f64::consts::TAU - raw }
    } else {
        0.0
    };

    // Periapsis and apoapsis
    let is_bound = e < 1.0;
    let periapsis_r = if e < 1.0 - 1e-10 {
        sma * (1.0 - e)
    } else if e > 1e-10 {
        // Hyperbolic: periapsis = a * (1 - e), but a is negative
        sma.abs() * (e - 1.0)
    } else {
        r // circular
    };

    let apoapsis_r = if is_bound && sma.is_finite() {
        sma * (1.0 + e)
    } else {
        f64::INFINITY
    };

    let periapsis_altitude = periapsis_r - planet_radius;
    let apoapsis_altitude = if apoapsis_r.is_finite() {
        apoapsis_r - planet_radius
    } else {
        f64::INFINITY
    };

    // Orbital period (Kepler's third law)
    let period = if is_bound && sma > 0.0 {
        std::f64::consts::TAU * (sma.powi(3) / gm).sqrt()
    } else {
        f64::INFINITY
    };

    ShipOrbitalElements {
        sma, eccentricity: e, inclination, raan, arg_periapsis, true_anomaly,
        periapsis_altitude, apoapsis_altitude, period, speed: v, is_bound,
    }
}

/// Compute the delta-v and burn direction needed to circularize at the current radius.
///
/// Returns `(delta_v_magnitude, burn_direction_unit_vector)`.
/// Positive delta-v = prograde burn (speed up), negative = retrograde (slow down).
/// `rel_pos` and `rel_vel` are planet-relative.
pub fn circularization_delta_v(
    rel_pos: DVec3,
    rel_vel: DVec3,
    planet_gm: f64,
) -> (f64, DVec3) {
    let r = rel_pos.length();
    if r < 1.0 || planet_gm <= 0.0 {
        return (0.0, DVec3::NEG_Z);
    }

    let v_circ = (planet_gm / r).sqrt();
    let r_hat = rel_pos / r;

    // Angular momentum defines orbital plane
    let h = rel_pos.cross(rel_vel);
    // Prograde direction: perpendicular to radial, in the orbital plane
    let prograde = if h.length_squared() > 1e-10 {
        h.cross(rel_pos).normalize()
    } else {
        // Degenerate (radial trajectory): pick arbitrary tangent
        let candidate = if r_hat.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        r_hat.cross(candidate).normalize()
    };

    // Current tangential speed (positive = prograde)
    let v_prograde = rel_vel.dot(prograde);

    let delta_v = v_circ - v_prograde;
    let burn_dir = if delta_v >= 0.0 { prograde } else { -prograde };

    (delta_v.abs(), burn_dir)
}

/// Compute the retrograde delta-v to lower periapsis from a circular orbit.
///
/// Ship is assumed to be in circular orbit at `current_radius`.
/// Target: periapsis at `planet_radius + target_periapsis_alt`.
/// Returns the magnitude of the retrograde burn (always positive).
pub fn deorbit_delta_v(
    current_radius: f64,
    target_periapsis_alt: f64,
    planet_radius: f64,
    gm: f64,
) -> f64 {
    let r_peri = planet_radius + target_periapsis_alt;
    if r_peri >= current_radius || gm <= 0.0 {
        return 0.0; // already at or below target
    }

    // Transfer orbit: apoapsis = current_radius, periapsis = r_peri
    let a_transfer = (current_radius + r_peri) / 2.0;

    // Vis-viva at apoapsis of transfer orbit: v = sqrt(GM * (2/r - 1/a))
    let v_transfer = (gm * (2.0 / current_radius - 1.0 / a_transfer)).sqrt();

    // Current circular velocity
    let v_circ = (gm / current_radius).sqrt();

    // Delta-v is the difference (we need to slow down)
    v_circ - v_transfer
}

/// Compute landing guidance — all thresholds derived from planet and ship properties.
///
/// Two-phase descent:
/// 1. Deceleration burn: retrograde thrust, suicide-burn timing
/// 2. Final approach: PD controller maintaining target descent rate
pub fn compute_landing_guidance(
    ship_pos: DVec3,
    ship_vel: DVec3,
    planet_pos: DVec3,
    planet: &PlanetParams,
    props: &ShipPhysicalProperties,
    engine_accel: f64,
) -> GuidanceCommand {
    let to_center = planet_pos - ship_pos;
    let dist = to_center.length();
    let altitude = dist - planet.radius_m;
    let radial_dir = if dist > 1.0 { -to_center / dist } else { DVec3::Y }; // "up"

    let v_radial = ship_vel.dot(radial_dir);
    let v_lateral = ship_vel - radial_dir * v_radial;
    let descent_rate = -v_radial; // positive = descending
    let speed = ship_vel.length();
    let g = planet.surface_gravity;

    // Derived thresholds (no magic numbers)
    let target_descent_rate = 2.0 * g.sqrt(); // ~6.3 m/s for Earth-g
    let final_approach_alt = if planet.atmosphere.has_atmosphere {
        planet.atmosphere.scale_height * 0.05
    } else {
        (500.0_f64).min(planet.radius_m * 0.0005)
    };

    if altitude < final_approach_alt && descent_rate < target_descent_rate * 3.0 {
        // PHASE 2: Final approach — PD controller on descent rate
        let rate_error = descent_rate - target_descent_rate;
        let kp = g * 0.2;
        let required_accel = g + kp * rate_error;

        let mut thrust_dir = radial_dir;
        if v_lateral.length() > 0.5 {
            let lateral_correction = -v_lateral.normalize() * 0.5;
            thrust_dir = (thrust_dir + lateral_correction).normalize();
        }

        let thrust_mag = (required_accel * props.mass_kg).clamp(0.0, engine_accel * props.mass_kg);

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: thrust_mag,
            phase: FlightPhase::Landing,
            completed: false,
            eta_real_seconds: altitude / target_descent_rate.max(0.1),
            felt_g: thrust_mag / (props.mass_kg * 9.81),
            dampener_active: false,
            requires_flip: false,
        }
    } else if descent_rate > target_descent_rate || speed > target_descent_rate * 2.0 {
        // PHASE 1: Deceleration burn
        let thrust_dir = if speed > 0.5 { -ship_vel.normalize() } else { radial_dir };
        let excess_speed = speed - target_descent_rate;
        let decel_budget = engine_accel - g;

        let should_burn = if decel_budget > 0.0 && excess_speed > 0.0 {
            let stopping_dist = excess_speed * excess_speed / (2.0 * decel_budget);
            altitude < stopping_dist * 1.5 || descent_rate > g * 5.0
        } else {
            true // not enough thrust to hover, burn always
        };

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: if should_burn { engine_accel * props.mass_kg } else { 0.0 },
            phase: FlightPhase::TerminalDescent,
            completed: false,
            eta_real_seconds: altitude / descent_rate.max(0.1),
            felt_g: if should_burn { engine_accel / 9.81 } else { 0.0 },
            dampener_active: false,
            requires_flip: false,
        }
    } else {
        // Coast (low speed, high altitude)
        GuidanceCommand {
            thrust_direction: radial_dir,
            thrust_magnitude: 0.0,
            phase: FlightPhase::AtmosphericEntry,
            completed: false,
            eta_real_seconds: altitude / descent_rate.max(0.1),
            felt_g: 0.0,
            dampener_active: false,
            requires_flip: false,
        }
    }
}

/// Compute takeoff guidance — all altitudes derived from planet geometry.
///
/// Three-phase ascent:
/// 1. Vertical thrust from surface
/// 2. Gravity turn toward orbital velocity
/// 3. Circularize at target orbit altitude
pub fn compute_takeoff_guidance(
    ship_pos: DVec3,
    ship_vel: DVec3,
    planet_pos: DVec3,
    planet: &PlanetParams,
    props: &ShipPhysicalProperties,
    engine_accel: f64,
    target_orbit_alt: f64,
) -> GuidanceCommand {
    let to_ship = ship_pos - planet_pos;
    let dist = to_ship.length();
    let altitude = dist - planet.radius_m;
    let radial = if dist > 1.0 { to_ship / dist } else { DVec3::Y };

    // Derived thresholds
    let clear_alt = (planet.radius_m * 0.00005).clamp(100.0, 1000.0);
    let atmo_top = if planet.atmosphere.has_atmosphere {
        planet.atmosphere.atmosphere_height
    } else {
        clear_alt * 5.0
    };

    // Prograde direction in orbital plane
    let orbit_n = {
        let candidate = DVec3::Y;
        let n = radial.cross(candidate);
        if n.length() < 0.01 { radial.cross(DVec3::X).normalize() } else { n.normalize() }
    };
    let prograde = orbit_n.cross(radial).normalize();

    let v_radial = ship_vel.dot(radial);
    let v_prograde = ship_vel.dot(prograde);

    if altitude < clear_alt {
        // PHASE 1: Vertical ascent
        let mut thrust_dir = radial;
        let v_lateral = ship_vel - radial * v_radial;
        if v_lateral.length() > 1.0 {
            let correction = -v_lateral.normalize() * 0.3;
            thrust_dir = (thrust_dir + correction).normalize();
        }

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: engine_accel * props.mass_kg,
            phase: FlightPhase::Liftoff,
            completed: false,
            eta_real_seconds: 0.0,
            felt_g: engine_accel / 9.81,
            dampener_active: false,
            requires_flip: false,
        }
    } else if altitude < atmo_top * 0.95 {
        // PHASE 2: Gravity turn
        let progress = ((altitude - clear_alt) / (atmo_top - clear_alt).max(1.0)).clamp(0.0, 1.0);
        let pitch_over = progress.powf(0.7); // 0 = vertical, 1 = horizontal
        let thrust_dir = (radial * (1.0 - pitch_over) + prograde * pitch_over).normalize();

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: engine_accel * props.mass_kg,
            phase: FlightPhase::GravityTurn,
            completed: false,
            eta_real_seconds: 0.0,
            felt_g: engine_accel / 9.81,
            dampener_active: false,
            requires_flip: false,
        }
    } else {
        // PHASE 3: Circularize
        let v_circ = (planet.gm / dist).sqrt();
        let deficit = v_circ - v_prograde;

        if deficit > 10.0 {
            GuidanceCommand {
                thrust_direction: prograde,
                thrust_magnitude: engine_accel * props.mass_kg,
                phase: FlightPhase::AscentBurn,
                completed: false,
                eta_real_seconds: deficit / engine_accel,
                felt_g: engine_accel / 9.81,
                dampener_active: false,
                requires_flip: false,
            }
        } else {
            GuidanceCommand {
                thrust_direction: prograde,
                thrust_magnitude: 0.0,
                phase: FlightPhase::StableOrbit,
                completed: true,
                eta_real_seconds: 0.0,
                felt_g: 0.0,
                dampener_active: false,
                requires_flip: false,
            }
        }
    }
}

/// Check if a flight phase transition should occur.
///
/// All thresholds are relative to planet/ship properties — no magic numbers.
/// Deterministic: shared between client (HUD preview) and server (authoritative).
pub fn check_phase_transition(
    current_phase: FlightPhase,
    mode: AutopilotMode,
    ship_pos: DVec3,
    ship_vel: DVec3,
    planet_pos: DVec3,
    planet_vel: DVec3,
    planet: &PlanetParams,
    star: &StarParams,
    soi_radius: f64,
    target_orbit_alt: f64,
    props: &ShipPhysicalProperties,
) -> FlightPhase {
    let rel_pos = ship_pos - planet_pos;
    let rel_vel = ship_vel - planet_vel;
    let dist = rel_pos.length();
    let altitude = dist - planet.radius_m;
    let speed = rel_vel.length();

    match current_phase {
        FlightPhase::Brake | FlightPhase::Arrived => {
            if dist < soi_radius {
                match mode {
                    AutopilotMode::OrbitInsertion | AutopilotMode::LandingSequence => FlightPhase::SoiApproach,
                    _ => FlightPhase::Arrived,
                }
            } else {
                current_phase
            }
        }
        FlightPhase::SoiApproach => {
            let v_circ = circular_orbit_velocity(planet, altitude);
            // Circularize at whatever altitude the ship is when speed is manageable.
            // Ship establishes a high orbit first, then DeorbitBurn descends to target.
            if speed < v_circ * 2.0 {
                FlightPhase::CircularizeBurn
            } else {
                current_phase
            }
        }
        FlightPhase::CircularizeBurn => {
            let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);
            if oe.eccentricity < 0.02 && oe.is_bound {
                match mode {
                    AutopilotMode::LandingSequence => FlightPhase::DeorbitBurn,
                    _ => FlightPhase::StableOrbit,
                }
            } else {
                current_phase
            }
        }
        FlightPhase::StableOrbit => {
            match mode {
                AutopilotMode::LandingSequence => FlightPhase::DeorbitBurn,
                AutopilotMode::DepartureSequence => FlightPhase::EscapeBurn,
                _ => current_phase,
            }
        }
        FlightPhase::DeorbitBurn => {
            let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);
            if oe.periapsis_altitude < planet.atmosphere.atmosphere_height || (!planet.atmosphere.has_atmosphere && oe.periapsis_altitude < target_orbit_alt * 0.1) {
                FlightPhase::AtmosphericEntry
            } else {
                current_phase
            }
        }
        FlightPhase::AtmosphericEntry => {
            let threshold_alt = if planet.atmosphere.has_atmosphere {
                planet.atmosphere.scale_height * 0.5
            } else {
                planet.radius_m * 0.001
            };
            let threshold_speed = circular_orbit_velocity(planet, 0.0) * 0.1;
            if altitude < threshold_alt || speed < threshold_speed {
                FlightPhase::TerminalDescent
            } else {
                current_phase
            }
        }
        FlightPhase::TerminalDescent => {
            if altitude < props.landing_gear_height + 1.0 && speed < 0.5 {
                FlightPhase::Landed
            } else {
                current_phase
            }
        }
        FlightPhase::Landed => {
            match mode {
                AutopilotMode::TakeoffSequence => FlightPhase::Liftoff,
                _ => current_phase,
            }
        }
        FlightPhase::Liftoff => {
            let clear_alt = (planet.radius_m * 0.00005).clamp(100.0, 1000.0);
            if altitude > clear_alt {
                FlightPhase::GravityTurn
            } else {
                current_phase
            }
        }
        FlightPhase::GravityTurn => {
            let atmo_top = if planet.atmosphere.has_atmosphere {
                planet.atmosphere.atmosphere_height
            } else {
                (planet.radius_m * 0.00005).clamp(100.0, 1000.0) * 5.0
            };
            if altitude > atmo_top * 0.95 {
                FlightPhase::AscentBurn
            } else {
                current_phase
            }
        }
        FlightPhase::AscentBurn => {
            let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);
            if oe.eccentricity < 0.02 && oe.is_bound {
                FlightPhase::StableOrbit
            } else {
                current_phase
            }
        }
        FlightPhase::EscapeBurn => {
            if dist > soi_radius {
                FlightPhase::Accelerate
            } else {
                current_phase
            }
        }
        // Interplanetary phases don't transition here
        _ => current_phase,
    }
}

/// Maximum engine tier allowed inside a planet's atmosphere.
/// Dampened tiers (Cruise/Long Range/Emergency) are blocked.
pub fn max_tier_in_atmosphere() -> u8 {
    1 // Impulse
}

/// Clamp a requested engine tier based on whether the ship is in atmosphere.
pub fn effective_tier(requested: u8, in_atmosphere: bool) -> u8 {
    if in_atmosphere {
        requested.min(max_tier_in_atmosphere())
    } else {
        requested
    }
}

/// Circular orbit velocity at a given altitude above a planet's surface.
/// Uses brahe's periapsis_velocity with e=0 for circular orbits.
pub fn circular_orbit_velocity(planet: &crate::system::PlanetParams, altitude: f64) -> f64 {
    let r = planet.radius_m + altitude;
    // v_circ = sqrt(GM/r) — equivalent to brahe::periapsis_velocity(r, 0, gm)
    (planet.gm / r).sqrt()
}

/// Escape velocity at a given altitude above a planet's surface.
pub fn escape_velocity(planet: &crate::system::PlanetParams, altitude: f64) -> f64 {
    circular_orbit_velocity(planet, altitude) * std::f64::consts::SQRT_2
}

/// Orbital period at a given altitude above a planet's surface (seconds).
pub fn orbital_period(planet: &crate::system::PlanetParams, altitude: f64) -> f64 {
    let r = planet.radius_m + altitude;
    std::f64::consts::TAU * (r.powi(3) / planet.gm).sqrt()
}

/// Flight phase of the autopilot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightPhase {
    // -- Interplanetary transit --
    /// Thrusting toward intercept point.
    Accelerate,
    /// Rotating ship 180 degrees for braking burn.
    Flip,
    /// Decelerating toward target.
    Brake,
    /// Arrived at SOI boundary (used by DirectApproach mode).
    Arrived,

    // -- Orbital operations --
    /// Inside SOI, decelerating to orbital velocity.
    SoiApproach,
    /// Executing prograde/retrograde burn to circularize.
    CircularizeBurn,
    /// In stable orbit, no thrust (gravity maintains trajectory).
    StableOrbit,

    // -- Descent --
    /// Retrograde burn to lower periapsis into atmosphere.
    DeorbitBurn,
    /// In atmosphere, aerobraking or controlled aerodynamic flight.
    AtmosphericEntry,
    /// Below terminal descent altitude, powered deceleration.
    TerminalDescent,
    /// Final approach, precision altitude control.
    Landing,
    /// On the planet surface, engines off.
    Landed,

    // -- Ascent --
    /// Vertical thrust from surface.
    Liftoff,
    /// Pitching from vertical toward orbital velocity.
    GravityTurn,
    /// Circularizing from ascent trajectory.
    AscentBurn,
    /// Prograde burn to exceed escape velocity.
    EscapeBurn,

    // -- Interstellar warp --
    /// Aligning ship heading toward target star direction.
    WarpAlign,
    /// Accelerating toward system SOI boundary (still in system shard).
    WarpAccelerate,
    /// Cruising at warp speed in interstellar space (galaxy shard).
    WarpCruise,
    /// Decelerating toward destination star SOI (galaxy shard).
    WarpDecelerate,
    /// Entering destination system SOI.
    WarpArrival,
}

/// Orbital elements computed from a ship's state vector relative to a planet.
/// All distances in meters, angles in radians, time in seconds.
#[derive(Debug, Clone)]
pub struct ShipOrbitalElements {
    /// Semi-major axis (meters). Negative for hyperbolic trajectories.
    pub sma: f64,
    /// Eccentricity. 0 = circular, <1 = elliptical, >=1 = hyperbolic/parabolic.
    pub eccentricity: f64,
    /// Orbital inclination (radians).
    pub inclination: f64,
    /// Right ascension of ascending node (radians).
    pub raan: f64,
    /// Argument of periapsis (radians).
    pub arg_periapsis: f64,
    /// True anomaly (radians).
    pub true_anomaly: f64,
    /// Periapsis altitude above planet surface (meters).
    pub periapsis_altitude: f64,
    /// Apoapsis altitude above planet surface (meters). `f64::INFINITY` for hyperbolic.
    pub apoapsis_altitude: f64,
    /// Orbital period (seconds). `f64::INFINITY` for hyperbolic.
    pub period: f64,
    /// Current orbital speed (m/s).
    pub speed: f64,
    /// Whether the orbit is bound (eccentricity < 1).
    pub is_bound: bool,
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
    /// Whether the ship needs to flip 180° (transitioning from Accelerate to Brake).
    pub requires_flip: bool,
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
// Intercept solver (Lambert-driven)
// ---------------------------------------------------------------------------

/// Result of the Lambert-driven intercept solver.
#[derive(Debug, Clone)]
pub struct InterceptSolution {
    /// Position on the SOI boundary where the ship will enter.
    pub intercept_pos: DVec3,
    /// Estimated time of flight in real seconds.
    pub tof_real_seconds: f64,
    /// Total delta-v budget (departure + arrival).
    pub total_delta_v: f64,
    /// Planet's real-time velocity at the predicted arrival time (m/s per real second).
    /// Ship should match this velocity to enter SOI with near-zero relative speed.
    pub arrival_planet_vel: DVec3,
}

/// Compute powered-flight time-of-flight from a delta-v budget.
///
/// Given the delta-v needed to depart and arrive, plus forward/reverse acceleration
/// and flip duration, returns the total real-time TOF.
fn powered_flight_tof(
    departure_dv: f64,
    arrival_dv: f64,
    accel_fwd: f64,
    accel_rev: f64,
    flip_time: f64,
) -> f64 {
    let t_accel = if accel_fwd > 0.0 { departure_dv / accel_fwd } else { 0.0 };
    let t_brake = if accel_rev > 0.0 { arrival_dv / accel_rev } else { 0.0 };
    t_accel + flip_time + t_brake
}

/// Solve for the intercept point using Lambert's method and iterative TOF convergence.
///
/// Returns an `InterceptSolution` with the gravitationally-correct departure direction,
/// intercept position on the SOI boundary, and time-of-flight estimate.
///
/// Falls back to straight-line brachistochrone if Lambert fails (e.g., singular transfer).
pub fn solve_intercept(
    ship_pos: DVec3,
    ship_vel: DVec3,
    target_planet: &PlanetParams,
    star: &StarParams,
    system: &SystemParams,
    planet_positions: &[DVec3],
    celestial_time: f64,
    time_scale: f64,
    ship_props: &ShipPhysicalProperties,
    thrust_tier: u8,
) -> Option<InterceptSolution> {
    let accel_fwd = ship_props.engine_acceleration(thrust_tier);
    let accel_rev = ship_props.engine_acceleration_reverse(thrust_tier);
    let flip_time = ship_props.flip_duration();
    let soi = compute_soi_radius(target_planet, star);

    // Current target position.
    let target_now = compute_planet_position(target_planet, celestial_time);
    let d0 = (target_now - ship_pos).length();
    if d0 < soi {
        // Already inside SOI — velocity is already planet-relative.
        return Some(InterceptSolution {
            intercept_pos: target_now,
            tof_real_seconds: 0.0,
            total_delta_v: 0.0,
            arrival_planet_vel: DVec3::ZERO,
        });
    }

    // Seed TOF with classic brachistochrone (straight-line, no gravity).
    let dir0 = (target_now - ship_pos).normalize_or_zero();
    let v_along = ship_vel.dot(dir0);
    let mut tof_real = brachistochrone_tof(d0 - soi, v_along, accel_fwd);
    if tof_real <= 0.0 {
        tof_real = 1.0; // minimum seed
    }

    let mut best_intercept_pos = target_now - dir0 * soi;
    let mut best_total_dv = 0.0;
    let mut best_arrival_vel = compute_planet_velocity(target_planet, star.gm, celestial_time)
        * time_scale;

    // Iteratively refine: planet moves during transit, Lambert gives curved direction.
    for _ in 0..20 {
        let arrival_celestial = celestial_time + tof_real * time_scale;
        let planet_future = compute_planet_position(target_planet, arrival_celestial);

        // SOI entry point on the near side of the SOI sphere.
        let approach_dir = (planet_future - ship_pos).normalize_or_zero();
        let intercept_pos = planet_future - approach_dir * soi;

        // Try Lambert: ship_pos → intercept_pos under star gravity.
        // Lambert works in celestial time (same time domain as GM and orbital elements).
        let tof_celestial = tof_real * time_scale;
        if tof_celestial > 0.0 {
            let r1 = [ship_pos.x, ship_pos.y, ship_pos.z];
            let r2 = [intercept_pos.x, intercept_pos.y, intercept_pos.z];
            match gooding_lambert::lambert(
                star.gm,
                r1,
                r2,
                tof_celestial,
                0,
                gooding_lambert::Direction::Prograde,
                gooding_lambert::MultiRevPeriod::LongPeriod,
            ) {
                Ok(sol) => {
                    // Lambert velocities are in m/celestial_s. Convert to m/real_s.
                    let v1 = DVec3::new(sol.v1[0], sol.v1[1], sol.v1[2]) * time_scale;
                    let v2 = DVec3::new(sol.v2[0], sol.v2[1], sol.v2[2]) * time_scale;

                    // Delta-v at departure: change from current velocity to Lambert departure.
                    let departure_dv = (v1 - ship_vel).length();

                    // Planet velocity at arrival time for SOI matching.
                    let planet_vel_future =
                        compute_planet_velocity(target_planet, star.gm, arrival_celestial)
                            * time_scale;
                    let arrival_dv = (v2 - planet_vel_future).length();

                    // Powered TOF from delta-v budget.
                    let tof_new = powered_flight_tof(
                        departure_dv, arrival_dv, accel_fwd, accel_rev, flip_time,
                    );

                    best_intercept_pos = intercept_pos;
                    best_total_dv = departure_dv + arrival_dv;
                    best_arrival_vel = planet_vel_future;

                    // Convergence check.
                    if tof_new > 0.0
                        && (tof_new - tof_real).abs() < 0.1 + tof_real * 0.001
                    {
                        tof_real = tof_new;
                        break;
                    }
                    if tof_new > 0.0 {
                        tof_real = tof_new;
                    }
                }
                Err(gooding_lambert::LambertError::SingularTransfer) => {
                    // 180° transfer angle — perturb slightly and retry next iteration.
                    // The iteration will use a slightly different TOF, changing the geometry.
                    let d = (intercept_pos - ship_pos).length();
                    let v_along = ship_vel.dot(approach_dir);
                    let tof_new = brachistochrone_tof(d, v_along, accel_fwd);
                    if tof_new > 0.0 {
                        best_arrival_vel = compute_planet_velocity(
                            target_planet, star.gm, arrival_celestial,
                        ) * time_scale;
                        tof_real = tof_new * 1.01; // small perturbation
                    }
                }
                Err(_) => {
                    // Lambert failed — fall back to straight-line brachistochrone.
                    let d = (intercept_pos - ship_pos).length();
                    let v_along = ship_vel.dot(approach_dir);
                    let tof_new = brachistochrone_tof(d, v_along, accel_fwd);
                    if tof_new > 0.0 {
                        best_intercept_pos = intercept_pos;
                        best_arrival_vel = compute_planet_velocity(
                            target_planet, star.gm, arrival_celestial,
                        ) * time_scale;
                        if (tof_new - tof_real).abs() < 0.1 + tof_real * 0.001 {
                            tof_real = tof_new;
                            break;
                        }
                        tof_real = tof_new;
                    }
                }
            }
        } else {
            // Fallback: straight-line brachistochrone.
            let d = (intercept_pos - ship_pos).length();
            let v_along = ship_vel.dot(approach_dir);
            let tof_new = brachistochrone_tof(d, v_along, accel_fwd);
            if tof_new > 0.0 {
                best_intercept_pos = intercept_pos;
                best_arrival_vel = compute_planet_velocity(
                    target_planet, star.gm, arrival_celestial,
                ) * time_scale;
                if (tof_new - tof_real).abs() < 0.1 + tof_real * 0.001 {
                    tof_real = tof_new;
                    break;
                }
                tof_real = tof_new;
            }
        }
    }

    Some(InterceptSolution {
        intercept_pos: best_intercept_pos,
        tof_real_seconds: tof_real,
        total_delta_v: best_total_dv,
        arrival_planet_vel: best_arrival_vel,
    })
}

// ---------------------------------------------------------------------------
// Per-tick guidance (gravity-aware)
// ---------------------------------------------------------------------------

/// Compute the thrust command for the current tick.
///
/// Uses the Lambert-derived departure direction for the acceleration phase,
/// gravity-aware stopping distance for the brake trigger, and accounts for
/// flip coast time. All thresholds derived from ship properties — no magic numbers.
/// Whether the ship is inside a planet's SOI for gravity model selection.
/// When inside SOI, guidance uses planet-only gravity (matching the physics
/// integrator). Outside SOI, full N-body gravity is used.
pub fn compute_guidance(
    ship_pos: DVec3,
    ship_vel: DVec3,
    intercept_pos: DVec3,
    target_vel: DVec3,
    target_planet_index: usize,
    system: &SystemParams,
    planet_positions: &[DVec3],
    celestial_time: f64,
    ship_props: &ShipPhysicalProperties,
    thrust_tier: u8,
    in_soi: bool,
) -> GuidanceCommand {
    let et = engine_tier(thrust_tier);
    let accel_fwd = ship_props.engine_acceleration(thrust_tier);
    let accel_rev = ship_props.engine_acceleration_reverse(thrust_tier);
    let thrust_force = et.thrust_force_n * ship_props.thrust_multiplier;

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
            requires_flip: false,
        };
    }

    // Always use straight-line direction to intercept for guidance calculations.
    let desired_dir = to_intercept / dist_to_intercept;

    // Gravity at current position — must match the physics integrator's model.
    // Inside SOI: planet-only gravity (star handled by co-movement frame).
    // Outside SOI: full N-body gravity.
    let gravity = if in_soi {
        let r = ship_pos - planet_positions[target_planet_index];
        let dist_sq = r.length_squared();
        if dist_sq > 1.0 {
            -r.normalize() * system.planets[target_planet_index].gm / dist_sq
        } else {
            DVec3::ZERO
        }
    } else {
        compute_gravity_acceleration(
            ship_pos,
            &system.star,
            &system.planets,
            planet_positions,
            celestial_time,
        )
    };

    // Decompose gravity into along-track and perpendicular components.
    let grav_along = gravity.dot(desired_dir);
    let grav_perp = gravity - desired_dir * grav_along;

    // Relative velocity: what needs to be killed to match the target velocity at arrival.
    // For interplanetary approach, target_vel = planet orbital velocity.
    // For SOI approach (already planet-relative), target_vel = ZERO.
    let rel_vel = ship_vel - target_vel;

    // Absolute approach velocity (for position-closing check).
    let v_closing = ship_vel.dot(desired_dir);

    // Relative velocity components (for stopping distance and brake intensity).
    let speed = rel_vel.length();
    let v_along = rel_vel.dot(desired_dir);

    // --- Stopping distance computation (no magic numbers) ---
    //
    // --- Stopping distance (derived from ship physics, no magic numbers) ---
    //
    // Phase 1: Flip coast — ship rotates 180° with effectively zero thrust.
    // Duration derived from angular acceleration and max angular velocity.
    //
    // Phase 2: Kinematic braking — full thrust retrograde.
    // Distance = v_closing_after_flip * speed_after_flip / (2 * a_brake)
    // because the engine kills the FULL relative velocity (closing + tangential).

    let g_along = gravity.dot(desired_dir);

    // Flip coast time: simulated from ship's PD controller dynamics.
    // Accounts for proportional deceleration near the target angle.
    let t_flip_coast = ship_props.flip_duration();

    // Distance toward intercept during flip coast (zero thrust, gravity only).
    let v_closing_after_flip = v_closing + g_along * t_flip_coast;
    let flip_coast_along = v_closing * t_flip_coast + 0.5 * g_along * t_flip_coast * t_flip_coast;

    // Relative speed after flip coast (what the engine must kill).
    let vel_after_flip = rel_vel + gravity * t_flip_coast;
    let speed_after_flip = vel_after_flip.length();

    // Braking deceleration along approach direction.
    let g_along_brake = -gravity.dot(desired_dir);
    let a_brake_along = (accel_rev + g_along_brake).max(accel_rev * 0.1);

    // Kinematic braking distance toward the intercept.
    // Engine thrusts retrograde (-rel_vel), time to stop = speed/a_brake.
    // Distance toward intercept = v_closing * speed / (2 * a_brake).
    let brake_dist_along = if v_closing_after_flip > 0.0 && speed_after_flip > 1.0 {
        v_closing_after_flip * speed_after_flip / (2.0 * a_brake_along)
    } else {
        0.0
    };

    let total_stopping_dist = flip_coast_along.max(0.0) + brake_dist_along;

    // ETA estimate.
    let eta_real = brachistochrone_tof(dist_to_intercept, v_closing, accel_fwd);

    // Phase determination: brake when stopping distance reaches remaining distance.
    if total_stopping_dist >= dist_to_intercept && v_closing > 0.0 {
        // BRAKE phase: thrust retrograde relative to target velocity.
        // Kills the excess velocity to match the target at the intercept.
        let thrust_dir = if speed > 0.1 {
            -rel_vel.normalize()
        } else {
            -desired_dir
        };

        GuidanceCommand {
            thrust_direction: thrust_dir,
            thrust_magnitude: thrust_force,
            phase: FlightPhase::Brake,
            completed: false,
            eta_real_seconds: eta_real,
            felt_g: et.felt_g,
            dampener_active: et.dampened,
            requires_flip: true,
        }
    } else {
        // ACCELERATE phase: thrust directly toward intercept point.
        // No separate gravity or perpendicular velocity corrections needed —
        // desired_dir recalculates each tick from (intercept_pos - ship_pos),
        // which naturally compensates for any drift. The PD rotation controller
        // handles smooth alignment. Separate corrections caused oscillation.
        GuidanceCommand {
            thrust_direction: desired_dir,
            thrust_magnitude: thrust_force,
            phase: FlightPhase::Accelerate,
            completed: false,
            eta_real_seconds: eta_real,
            felt_g: et.felt_g,
            dampener_active: et.dampened,
            requires_flip: false,
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
    ship_props: &ShipPhysicalProperties,
    thrust_tier: u8,
    sample_count: usize,
) -> Option<TrajectoryPlan> {
    if target_planet_index >= system.planets.len() {
        return None;
    }

    let planet = &system.planets[target_planet_index];
    let star = &system.star;
    let soi = compute_soi_radius(planet, star);

    // Compute planet positions for solve_intercept.
    let planet_positions: Vec<DVec3> = system
        .planets
        .iter()
        .map(|p| compute_planet_position(p, celestial_time))
        .collect();

    // Solve intercept.
    let sol = solve_intercept(
        ship_pos,
        ship_vel,
        planet,
        star,
        system,
        &planet_positions,
        celestial_time,
        system.scale.time_scale,
        ship_props,
        thrust_tier,
    )?;
    let intercept_pos = sol.intercept_pos;
    let eta_real = sol.tof_real_seconds;
    let target_vel = sol.arrival_planet_vel;

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
        let (current_intercept, current_target_vel) = if i % 20 == 0 {
            solve_intercept(
                pos, vel, planet, star, system, &planet_positions,
                ct, system.scale.time_scale, ship_props, thrust_tier,
            )
            .map(|s| (s.intercept_pos, s.arrival_planet_vel))
            .unwrap_or((intercept_pos, target_vel))
        } else {
            (intercept_pos, target_vel)
        };

        // Guidance at this point (interplanetary — not inside SOI).
        let guidance = compute_guidance(
            pos, vel, current_intercept, current_target_vel, target_planet_index,
            system, &planet_positions, ct, ship_props, thrust_tier, false,
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
        let thrust_accel = guidance.thrust_direction * (guidance.thrust_magnitude / ship_props.mass_kg);
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
        ship_pos, ship_vel, intercept_pos, target_vel, target_planet_index,
        system, &planet_positions_now, celestial_time, ship_props, thrust_tier, false,
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
                ENGINE_TIERS[i].thrust_force_n > ENGINE_TIERS[i - 1].thrust_force_n,
                "tier {} thrust not greater than tier {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn engine_tier_thrust_forces() {
        assert!((ENGINE_TIERS[0].thrust_force_n - 49_050.0).abs() < 1.0, "tier 0 should be ~49 kN");
        assert!((ENGINE_TIERS[1].thrust_force_n - 294_300.0).abs() < 1.0, "tier 1 should be ~294 kN");
        assert!((ENGINE_TIERS[2].thrust_force_n - 490_500_000.0).abs() < 100.0, "tier 2 should be ~490 MN");
        assert!((ENGINE_TIERS[3].thrust_force_n - 4_905_000_000.0).abs() < 1000.0, "tier 3 should be ~4.9 GN");
        assert!((ENGINE_TIERS[4].thrust_force_n - 24_525_000_000.0).abs() < 1000.0, "tier 4 should be ~24.5 GN");
    }

    #[test]
    fn engine_acceleration_mass_dependent() {
        let props = ShipPhysicalProperties::starter_ship(); // 10,000 kg
        // Starter ship at Cruise tier: 490_500_000 N / 10_000 kg = 49,050 m/s²
        let accel = props.engine_acceleration(2);
        assert!((accel - 49_050.0).abs() < 1.0, "starter ship cruise accel should be ~49050, got {accel}");

        // Heavy ship: double mass = half acceleration
        let mut heavy = props.clone();
        heavy.mass_kg = 20_000.0;
        let accel_heavy = heavy.engine_acceleration(2);
        assert!((accel_heavy - 24_525.0).abs() < 1.0, "heavy ship cruise accel should be ~24525, got {accel_heavy}");
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

    fn test_ship_props() -> ShipPhysicalProperties {
        ShipPhysicalProperties::starter_ship()
    }

    #[test]
    fn solve_intercept_converges() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;
        let props = test_ship_props();

        // Ship at spawn offset from planet 0.
        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);
        let ship_vel = DVec3::ZERO;

        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let result = solve_intercept(
            ship_pos, ship_vel, planet, star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 1,
        );

        assert!(result.is_some(), "intercept should converge");
        let sol = result.unwrap();
        assert!(sol.tof_real_seconds > 0.0, "TOF should be positive");
        assert!(sol.intercept_pos.length() > 0.0, "intercept should be non-zero");

        // Intercept should be near where the planet will be.
        let arrival_celestial = 0.0 + sol.tof_real_seconds * sys.scale.time_scale;
        let planet_future = compute_planet_position(planet, arrival_celestial);
        let soi = compute_soi_radius(planet, star);
        let dist_to_planet = (sol.intercept_pos - planet_future).length();
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
        let props = test_ship_props();

        // Ship inside planet SOI.
        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(100.0, 0.0, 0.0);

        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let result = solve_intercept(
            ship_pos, DVec3::ZERO, planet, star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 1,
        );

        assert!(result.is_some());
        let sol = result.unwrap();
        assert!(sol.tof_real_seconds < 1.0, "TOF should be near zero when already at target");
    }

    #[test]
    fn guidance_accelerate_phase() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;
        let props = test_ship_props();

        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);
        let ship_vel = DVec3::ZERO;

        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let sol = solve_intercept(
            ship_pos, ship_vel, planet, star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 1,
        ).unwrap();

        let cmd = compute_guidance(
            ship_pos, ship_vel, sol.intercept_pos, sol.arrival_planet_vel,
            0, &sys, &planet_positions, 0.0, &props, 1, false,
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
        let props = test_ship_props();

        let planet_pos = compute_planet_position(planet, 0.0);
        let soi = compute_soi_radius(planet, star);
        // Ship close to target at high speed.
        // Impulse tier: 294_300 N / 10_000 kg = 29.43 m/s²
        // With flip coast (~6s) and gravity corrections, we need sufficient speed
        // to trigger braking at 20 km distance.
        let remaining = 20_000.0;
        let ship_pos = planet_pos + DVec3::new(soi + remaining, 0.0, 0.0);
        let intercept_pos = planet_pos + DVec3::new(soi, 0.0, 0.0);
        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        // v=1200 m/s relative to target should trigger brake with flip coast margin.
        let ship_vel_fast = DVec3::new(-1_200.0, 0.0, 0.0);
        let cmd = compute_guidance(
            ship_pos, ship_vel_fast, intercept_pos, DVec3::ZERO,
            0, &sys, &planet_positions, 0.0, &props, 1, false,
        );

        assert_eq!(cmd.phase, FlightPhase::Brake);
    }

    #[test]
    fn guidance_inside_soi_does_not_prematurely_arrive() {
        // SOI arrival is handled by the phase state machine (check_phase_transition),
        // not by compute_guidance. Guidance should keep steering, not mark Arrived.
        let sys = test_system();
        let props = test_ship_props();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let soi = compute_soi_radius(&sys.planets[0], &sys.star);

        // Ship inside SOI, moving toward planet.
        let ship_pos = planet_pos + DVec3::new(soi * 0.5, 0.0, 0.0);
        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let cmd = compute_guidance(
            ship_pos, DVec3::ZERO, planet_pos, DVec3::ZERO,
            0, &sys, &planet_positions, 0.0, &props, 1, true,
        );

        // Should NOT mark as completed — phase machine handles SOI transitions.
        assert!(!cmd.completed, "guidance should not prematurely disengage inside SOI");
    }

    #[test]
    fn plan_trajectory_produces_points() {
        let sys = test_system();
        let props = test_ship_props();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);

        // Use Cruise tier (2) for trajectory planning — Impulse is too slow to
        // demonstrate full accel/brake/arrive within a manageable sample window.
        let plan = plan_trajectory(ship_pos, DVec3::ZERO, 0, &sys, 0.0, &props, 2, 400);

        assert!(plan.is_some(), "trajectory plan should succeed");
        let plan = plan.unwrap();
        assert!(!plan.points.is_empty(), "should have trajectory points");
        assert!(plan.eta_real_seconds > 0.0, "ETA should be positive");
        assert!(plan.target_soi_radius > 0.0);

        let has_accel = plan
            .points
            .iter()
            .any(|p| p.phase == FlightPhase::Accelerate);
        let has_brake = plan.points.iter().any(|p| p.phase == FlightPhase::Brake);
        let has_arrived = plan.points.iter().any(|p| p.phase == FlightPhase::Arrived);
        assert!(has_accel, "should have acceleration phase");
        // At Cruise tier, the trajectory should show braking or SOI arrival.
        if plan.points.len() > 10 {
            assert!(has_brake || has_arrived,
                "long trajectory should have braking or arrival phase");
        }
    }

    #[test]
    fn plan_trajectory_invalid_target() {
        let sys = test_system();
        let props = test_ship_props();
        let plan = plan_trajectory(DVec3::ZERO, DVec3::ZERO, 99, &sys, 0.0, &props, 1, 200);
        assert!(plan.is_none(), "invalid target should return None");
    }

    #[test]
    fn higher_thrust_tier_faster_eta() {
        let sys = test_system();
        let props = test_ship_props();
        let planet_pos = compute_planet_position(&sys.planets[0], 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);

        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let eta_impulse = solve_intercept(
            ship_pos, DVec3::ZERO, &sys.planets[0], &sys.star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 1,
        ).unwrap().tof_real_seconds;

        let eta_cruise = solve_intercept(
            ship_pos, DVec3::ZERO, &sys.planets[0], &sys.star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 2,
        ).unwrap().tof_real_seconds;

        assert!(
            eta_cruise < eta_impulse,
            "Cruise ({:.0}s) should be faster than Impulse ({:.0}s)",
            eta_cruise,
            eta_impulse
        );
    }

    // -- Orbital math tests --

    #[test]
    fn orbital_elements_circular_orbit() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let altitude = 100_000.0; // 100 km
        let r = planet.radius_m + altitude;

        // Circular orbit: velocity perpendicular to radial, magnitude = v_circ
        let v_circ = (planet.gm / r).sqrt();
        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_circ, 0.0); // prograde in Y

        let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);

        assert!(oe.is_bound, "circular orbit should be bound");
        assert!(oe.eccentricity < 0.01, "eccentricity should be ~0, got {}", oe.eccentricity);
        assert!((oe.sma - r).abs() / r < 0.01, "SMA should equal radius for circular orbit");
        assert!((oe.periapsis_altitude - altitude).abs() / altitude < 0.01,
            "periapsis alt should be ~{altitude}, got {}", oe.periapsis_altitude);
        assert!((oe.apoapsis_altitude - altitude).abs() / altitude < 0.01,
            "apoapsis alt should be ~{altitude}, got {}", oe.apoapsis_altitude);
        assert!(oe.period > 0.0 && oe.period.is_finite(), "period should be finite");
    }

    #[test]
    fn orbital_elements_elliptical() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 200_000.0; // 200 km altitude

        // Elliptical: give ship 120% of circular velocity (will have higher apoapsis)
        let v_circ = (planet.gm / r).sqrt();
        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_circ * 1.2, 0.0);

        let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);

        assert!(oe.is_bound, "should be bound (below escape velocity)");
        assert!(oe.eccentricity > 0.01, "should be elliptical, e={}", oe.eccentricity);
        assert!(oe.eccentricity < 1.0, "should not be hyperbolic");
        assert!(oe.apoapsis_altitude > oe.periapsis_altitude,
            "apoapsis should be higher than periapsis");
    }

    #[test]
    fn orbital_elements_hyperbolic() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 100_000.0;

        // Hyperbolic: give ship 200% of circular velocity (above escape)
        let v_circ = (planet.gm / r).sqrt();
        let v_escape = v_circ * std::f64::consts::SQRT_2;
        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_escape * 1.5, 0.0);

        let oe = cartesian_to_orbital_elements(rel_pos, rel_vel, planet.gm, planet.radius_m);

        assert!(!oe.is_bound, "should be unbound (hyperbolic)");
        assert!(oe.eccentricity >= 1.0, "eccentricity should be >= 1, got {}", oe.eccentricity);
        assert!(oe.sma < 0.0, "SMA should be negative for hyperbolic");
        assert!(oe.apoapsis_altitude.is_infinite(), "apoapsis should be infinite");
        assert!(oe.period.is_infinite(), "period should be infinite");
    }

    #[test]
    fn circularization_already_circular() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 100_000.0;
        let v_circ = (planet.gm / r).sqrt();

        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_circ, 0.0);

        let (dv, _) = circularization_delta_v(rel_pos, rel_vel, planet.gm);
        assert!(dv < 1.0, "delta-v should be near zero for circular orbit, got {dv}");
    }

    #[test]
    fn circularization_too_fast() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 100_000.0;
        let v_circ = (planet.gm / r).sqrt();

        // Ship going 20% too fast
        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_circ * 1.2, 0.0);

        let (dv, burn_dir) = circularization_delta_v(rel_pos, rel_vel, planet.gm);
        assert!(dv > 10.0, "delta-v should be significant, got {dv}");
        // Burn should be retrograde (opposing velocity direction)
        assert!(burn_dir.dot(rel_vel.normalize()) < -0.9,
            "burn should be retrograde, dot = {}", burn_dir.dot(rel_vel.normalize()));
    }

    #[test]
    fn circularization_too_slow() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 100_000.0;
        let v_circ = (planet.gm / r).sqrt();

        // Ship going 80% of circular velocity
        let rel_pos = DVec3::new(r, 0.0, 0.0);
        let rel_vel = DVec3::new(0.0, v_circ * 0.8, 0.0);

        let (dv, burn_dir) = circularization_delta_v(rel_pos, rel_vel, planet.gm);
        assert!(dv > 10.0, "delta-v should be significant, got {dv}");
        // Burn should be prograde
        assert!(burn_dir.dot(rel_vel.normalize()) > 0.9,
            "burn should be prograde, dot = {}", burn_dir.dot(rel_vel.normalize()));
    }

    #[test]
    fn deorbit_delta_v_reasonable() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let orbit_alt = 100_000.0; // 100 km orbit
        let r = planet.radius_m + orbit_alt;
        let target_periapsis = 20_000.0; // lower periapsis to 20 km

        let dv = deorbit_delta_v(r, target_periapsis, planet.radius_m, planet.gm);
        let v_circ = (planet.gm / r).sqrt();

        assert!(dv > 0.0, "deorbit delta-v should be positive");
        assert!(dv < v_circ * 0.5, "deorbit delta-v should be a fraction of orbital velocity");
    }

    #[test]
    fn deorbit_delta_v_zero_when_already_low() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let r = planet.radius_m + 50_000.0;

        // Target periapsis above current orbit — no burn needed
        let dv = deorbit_delta_v(r, 60_000.0, planet.radius_m, planet.gm);
        assert!((dv - 0.0).abs() < 0.01, "should be zero when target is above current orbit");
    }

    #[test]
    fn autopilot_mode_roundtrip() {
        for v in 0..=4 {
            let mode = AutopilotMode::from_u8(v);
            assert_eq!(mode.to_u8(), v, "roundtrip failed for {v}");
        }
        // Unknown values default to DirectApproach
        assert_eq!(AutopilotMode::from_u8(255).to_u8(), 0);
    }

    #[test]
    fn ship_properties_moment_of_inertia() {
        let props = ShipPhysicalProperties::starter_ship();
        let (ix, iy, iz) = props.moment_of_inertia();
        // 10000 kg, 4×3×8 box
        // I_x = m*(h²+l²)/12 = 10000*(9+64)/12 = 60833
        assert!((ix - 60_833.0).abs() < 1.0, "I_x should be ~60833, got {ix}");
        // I_y = m*(w²+l²)/12 = 10000*(16+64)/12 = 66667
        assert!((iy - 66_667.0).abs() < 1.0, "I_y should be ~66667, got {iy}");
        // I_z = m*(w²+h²)/12 = 10000*(16+9)/12 = 20833
        assert!((iz - 20_833.0).abs() < 1.0, "I_z should be ~20833, got {iz}");
    }

    // -- Landing / takeoff guidance tests --

    #[test]
    fn landing_guidance_decelerates_when_fast() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = test_ship_props();
        let planet_pos = DVec3::ZERO;
        let altitude = 5000.0;
        let ship_pos = DVec3::new(planet.radius_m + altitude, 0.0, 0.0);
        // Falling fast toward planet
        let ship_vel = DVec3::new(-500.0, 0.0, 0.0);
        let accel = props.engine_acceleration(1); // Impulse

        let cmd = compute_landing_guidance(ship_pos, ship_vel, planet_pos, planet, &props, accel);
        assert!(cmd.thrust_magnitude > 0.0, "should be thrusting to decelerate");
        // Thrust should oppose velocity (retrograde)
        assert!(cmd.thrust_direction.dot(ship_vel.normalize()) < -0.5,
            "thrust should oppose velocity");
    }

    #[test]
    fn landing_guidance_gentle_final_approach() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = test_ship_props();
        let planet_pos = DVec3::ZERO;
        let altitude = 100.0; // below final approach
        let ship_pos = DVec3::new(planet.radius_m + altitude, 0.0, 0.0);
        // Descending slowly
        let ship_vel = DVec3::new(-5.0, 0.0, 0.0);
        let accel = props.engine_acceleration(1);

        let cmd = compute_landing_guidance(ship_pos, ship_vel, planet_pos, planet, &props, accel);
        assert_eq!(cmd.phase, FlightPhase::Landing);
    }

    #[test]
    fn takeoff_guidance_vertical_at_surface() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = test_ship_props();
        let planet_pos = DVec3::ZERO;
        let altitude = 10.0; // just above surface
        let ship_pos = DVec3::new(planet.radius_m + altitude, 0.0, 0.0);
        let ship_vel = DVec3::ZERO;
        let accel = props.engine_acceleration(1);

        let cmd = compute_takeoff_guidance(
            ship_pos, ship_vel, planet_pos, planet, &props, accel, 100_000.0,
        );
        assert_eq!(cmd.phase, FlightPhase::Liftoff);
        assert!(cmd.thrust_magnitude > 0.0, "should be thrusting upward");
        // Thrust should be mostly radial (upward)
        let radial = ship_pos.normalize();
        assert!(cmd.thrust_direction.dot(radial) > 0.9, "thrust should be mostly radial");
    }

    #[test]
    fn phase_transition_brake_to_soi_approach() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = ShipPhysicalProperties::starter_ship();
        let planet_pos = compute_planet_position(planet, 0.0);
        let planet_vel = DVec3::ZERO;
        let soi = compute_soi_radius(planet, &sys.star);
        // Ship inside SOI
        let ship_pos = planet_pos + DVec3::new(soi * 0.5, 0.0, 0.0);

        let phase = check_phase_transition(
            FlightPhase::Brake, AutopilotMode::OrbitInsertion,
            ship_pos, DVec3::ZERO, planet_pos, planet_vel, planet, &sys.star,
            soi, 100_000.0, &props,
        );
        assert_eq!(phase, FlightPhase::SoiApproach);
    }

    #[test]
    fn phase_transition_direct_approach_arrives() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = ShipPhysicalProperties::starter_ship();
        let planet_pos = compute_planet_position(planet, 0.0);
        let soi = compute_soi_radius(planet, &sys.star);
        let ship_pos = planet_pos + DVec3::new(soi * 0.5, 0.0, 0.0);

        let phase = check_phase_transition(
            FlightPhase::Brake, AutopilotMode::DirectApproach,
            ship_pos, DVec3::ZERO, planet_pos, DVec3::ZERO, planet, &sys.star,
            soi, 100_000.0, &props,
        );
        assert_eq!(phase, FlightPhase::Arrived, "DirectApproach should arrive, not orbit");
    }

    #[test]
    fn phase_transition_circularize_to_orbit() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let props = ShipPhysicalProperties::starter_ship();
        let alt = 100_000.0;
        let r = planet.radius_m + alt;
        let v_circ = (planet.gm / r).sqrt();
        let planet_pos = DVec3::ZERO;
        let ship_pos = DVec3::new(r, 0.0, 0.0);
        let ship_vel = DVec3::new(0.0, v_circ, 0.0); // circular
        let soi = compute_soi_radius(planet, &sys.star);

        let phase = check_phase_transition(
            FlightPhase::CircularizeBurn, AutopilotMode::OrbitInsertion,
            ship_pos, ship_vel, planet_pos, DVec3::ZERO, planet, &sys.star,
            soi, alt, &props,
        );
        assert_eq!(phase, FlightPhase::StableOrbit);
    }

    #[test]
    fn guidance_brakes_to_match_planet_velocity() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;
        let props = test_ship_props();

        let planet_pos = compute_planet_position(planet, 0.0);
        let soi = compute_soi_radius(planet, star);
        let planet_vel = compute_planet_velocity(planet, star.gm, 0.0)
            * sys.scale.time_scale;

        // Ship close to SOI boundary, moving at planet velocity + 1200 m/s approach.
        // Relative to planet: 1200 m/s toward target. Should trigger braking.
        let remaining = 20_000.0;
        let approach_dir = DVec3::new(-1.0, 0.0, 0.0);
        let ship_pos = planet_pos - approach_dir * (soi + remaining);
        let ship_vel = planet_vel + approach_dir * 1200.0;

        let intercept_pos = planet_pos - approach_dir * soi;
        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let cmd = compute_guidance(
            ship_pos, ship_vel, intercept_pos, planet_vel,
            0, &sys, &planet_positions, 0.0, &props, 1, false,
        );

        assert_eq!(cmd.phase, FlightPhase::Brake,
            "should brake when relative velocity demands it");
    }

    #[test]
    fn guidance_does_not_brake_when_comoving() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;
        let props = test_ship_props();

        let planet_pos = compute_planet_position(planet, 0.0);
        let soi = compute_soi_radius(planet, star);
        let planet_vel = compute_planet_velocity(planet, star.gm, 0.0)
            * sys.scale.time_scale;

        // Ship far from target, co-moving with planet (zero relative velocity).
        let ship_pos = planet_pos + DVec3::new(soi * 5.0, 0.0, 0.0);
        let ship_vel = planet_vel;

        let intercept_pos = planet_pos + DVec3::new(soi, 0.0, 0.0);
        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let cmd = compute_guidance(
            ship_pos, ship_vel, intercept_pos, planet_vel,
            0, &sys, &planet_positions, 0.0, &props, 1, false,
        );

        assert_eq!(cmd.phase, FlightPhase::Accelerate,
            "should accelerate, not brake, when co-moving with target");
    }

    #[test]
    fn solve_intercept_returns_arrival_velocity() {
        let sys = test_system();
        let planet = &sys.planets[0];
        let star = &sys.star;
        let props = test_ship_props();

        let planet_pos = compute_planet_position(planet, 0.0);
        let ship_pos = planet_pos + DVec3::new(sys.scale.spawn_offset * 10.0, 0.0, 0.0);

        let planet_positions: Vec<DVec3> = sys.planets.iter()
            .map(|p| compute_planet_position(p, 0.0)).collect();

        let sol = solve_intercept(
            ship_pos, DVec3::ZERO, planet, star,
            &sys, &planet_positions, 0.0, sys.scale.time_scale,
            &props, 2,
        ).unwrap();

        // Arrival velocity should be non-zero (planet is orbiting).
        assert!(sol.arrival_planet_vel.length() > 1000.0,
            "planet should have significant orbital velocity, got {:.0} m/s",
            sol.arrival_planet_vel.length());

        // Should be close to the planet's velocity at the predicted arrival time.
        let arrival_ct = sol.tof_real_seconds * sys.scale.time_scale;
        let expected_vel = compute_planet_velocity(planet, star.gm, arrival_ct)
            * sys.scale.time_scale;
        let diff = (sol.arrival_planet_vel - expected_vel).length();
        assert!(diff < expected_vel.length() * 0.01,
            "arrival velocity should match planet velocity at arrival time, diff={:.0}",
            diff);
    }
}
