//! Starfield — the night-side "black sky looks alive" effect.
//!
//! 2 000 emissive points are scattered on a 2-km sphere at world-space
//! infinity. They stay at a fixed world position; the atmosphere's
//! transmittance LUT handles day-side occlusion automatically (stars fade
//! during the day + twilight, bright at night — the "stars you can see at
//! dusk" graduated reveal).
//!
//! Single shared `StandardMaterial` with a bright emissive colour means Bevy
//! auto-batches all star entities into a handful of draw calls, and Bloom
//! picks up the emissives for the twinkling look.
//!
//! Deterministic layout: the same seed produces the same sky across runs, so
//! constellations are stable and the future Milky Way band sits at the same
//! galactic latitude every session.

use std::f32::consts::TAU;

use bevy::prelude::*;

pub struct StarfieldPlugin;

impl Plugin for StarfieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_starfield);
    }
}

fn spawn_starfield(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Tiny icosphere — small enough to read as a point, but a real mesh so
    // Bevy can batch + frustum-cull. 0.8 m radius at 2 km distance → about a
    // quarter of a degree subtended, similar to a bright first-magnitude star.
    let star_mesh = meshes.add(Sphere::new(0.8).mesh().ico(1).unwrap());

    // Warm-white star material. High emissive drives bloom; base colour so
    // low that atmospheric transmittance completely dominates during the day.
    let warm_star = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.04, 0.05),
        emissive: LinearRgba::rgb(45.0, 42.0, 36.0),
        ..default()
    });
    // Cool-blue variant for stellar-class variety.
    let cool_star = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.05, 0.06),
        emissive: LinearRgba::rgb(28.0, 38.0, 55.0),
        ..default()
    });
    // Red-giant variant — a handful of these anchor the sky.
    let red_star = materials.add(StandardMaterial {
        base_color: Color::srgb(0.06, 0.03, 0.02),
        emissive: LinearRgba::rgb(60.0, 18.0, 10.0),
        ..default()
    });

    // 2 000 stars, deterministic from a fixed seed.
    let mut rng = XorShift32 { state: 0xC0FFEE_u32 };
    for i in 0..2_000 {
        // Uniform point on a unit sphere (Marsaglia's method).
        let pos = loop {
            let x = rng.next_f32() * 2.0 - 1.0;
            let y = rng.next_f32() * 2.0 - 1.0;
            let s = x * x + y * y;
            if s < 1.0 {
                let factor = 2.0 * (1.0 - s).sqrt();
                break Vec3::new(x * factor, 1.0 - 2.0 * s, y * factor);
            }
        };
        let world = pos.normalize() * 2_000.0;

        // Lift ~30 % of stars slightly off the default warm-white palette.
        let material = match i % 13 {
            0..=8 => warm_star.clone(),
            9..=11 => cool_star.clone(),
            _ => red_star.clone(),
        };

        // Vary brightness with a log-magnitude-like curve so a few stars
        // dominate and the rest are faint. Scale applied via Transform.
        let brightness = 0.5 + rng.next_f32().powi(4) * 1.5;

        commands.spawn((
            Mesh3d(star_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(world).with_scale(Vec3::splat(brightness)),
            // Opt out of shadow casting — stars are infinity lights, not shadow casters.
            NotShadowCaster,
        ));
    }
}

// Simple xorshift32 so the starfield doesn't need an RNG crate.
struct XorShift32 {
    state: u32,
}
impl XorShift32 {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

// Brings `NotShadowCaster` into scope. Kept local so the bevy::prelude use
// in main.rs doesn't need extra imports.
use bevy::light::NotShadowCaster;

// `TAU` is imported for future Milky-Way-band variant of this plugin
// (clustering stars along a great circle). Silences dead-code lint until
// that feature lands.
#[allow(dead_code)]
const _PLANE_CIRCLE: f32 = TAU;
