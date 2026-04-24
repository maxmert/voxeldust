//! Precipitation + ambient weather for the Bevy scaffold.
//!
//! Implements a lightweight GPU-instanced rain particle system on top of
//! Bevy's mesh renderer. A single thin cuboid mesh is spawned N times with
//! different `Transform`s around the camera; a CPU system advects each
//! particle downward along a wind vector and wraps it around a cylinder
//! centred on the player. Bevy's `MeshInstanceManager` (implicit in the
//! default render graph) batches all N draws into a single indirect call
//! because they share mesh + material.
//!
//! Scope gate: this is a scaffold-quality precipitation system. Phase 8 of
//! the migration plan promotes it to `bevy_hanabi` once the ecosystem crate
//! has 0.18-compatible particle spawning, and hooks density into the weather
//! map once Phase 3 streams it from the server. For now, a `Resource` toggles
//! rain on/off + intensity, driven by the HUD.
//!
//! Streaks stay bright because of the camera's `AutoExposure`/`Bloom` — they
//! render white-emissive and get tonemapped naturally.

use bevy::{
    light::NotShadowCaster,
    prelude::*,
};

const RAIN_PARTICLE_COUNT: usize = 4_000;
const RAIN_RADIUS: f32 = 45.0;
const RAIN_HEIGHT: f32 = 40.0;
const FALL_SPEED: f32 = 28.0;

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherState>()
            .add_systems(Startup, spawn_rain)
            .add_systems(Update, (update_rain, sway_grass_marker));
    }
}

/// Shared weather state. When `precip > 0`, rain particles are visible and
/// advecting. In a future Phase-3 build, this becomes an interpolated sample
/// of the server weather map at the player's planet-relative direction.
#[derive(Resource)]
pub struct WeatherState {
    pub precip: f32,
    pub wind: Vec3,
}

impl Default for WeatherState {
    fn default() -> Self {
        // Light rain by default — enough to see the streaks + wet atmosphere
        // without drowning the other visual features in the scaffold.
        Self {
            precip: 0.35,
            wind: Vec3::new(3.0, 0.0, 2.0),
        }
    }
}

/// Marker for rain particle entities.
#[derive(Component)]
struct RainParticle;

fn spawn_rain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Streak geometry: tall thin cuboid, y-axis aligned. Looks like a
    // velocity-blurred rain streak when rendered bright.
    let streak_mesh = meshes.add(Cuboid::new(0.035, 0.65, 0.035));

    // White-blue emissive, slightly transparent via alpha channel so streaks
    // stack visually without painting the sky white. `AutoExposure` +
    // `Bloom::NATURAL` give the haloed-thread look.
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.7, 0.82, 0.95, 0.7),
        emissive: LinearRgba::rgb(3.0, 3.8, 4.5),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // Deterministic seed → identical rain layout across runs for golden
    // screenshots and CI-style regression checks.
    let mut rng = XorShift32 { state: 0xDEADBEEF_u32 };
    for _ in 0..RAIN_PARTICLE_COUNT {
        let r = RAIN_RADIUS * rng.next_f32().sqrt();
        let theta = rng.next_f32() * std::f32::consts::TAU;
        let y = (rng.next_f32() * 2.0 - 1.0) * RAIN_HEIGHT * 0.5;
        let x = r * theta.cos();
        let z = r * theta.sin();
        commands.spawn((
            Mesh3d(streak_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_xyz(x, y + 30.0, z),
            NotShadowCaster,
            RainParticle,
        ));
    }
}

/// Advance each rain streak by wind + gravity, wrap around the cylindrical
/// bounds anchored on the camera position. Density modulated by the weather
/// state's precipitation field — at `precip == 0` all particles are lifted
/// above the camera and hidden by frustum culling.
fn update_rain(
    time: Res<Time>,
    weather: Res<WeatherState>,
    camera: Query<&Transform, (With<Camera3d>, Without<RainParticle>)>,
    mut particles: Query<&mut Transform, With<RainParticle>>,
) {
    let Ok(cam) = camera.single() else { return };
    let cam_pos = cam.translation;
    let dt = time.delta_secs();
    let precip = weather.precip.clamp(0.0, 1.0);
    let density_cutoff = 1.0 - precip;

    let wind_x = weather.wind.x;
    let wind_z = weather.wind.z;
    let fall = FALL_SPEED;

    let mut i: u32 = 0;
    for mut tf in &mut particles {
        i = i.wrapping_add(1);

        // Deterministic per-particle cutoff: hash the index into [0,1) and
        // compare against `density_cutoff`. Above-cutoff particles are
        // teleported way above the camera so they're out of view.
        let cutoff_hash = ((i.wrapping_mul(2654435761)) >> 8) as f32 / 16_777_216.0;
        if cutoff_hash < density_cutoff {
            tf.translation.y = cam_pos.y + 500.0; // hidden
            continue;
        }

        // Advect
        tf.translation.y -= fall * dt;
        tf.translation.x += wind_x * dt;
        tf.translation.z += wind_z * dt;

        // Wrap around a vertical cylinder centred on the camera.
        let rel_x = tf.translation.x - cam_pos.x;
        let rel_z = tf.translation.z - cam_pos.z;
        let dist = (rel_x * rel_x + rel_z * rel_z).sqrt();
        if dist > RAIN_RADIUS {
            // Re-spawn on the opposite side — simulates particles flowing in
            // with the wind.
            let angle = rel_z.atan2(rel_x) + std::f32::consts::PI;
            let new_r = RAIN_RADIUS * 0.95;
            tf.translation.x = cam_pos.x + new_r * angle.cos();
            tf.translation.z = cam_pos.z + new_r * angle.sin();
        }

        // Wrap Y: when a streak reaches below the ground, lift it above.
        let rel_y = tf.translation.y - cam_pos.y;
        if rel_y < -RAIN_HEIGHT * 0.5 {
            // Use hash jitter for y offset to avoid all streaks respawning
            // at the same altitude (which would cause a visible "wave").
            let jitter = (((i.wrapping_mul(1597334677)) >> 8) as f32 / 16_777_216.0) * 4.0;
            tf.translation.y = cam_pos.y + RAIN_HEIGHT * 0.5 + jitter;
        }
    }
}

/// Placeholder sway system — wired up here so future animated grass /
/// foliage can latch onto a shared wind resource. Currently no-op because the
/// scaffold doesn't have grass meshes yet.
fn sway_grass_marker(_: Res<WeatherState>) {}

// Simple xorshift32 so the rain layout doesn't need an RNG crate.
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
