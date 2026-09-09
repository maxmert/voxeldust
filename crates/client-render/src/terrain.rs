//! ★ THE TERRAIN ON SCREEN (the voxel foundation, slice 7; ruling V12) — the engine's side of the
//! chunk lane: it ASKS the library for chunks, HARVESTS the finished ones, and DRAWS them as children
//! of their realm's row. It never calls the generator, never picks a vertex, never decides a rung.
//!
//! ```text
//!   every frame
//!   ───────────
//!   1. every realm row with a SURFACE statement → the lane builds its body (once)
//!   2. the eye in that body's frame (from the delivered row and the delivered eye — display
//!      arithmetic, no pose derived) → the chunks wanted at THE rung (`D-TERRAIN-3`: one rung, a dev
//!      flag, deleted by slice 8) → request the new ones, release the old ones
//!   3. harvest a few finished chunks → a mesh each, a child of the row (floating origin)
//!   4. every chunk's transform = row placement ⊕ facing · chunk origin, reduced against the eye in
//!      f64 and narrowed once
//!   5. the sun: one directional light from the brightest luminous row in the window
//! ```
//!
//! The workers are THREADS behind the library's seam (`std::thread` + `crossbeam-channel`, no new
//! library): the render thread never generates.
//!
//! **Example.** The pilot's hull sits on the home planet. The planet's row carries its surface
//! statement; the lane holds its body; this system asks for the 25 columns under the hull at rung 0,
//! the eight workers build them, and three frames later the hills are on screen, lit by the star's
//! row, turning with the planet because they are children of its row.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::BTreeSet;
use std::sync::Mutex;

use vd_client::chunks::{
    ChunkJob, ChunkLane, ChunkReady, ChunkWorkers, InlineWorkers, MAX_RADIUS, body_frame_point,
    chunks_around, geometry_of,
};
use vd_client::realm_scene::{BoxShape, RealmBox};
use vd_core::geometry::Boundary;
use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_terrain::chunk::ChunkKey;

/// THE ONE-RUNG DEV FLAG (`D-TERRAIN-3`, deleted by slice 8): which rung this session draws. Absent:
/// no terrain is drawn, and every flight that ran before this slice is byte-identical.
pub const RUNG_ENV: &str = "VD_TERRAIN_RUNG";
/// How many chunk columns around the eye's column (each way) this session draws. Absent: two.
pub const RADIUS_ENV: &str = "VD_TERRAIN_RADIUS";
/// Flat shading instead of smooth (a debug switch; both are style, ruling S6-5).
pub const FLAT_ENV: &str = "VD_TERRAIN_FLAT";
/// Finished chunks harvested per frame — a bounded harvest, never a stall.
const HARVEST_PER_FRAME: usize = 4;
/// The default radius in chunk columns.
const DEFAULT_RADIUS: i32 = 2;
/// THE SHADOW'S REACH, in chunks past the drawn radius (M8-L, ruling V13 L23): the cascaded shadow
/// map covers the whole drawn patch and one chunk more, so no lit ground lies outside the shadow's
/// range and reads as a hole in the shade. Four cascades, the first ending at a 64th of the reach —
/// the engine's own default ratio. All of it is style: no vertex moves.
const SHADOW_REACH_CHUNKS: i32 = 1;
const SHADOW_CASCADES: usize = 4;
const SHADOW_FIRST_CASCADE_SHARE: f32 = 1.0 / 64.0;
/// THE NORMAL BIAS, DERIVED FROM THE SUN'S INCIDENCE (M8-L): a shadow map compares depths in texel
/// steps, and a face lit at an angle `i` from its normal spans `tan(i)` texels of depth per texel of
/// width, so a bias smaller than that shadows the face on itself ("shadow acne" — MEASURED on the
/// first re-lit pictures: a 15° star put the whole patch in its own shadow). The bias is the engine's
/// default plus `tan(i)` at the eye's own up, capped where the star is on the horizon.
const SHADOW_BIAS_TAN_CAP: f32 = 8.0;

/// THE SUN'S ILLUMINANCE, FROM THE CAMERA'S OWN EXPOSURE: the lux at which a white face turned
/// square to the sun renders white — `π / exposure`, the inverse of the engine's own pipeline
/// (`out = albedo · E / π · exposure`). Both cameras keep the engine's default exposure, which every
/// star sprite and every marker was measured under; a physical 100 000 lux at that exposure rendered
/// the ground thirty times white (MEASURED on the third ground picture: a cream field, no relief).
/// A picture's brightness is style; nothing here moves a vertex.
fn sun_lux(exposure: &bevy::camera::Exposure) -> f32 {
    std::f32::consts::PI / exposure.exposure()
}

/// What the flags said at startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainConfig {
    pub rung: Option<u8>,
    pub radius: i32,
    pub flat: bool,
}

impl TerrainConfig {
    /// Read the flags from the environment.
    #[must_use]
    pub fn from_env() -> TerrainConfig {
        let rung = std::env::var(RUNG_ENV)
            .ok()
            .and_then(|v| v.parse::<u8>().ok());
        let asked = std::env::var(RADIUS_ENV)
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(DEFAULT_RADIUS);
        let radius = asked.clamp(0, MAX_RADIUS);
        if radius != asked {
            tracing::warn!(
                asked,
                radius,
                "{RADIUS_ENV} is past the widest radius the lane serves — clamped"
            );
        }
        let flat = std::env::var(FLAT_ENV).is_ok_and(|v| v == "1");
        TerrainConfig { rung, radius, flat }
    }
}

/// THE THREADED WORKERS: a job channel fanned to `threads` threads, a done channel back. The
/// library's seam, filled in by the binary (HR5: the library's own tests use the inline workers).
pub struct ThreadedWorkers {
    jobs: Sender<ChunkJob>,
    done: Receiver<ChunkReady>,
    /// Jobs withdrawn before a worker took them: a worker checks before it starts, so a released
    /// chunk is never built (4 ms and 400 KB each, MEASURED at M7-2/M7-3).
    cancelled: Arc<Mutex<BTreeSet<(RealmId, ChunkKey)>>>,
}

impl ThreadedWorkers {
    /// Start `threads` workers.
    #[must_use]
    pub fn start(threads: usize) -> ThreadedWorkers {
        let (jobs, job_rx) = unbounded::<ChunkJob>();
        let (done_tx, done) = unbounded::<ChunkReady>();
        let cancelled: Arc<Mutex<BTreeSet<(RealmId, ChunkKey)>>> =
            Arc::new(Mutex::new(BTreeSet::new()));
        let mut n = 0;
        while n < threads.max(1) {
            let rx = job_rx.clone();
            let tx = done_tx.clone();
            let withdrawn = Arc::clone(&cancelled);
            std::thread::Builder::new()
                .name(format!("terrain-worker-{n}"))
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        let skip = withdrawn
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&(job.realm, job.key));
                        if skip {
                            continue;
                        }
                        if let Some(geometry) = geometry_of(&job.body, job.key) {
                            let _ = tx.send(ChunkReady {
                                realm: job.realm,
                                geometry,
                            });
                        }
                    }
                })
                .expect("a worker thread starts");
            n += 1;
        }
        ThreadedWorkers {
            jobs,
            done,
            cancelled,
        }
    }
}

impl ChunkWorkers for ThreadedWorkers {
    fn submit(&mut self, job: ChunkJob) {
        let _ = self.jobs.send(job);
    }

    fn cancel(&mut self, realm: RealmId, key: ChunkKey) {
        self.cancelled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((realm, key));
    }

    fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize) {
        let mut n = 0;
        while n < max {
            match self.done.try_recv() {
                Ok(ready) => out.push(ready),
                Err(_) => return,
            }
            n += 1;
        }
    }
}

/// One drawn chunk: which realm's row it rides.
#[derive(Component)]
pub struct TerrainChunk {
    pub realm: RealmId,
}

/// THE SUN: the one directional light, from the brightest luminous row.
#[derive(Component)]
pub struct TerrainSun;

/// The fill light opposite the sun — RETIRED (M8-L): a fill on the shadow side flattened every
/// picture; one light, one shadow. The marker stays so a stale entity of an older build is still
/// addressable by the query.
#[derive(Component)]
pub struct TerrainFill;

/// The terrain's state on the engine side.
#[derive(Resource)]
pub struct Terrain {
    pub config: TerrainConfig,
    pub lane: ChunkLane,
    entities: BTreeMap<(RealmId, ChunkKey), Entity>,
    material: Option<Handle<StandardMaterial>>,
    sun: Option<Entity>,
    /// Realms whose body has no rung this session asks for: warned once each.
    past_ladder: BTreeSet<RealmId>,
}

impl Terrain {
    /// The engine's terrain, for a client whose declared recipe tag is `declared`: over the threaded
    /// workers when a rung is named, over the inline ones (never asked) when none is — a flight
    /// without the flag starts no thread.
    #[must_use]
    pub fn new(config: TerrainConfig, declared: u64) -> Terrain {
        let workers: Box<dyn ChunkWorkers> = match config.rung {
            Some(_) => {
                let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
                Box::new(ThreadedWorkers::start(threads))
            }
            None => Box::new(InlineWorkers::default()),
        };
        Terrain {
            config,
            lane: ChunkLane::new(workers, declared),
            entities: BTreeMap::new(),
            material: None,
            sun: None,
            past_ladder: BTreeSet::new(),
        }
    }
}

/// A realm box's look as the boundary the lane wants.
fn look_of(rbox: &RealmBox) -> Boundary {
    match rbox.shape {
        BoxShape::Sphere { r } => Boundary::Shell { r },
        BoxShape::Box { half } => Boundary::Aabb { half },
    }
}

/// The row's facing as a rotation.
fn facing_of(rbox: &RealmBox) -> DQuat {
    DQuat::from_xyzw(
        f64::from(rbox.facing[0]),
        f64::from(rbox.facing[1]),
        f64::from(rbox.facing[2]),
        f64::from(rbox.facing[3]),
    )
    .normalize()
}

/// A Bevy mesh from a chunk's geometry: positions and normals relative to the chunk's origin, the
/// extractor's triangles as indices. `flat` duplicates the vertices and takes one normal per face.
fn mesh_of(geometry: &vd_client::chunks::ChunkGeometry, flat: bool) -> Mesh {
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.vertices.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals.clone())
    .with_inserted_indices(bevy::mesh::Indices::U32(
        geometry.triangles.iter().flatten().copied().collect(),
    ));
    if flat {
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();
    }
    mesh
}

/// The two terrain lights' transforms: the sun and its fill, re-aimed together every frame.
type LightQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Transform,
    (
        Or<(With<TerrainSun>, With<TerrainFill>)>,
        Without<TerrainChunk>,
    ),
>;

/// THE TERRAIN SYSTEM: runs after the realm boxes are placed, with the same eye.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources/queries
pub(crate) fn sync_terrain(
    net: Res<super::Net>,
    render_eye: Res<super::RenderEye>,
    mut terrain: ResMut<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    mut chunk_tf: Query<&mut Transform, (With<TerrainChunk>, Without<TerrainSun>)>,
    mut light_tf: LightQuery,
    boxes: Res<super::RealmBoxEntities>,
    mut visibility: Query<&mut Visibility>,
    key_light: Query<Entity, With<super::KeyLight>>,
    exposure: Query<&bevy::camera::Exposure, With<super::FollowCam>>,
) {
    let Some(rung) = terrain.config.rung else {
        return;
    };
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let scene = snap.scene_now(now_s);
    // 1. Every stated surface becomes a body (once); the eye-relative centre of every row.
    let mut centres: BTreeMap<RealmId, (DVec3, DQuat)> = BTreeMap::new();
    let mut brightest: Option<(f64, DVec3)> = None;
    for (realm, rbox) in scene.iter() {
        let draw_center = super::draw_center_of(rbox, &render_eye, &snap, now_s);
        centres.insert(realm, (draw_center, facing_of(rbox)));
        if let Some(surface) = rbox.surface {
            terrain.lane.state_surface(realm, &surface, &look_of(rbox));
        }
        if let Some((_, lux)) = rbox.luma
            && brightest.is_none_or(|(b, _)| lux > b)
        {
            brightest = Some((lux, draw_center));
        }
    }
    // 2. The chunks wanted around the eye at THE rung, per realm with a body.
    let mut wanted: BTreeMap<(RealmId, ChunkKey), ()> = BTreeMap::new();
    let radius = terrain.config.radius;
    let with_bodies: Vec<(RealmId, Arc<vd_terrain::BodyDefinition>)> = centres
        .keys()
        .filter_map(|realm| terrain.lane.body(*realm).map(|b| (*realm, Arc::clone(b))))
        .collect();
    for (realm, body) in &with_bodies {
        if rung >= body.ladder().rungs && terrain.past_ladder.insert(*realm) {
            tracing::warn!(
                ?realm,
                rung,
                rungs = body.ladder().rungs,
                "{RUNG_ENV} names a rung this body's ladder does not have — nothing is drawn for it"
            );
        }
        let (centre, facing) = centres[realm];
        // The eye is the render origin, so the eye relative to the row's centre is −draw_center.
        let eye_body = body_frame_point(
            [0.0, 0.0, 0.0],
            [centre.x, centre.y, centre.z],
            [facing.x, facing.y, facing.z, facing.w],
        );
        for key in chunks_around(body, eye_body, rung, radius) {
            wanted.insert((*realm, key), ());
        }
    }
    let held: Vec<(RealmId, ChunkKey)> = terrain.entities.keys().copied().collect();
    for (realm, key) in held {
        if !wanted.contains_key(&(realm, key)) {
            terrain.lane.release(realm, key);
            if let Some(entity) = terrain.entities.remove(&(realm, key)) {
                commands.entity(entity).despawn();
            }
        }
    }
    for (realm, key) in wanted.keys() {
        if !terrain.lane.holds(*realm, *key) {
            terrain.lane.request(*realm, *key);
        }
    }
    // 3. Harvest a few finished chunks.
    let flat = terrain.config.flat;
    let material = match &terrain.material {
        Some(m) => m.clone(),
        None => {
            let m = materials.add(StandardMaterial {
                base_color: Color::srgb(0.55, 0.50, 0.42),
                perceptual_roughness: 0.95,
                metallic: 0.0,
                cull_mode: Some(bevy::render::render_resource::Face::Back),
                ..default()
            });
            terrain.material = Some(m.clone());
            m
        }
    };
    for ready in terrain.lane.poll(HARVEST_PER_FRAME) {
        let realm = ready.realm;
        let key = ready.geometry.key;
        if !wanted.contains_key(&(realm, key)) {
            terrain.lane.release(realm, key);
            continue;
        }
        let mesh = meshes.add(mesh_of(&ready.geometry, flat));
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                TerrainChunk { realm },
                ChunkOrigin(ready.geometry.origin_m),
            ))
            .id();
        terrain.entities.insert((realm, key), entity);
    }
    // 4. Every chunk rides its row: transform = draw_center + facing · origin, in f64, narrowed once
    //    (`place_chunks`, below, which runs after the spawn commands apply). The count on screen goes
    //    to the diagnosis surface.
    let _ = &mut chunk_tf;
    net.terrain_drawn.store(
        terrain.entities.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    net.terrain_pending.store(
        terrain.lane.pending_count() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    // ONE DRAWING PER REALM: while a realm's terrain is on screen its proxy outline is hidden (from
    // inside its look shell the outline would paint the whole sky; from just outside it, the patch
    // is what there is — the black beyond its edge is `D-TERRAIN-3`, the one rung); the moment the
    // terrain is gone the outline is back. Slice 8 replaces the outline with the rungs beyond.
    let drawn_realms: BTreeSet<RealmId> = terrain.entities.keys().map(|(r, _)| *r).collect();
    for (realm, (entity, _)) in boxes.0.iter() {
        let has_terrain = drawn_realms.contains(realm);
        if let Ok(mut vis) = visibility.get_mut(*entity) {
            let want = if has_terrain {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
            if *vis != want {
                *vis = want;
            }
        }
    }
    // 5. The sun: from the brightest luminous row; when the window holds none (the star out of the
    //    planet's window, MEASURED on the first ground picture: a black ground under a black sky),
    //    a WORK LIGHT from straight above the observer — the radial of the first body's row — so
    //    the relief is seen. A work light is style; it moves no vertex.
    // `centre` is the body's centre relative to the eye, so `-centre` points UP at the observer's
    // feet; the work light STANDS there (the refuter found it standing below, lighting the
    // undersides of the hills).
    let overhead = with_bodies.first().map(|(realm, _)| {
        let (centre, _) = centres[realm];
        -centre.normalize_or_zero()
    });
    let sun_from = brightest.map(|(_, star)| star).or(overhead);
    if let Some(star) = sun_from {
        let dir = -star;
        if dir.length_squared() > 0.0 {
            let dir = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
            let transform = Transform::default().looking_to(dir, Vec3::Y);
            match terrain.sun {
                Some(sun) => {
                    if let Ok(mut t) = light_tf.get_mut(sun) {
                        *t = transform;
                    }
                }
                None => {
                    let lux = exposure
                        .iter()
                        .next()
                        .map_or_else(|| sun_lux(&bevy::camera::Exposure::default()), sun_lux);
                    // THE SHADOW (M8-L): the sun casts one, over the whole drawn patch. The reach
                    // is derived from what this session draws (the one rung and its radius, for
                    // one slice — slice 8 reads the ladder's residency instead).
                    let chunk_m = f64::from(vd_seed::ladder::cell_m(rung))
                        * vd_terrain::chunk::CHUNK_EDGE as f64;
                    let reach_m = (f64::from(radius + SHADOW_REACH_CHUNKS) * chunk_m) as f32;
                    let cascades = bevy::light::CascadeShadowConfigBuilder {
                        num_cascades: SHADOW_CASCADES,
                        first_cascade_far_bound: reach_m * SHADOW_FIRST_CASCADE_SHARE,
                        maximum_distance: reach_m,
                        ..default()
                    }
                    .build();
                    // The incidence at the eye: the angle between the light and the local up.
                    let up =
                        overhead.map_or(Vec3::Y, |u| Vec3::new(u.x as f32, u.y as f32, u.z as f32));
                    let cos_i = (-dir).dot(up).clamp(0.0, 1.0);
                    let tan_i =
                        ((1.0 - cos_i * cos_i).sqrt() / cos_i.max(1e-3)).min(SHADOW_BIAS_TAN_CAP);
                    let sun = commands
                        .spawn((
                            DirectionalLight {
                                illuminance: lux,
                                shadows_enabled: true,
                                shadow_normal_bias: DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS
                                    + tan_i,
                                ..default()
                            },
                            cascades,
                            transform,
                            TerrainSun,
                        ))
                        .id();
                    terrain.sun = Some(sun);
                    // ONE SUN: the stub world's fixed key light retires the moment the sun is born
                    // (it lit the ground from a direction no star stands in, at three times white).
                    for key in &key_light {
                        commands.entity(key).despawn();
                    }
                }
            }
        }
    }
}

/// A chunk's origin in its realm's frame, kept on the entity so its transform can be re-derived
/// every frame from the row without the lane.
#[derive(Component)]
pub struct ChunkOrigin(pub [f64; 3]);

/// Place every drawn chunk against the eye: `draw_center(row) + facing · origin`, in f64, narrowed
/// once. Runs after `sync_terrain` so a chunk spawned this frame is placed this frame.
pub(crate) fn place_chunks(
    net: Res<super::Net>,
    render_eye: Res<super::RenderEye>,
    terrain: Res<Terrain>,
    mut chunks: Query<(&TerrainChunk, &ChunkOrigin, &mut Transform)>,
    mut frame: Local<u32>,
) {
    // Nothing drawn: nothing to place, and no scene to clone (a flight in open space with the flag
    // set used to walk every row of the window for no pixel — SL9).
    if terrain.config.rung.is_none() || chunks.is_empty() {
        return;
    }
    let now_s = net.started_at.elapsed().as_secs_f64();
    let snap = net.snapshot.load();
    let scene = snap.scene_now(now_s);
    let mut rows: BTreeMap<RealmId, (DVec3, DQuat)> = BTreeMap::new();
    for (realm, rbox) in scene.iter() {
        rows.insert(
            realm,
            (
                super::draw_center_of(rbox, &render_eye, &snap, now_s),
                facing_of(rbox),
            ),
        );
    }
    let mut nearest = f64::INFINITY;
    let mut farthest = 0.0_f64;
    let mut placed = 0u32;
    let mut first: Option<(RealmId, [f64; 3], DVec3)> = None;
    for (chunk, origin, mut transform) in &mut chunks {
        if let Some((centre, facing)) = rows.get(&chunk.realm) {
            let o = DVec3::new(origin.0[0], origin.0[1], origin.0[2]);
            let p = *centre + *facing * o;
            transform.translation = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            transform.rotation = Quat::from_xyzw(
                facing.x as f32,
                facing.y as f32,
                facing.z as f32,
                facing.w as f32,
            );
            nearest = nearest.min(p.length());
            farthest = farthest.max(p.length());
            placed += 1;
            if first.is_none() {
                first = Some((chunk.realm, origin.0, p));
            }
        }
    }
    *frame += 1;
    if frame.is_multiple_of(DIAG_EVERY) {
        for (realm, (centre, facing)) in &rows {
            tracing::debug!(
                ?realm,
                centre = ?centre,
                facing = ?facing,
                dist = centre.length(),
                "terrain diag: row"
            );
        }
        tracing::debug!(
            placed,
            nearest,
            farthest,
            first = ?first,
            eye = ?render_eye.eye,
            eye_tier = ?render_eye.eye_lattice.map(|(_, t)| t),
            view = ?render_eye.view,
            "terrain diag: chunks"
        );
    }
}

/// Frames between two diagnostic lines (`RUST_LOG=vd_client_render=debug` shows them: every row's
/// drawn centre and facing, the chunks' nearest and farthest distance, the eye and its unit — the
/// lines that found the 230 km column, kept for the next such hunt).
pub(crate) const DIAG_EVERY: u32 = 60;
