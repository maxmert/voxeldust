//! ★ THE CHUNK LANE (the voxel foundation, slice 7; ruling V12 S7-3..S7-6) — what the client LIBRARY
//! hands an engine for a seed-shaped realm, and what it keeps.
//!
//! **What it keeps:** the recipe (`vd-terrain`, the same crate the server links — SL10 clause 2),
//! the body per realm (built from the realm's SURFACE statement: the seed in its frame and its look
//! shell's radius), the extractor, the worker seam, and every number that decides what is drawn.
//!
//! **What the engine gets:** three verbs and one data type.
//!
//! ```text
//!   request(realm, key, rung)      the engine asks; the lane queues the work on its workers
//!   poll(max) -> Vec<ChunkReady>   the engine harvests finished chunks, at most `max` per call
//!   release(realm, key)            the engine drops a chunk; the lane frees it
//!
//!   ChunkGeometry { key, rung, origin_m, vertices (f32, relative to origin_m), normals, triangles }
//! ```
//!
//! **The floating origin.** The extractor's positions are body-frame metres in `f64` (3 351 000 m
//! from the home planet's centre). A 32-bit float has about seven digits, so those numbers would lose
//! their millimetres. The lane subtracts the chunk's own ORIGIN (the centre of its middle cell) in
//! `f64` and hands the engine the differences, which are at most a chunk wide. The engine places the
//! chunk at `row placement ⊕ origin`, reduced against the eye before the last narrowing — the pattern
//! the realm boxes use.
//!
//! **Normals are style** (ruling S6-5): derived here from the triangles, area-weighted per vertex and
//! normalised — a smooth look. They never move a vertex.
//!
//! **One rung per realm — FOR ONE SLICE** (`D-TERRAIN-3`, deleted by slice 8): a request for a second
//! rung of a realm that already holds another is REFUSED and counted, never served, so a hard edge
//! between two rungs cannot reach a picture before its crossfade exists.
//!
//! **The worker seam.** [`ChunkWorkers`] is one trait with two methods: submit a job, drain finished
//! jobs. The Tier-A tests drive [`InlineWorkers`], which runs a job on the calling thread, so every
//! branch of the lane is covered without a thread; the shipped binary installs a threaded
//! implementation behind the same trait.
//!
//! **Example.** A pilot's client receives the home planet's row with its surface statement. The lane
//! builds the body from the seed, the engine asks for the 25 columns around the hull at rung 0, the
//! workers generate and extract them, and the engine harvests them a few per frame and draws them as
//! children of the planet's row. The planet's shard drew nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use vd_core::geometry::Boundary;
use vd_core::look::SurfaceStmt;
use vd_core::pose::{FrameRef, RealmId};
use vd_terrain::BodyDefinition;
use vd_terrain::Gf;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::extract::extract;
use vd_terrain::lattice::sample_box;
use vd_terrain::position::vertex_position_m;

/// A finished chunk: integer vertices turned into floats around a floating origin.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkGeometry {
    pub key: ChunkKey,
    /// The chunk's own origin in the realm's frame, in metres: the centre of its middle cell.
    pub origin_m: [f64; 3],
    /// Vertices in metres, relative to `origin_m`.
    pub vertices: Vec<[f32; 3]>,
    /// Per-vertex unit normals, derived from the triangles (style, never shape).
    pub normals: Vec<[f32; 3]>,
    /// The extractor's triangles, unchanged.
    pub triangles: Vec<[u32; 3]>,
}

/// A job for the workers: one chunk of one body.
#[derive(Clone, Debug)]
pub struct ChunkJob {
    pub realm: RealmId,
    pub body: Arc<BodyDefinition>,
    pub key: ChunkKey,
}

/// A finished job, as the engine harvests it.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkReady {
    pub realm: RealmId,
    pub geometry: ChunkGeometry,
}

/// THE WORKER SEAM: where a job runs. The lane never cares. `Send + Sync` so an engine may keep the
/// lane in a resource its systems share.
pub trait ChunkWorkers: Send + Sync {
    /// Take a job. It may finish now (inline) or later (a thread).
    fn submit(&mut self, job: ChunkJob);
    /// Move up to `max` finished jobs into `out`.
    fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize);
    /// Withdraw a job the lane no longer wants: one not started is never run, one finished is not
    /// handed out. A job already running finishes and is dropped at the lane's `poll`.
    fn cancel(&mut self, realm: RealmId, key: ChunkKey);
}

/// The inline workers: a job runs on the calling thread at `submit`, and waits in a queue for
/// `drain`. The Tier-A implementation, and a correct one for a single-threaded host.
#[derive(Default)]
pub struct InlineWorkers {
    done: std::collections::VecDeque<ChunkReady>,
}

impl ChunkWorkers for InlineWorkers {
    fn submit(&mut self, job: ChunkJob) {
        // Branchless (HR5): the lane refuses a key outside the ladder before it submits, so the
        // `None` arm of the seam's own refusal is never reached through the lane.
        self.done
            .extend(geometry_of(&job.body, job.key).map(|geometry| ChunkReady {
                realm: job.realm,
                geometry,
            }));
    }

    fn cancel(&mut self, realm: RealmId, key: ChunkKey) {
        self.done
            .retain(|r| (r.realm != realm) | (r.geometry.key != key));
    }

    fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize) {
        let mut n = 0;
        while n < max {
            match self.done.pop_front() {
                Some(ready) => out.push(ready),
                None => return,
            }
            n += 1;
        }
    }
}

/// Run the recipe and the extractor for one chunk and turn the result into engine floats; `None`
/// for a key outside the body (the lane never submits one, so this is the seam's own refusal).
#[must_use]
pub fn geometry_of(body: &BodyDefinition, key: ChunkKey) -> Option<ChunkGeometry> {
    let samples = sample_box(body, key)?;
    let mesh = extract(&samples);
    let half = (CHUNK_EDGE / 2) as i16 * vd_terrain::VERTEX_QUANTUM as i16;
    let origin = vertex_position_m(body, &samples, [half, half, half]);
    let origin_m = [origin[0].to_f64(), origin[1].to_f64(), origin[2].to_f64()];
    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            let p = vertex_position_m(body, &samples, *v);
            [
                (p[0] - origin[0]).to_f64() as f32,
                (p[1] - origin[1]).to_f64() as f32,
                (p[2] - origin[2]).to_f64() as f32,
            ]
        })
        .collect();
    let normals = smooth_normals(&vertices, &mesh.triangles);
    Some(ChunkGeometry {
        key,
        origin_m,
        vertices,
        normals,
        triangles: mesh.triangles,
    })
}

/// Per-vertex normals: the sum of the adjoining triangles' area-weighted normals, normalised. A
/// vertex no triangle uses (never emitted by the extractor) would keep a zero normal.
#[must_use]
pub fn smooth_normals(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut sums = vec![[0.0f32; 3]; vertices.len()];
    for t in triangles {
        let a = vertices[t[0] as usize];
        let b = vertices[t[1] as usize];
        let c = vertices[t[2] as usize];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        for i in t {
            let s = &mut sums[*i as usize];
            s[0] += n[0];
            s[1] += n[1];
            s[2] += n[2];
        }
    }
    sums.iter()
        .map(|s| {
            let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            if len > 0.0 {
                [s[0] / len, s[1] / len, s[2] / len]
            } else {
                *s
            }
        })
        .collect()
}

/// The lane's counters: every refusal is counted, never silent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkCounters {
    /// A surface statement whose recipe tag is not this client's: never drawn.
    pub foreign_generator: u64,
    /// A surface statement whose frame carries no seed, or whose radius the ladder refuses.
    pub no_body: u64,
    /// A request for a second rung of a realm that already holds one (`D-TERRAIN-3`).
    pub second_rung: u64,
    /// A request for a key outside the body's ladder.
    pub outside: u64,
    /// Jobs submitted to the workers.
    pub submitted: u64,
    /// Chunks harvested by the engine.
    pub harvested: u64,
}

/// The lane: the bodies it knows, the chunks it holds, the workers it drives.
pub struct ChunkLane {
    workers: Box<dyn ChunkWorkers>,
    /// This client's declared recipe tag: a realm's surface must state the same one.
    declared: u64,
    bodies: BTreeMap<RealmId, Arc<BodyDefinition>>,
    /// Realms whose surface statement was refused: counted ONCE, never re-read each frame.
    refused: BTreeSet<RealmId>,
    /// The one rung each realm holds (`D-TERRAIN-3`).
    rung_of: BTreeMap<RealmId, u8>,
    resident: BTreeSet<(RealmId, ChunkKey)>,
    pending: BTreeSet<(RealmId, ChunkKey)>,
    /// Resident plus pending chunks per realm, so a release is a lookup, not a scan (SL9).
    held: BTreeMap<RealmId, usize>,
    counters: ChunkCounters,
}

impl ChunkLane {
    /// A lane over `workers`, for a client whose declared recipe tag is `declared`.
    #[must_use]
    pub fn new(workers: Box<dyn ChunkWorkers>, declared: u64) -> ChunkLane {
        ChunkLane {
            workers,
            declared,
            bodies: BTreeMap::new(),
            refused: BTreeSet::new(),
            rung_of: BTreeMap::new(),
            resident: BTreeSet::new(),
            pending: BTreeSet::new(),
            held: BTreeMap::new(),
            counters: ChunkCounters::default(),
        }
    }

    /// The counters.
    #[must_use]
    pub fn counters(&self) -> ChunkCounters {
        self.counters
    }

    /// How many chunks are still building: the instrument a picture gate waits on for "the whole
    /// wanted set landed", beside `terrain_chunks_drawn` for "the first one did".
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// State a realm's surface: its statement and its look. The body is built from the seed in the
    /// statement's frame and the look shell's radius. A statement with a foreign recipe tag, a
    /// frame without a seed, or a radius the ladder refuses is counted and the realm gets no body.
    /// Stating the same surface twice keeps the body; a refused realm is counted once and then
    /// ignored until it is forgotten (a refusal used to be re-counted and the body re-derived on
    /// every frame — the refuter's finding).
    pub fn state_surface(&mut self, realm: RealmId, surface: &SurfaceStmt, look: &Boundary) {
        if self.bodies.contains_key(&realm) | self.refused.contains(&realm) {
            return;
        }
        if surface.generator != self.declared {
            self.counters.foreign_generator += 1;
            self.refused.insert(realm);
            return;
        }
        let body = match (surface.frame, look) {
            (FrameRef::PlanetCentered { planet_seed }, Boundary::Shell { r }) => {
                BodyDefinition::from_seed(planet_seed, *r)
            }
            _ => None,
        };
        match body {
            Some(body) => {
                self.bodies.insert(realm, Arc::new(body));
            }
            None => {
                self.counters.no_body += 1;
                self.refused.insert(realm);
            }
        }
    }

    /// The body the lane holds for a realm.
    #[must_use]
    pub fn body(&self, realm: RealmId) -> Option<&Arc<BodyDefinition>> {
        self.bodies.get(&realm)
    }

    /// Forget a realm: its body and every chunk of it.
    pub fn forget(&mut self, realm: RealmId) {
        self.bodies.remove(&realm);
        self.refused.remove(&realm);
        self.rung_of.remove(&realm);
        self.held.remove(&realm);
        self.resident.retain(|(r, _)| *r != realm);
        let pending: Vec<ChunkKey> = self
            .pending
            .iter()
            .filter(|(r, _)| *r == realm)
            .map(|(_, k)| *k)
            .collect();
        for key in pending {
            self.pending.remove(&(realm, key));
            self.workers.cancel(realm, key);
        }
    }

    /// Whether the lane holds or is building a chunk.
    #[must_use]
    pub fn holds(&self, realm: RealmId, key: ChunkKey) -> bool {
        self.resident.contains(&(realm, key)) | self.pending.contains(&(realm, key))
    }

    /// Ask for a chunk. A realm without a body, a key outside its ladder, or a second rung of a
    /// realm that holds one (`D-TERRAIN-3`) is refused and counted; a chunk already held or
    /// building is not queued twice.
    pub fn request(&mut self, realm: RealmId, key: ChunkKey) {
        let Some(body) = self.bodies.get(&realm) else {
            self.counters.no_body += 1;
            return;
        };
        if !vd_terrain::lattice::in_ladder(body, key) {
            self.counters.outside += 1;
            return;
        }
        match self.rung_of.get(&realm) {
            Some(rung) if *rung != key.rung => {
                self.counters.second_rung += 1;
                return;
            }
            _ => {
                self.rung_of.insert(realm, key.rung);
            }
        }
        if self.holds(realm, key) {
            return;
        }
        self.pending.insert((realm, key));
        *self.held.entry(realm).or_insert(0) += 1;
        self.counters.submitted += 1;
        self.workers.submit(ChunkJob {
            realm,
            body: Arc::clone(body),
            key,
        });
    }

    /// Harvest up to `max` finished chunks. A chunk released while it was building is dropped
    /// here, never handed out.
    pub fn poll(&mut self, max: usize) -> Vec<ChunkReady> {
        let mut out = Vec::new();
        self.workers.drain(&mut out, max);
        out.retain(|ready| self.pending.remove(&(ready.realm, ready.geometry.key)));
        for ready in &out {
            self.resident.insert((ready.realm, ready.geometry.key));
        }
        self.counters.harvested += out.len() as u64;
        out
    }

    /// Drop a chunk: resident or still building (withdrawn from the workers). The last chunk of a
    /// realm releases its rung. A lookup per realm, never a scan of every chunk held (SL9).
    pub fn release(&mut self, realm: RealmId, key: ChunkKey) {
        let was_resident = self.resident.remove(&(realm, key));
        let was_pending = self.pending.remove(&(realm, key));
        if was_pending {
            self.workers.cancel(realm, key);
        }
        if was_resident | was_pending {
            let left = self.held.entry(realm).or_insert(1);
            *left -= 1;
            if *left == 0 {
                self.held.remove(&realm);
                self.rung_of.remove(&realm);
            }
        }
    }

    /// The chunks the lane holds for a realm.
    #[must_use]
    pub fn resident(&self, realm: RealmId) -> Vec<ChunkKey> {
        self.resident
            .iter()
            .filter(|(r, _)| *r == realm)
            .map(|(_, k)| *k)
            .collect()
    }
}

/// THE CHUNKS AROUND A POINT at one rung — the surface chunk of every column within `radius`
/// chunks of the column under `point_m` (a position in the body's frame, from a delivered row and
/// the delivered eye), plus the chunk below and above it. FOR ONE SLICE (`D-TERRAIN-3`): slice 8's
/// tier rule and residency band replace it. Columns past the face's edge are left out here; the
/// partial chunk at the edge is included, because it is a chunk of the ladder.
///
/// A point farther from the centre than [`FAR_EYE_RADII`] radii gets nothing: from there the whole
/// hemisphere is in view and one column under the eye draws nothing a player can see (MEASURED on
/// the first ground picture: the two other planets of the home system, 1.5 × 10¹¹ m away, each
/// cost a column of chunks placed where nobody looks).
///
/// The face coordinates a direction gives are the TANGENTS `W(a)`; the cell index wants the face
/// parameter `a`, so the inverse bend sits between them (MEASURED on the ground picture: without it
/// the column asked for lay 230 km from the eye; `vd_core::grid::shell` does the same).
/// How many body radii from the centre an eye may stand and still get the column under it.
pub const FAR_EYE_RADII: f64 = 2.0;
/// The widest radius in chunk columns a caller may ask for: (2·64 + 1)² columns is the most any
/// one-rung picture needs, and a radius past it (an operator's typo) would freeze the render
/// thread for minutes per frame. Clamped, and the clamp is reported by the caller.
pub const MAX_RADIUS: i32 = 64;

#[must_use]
pub fn chunks_around(
    body: &BodyDefinition,
    point_m: [f64; 3],
    rung: u8,
    radius: i32,
) -> Vec<ChunkKey> {
    use vd_seed::bend::{face_coords, face_of, unbend};
    use vd_seed::ladder::index_of;
    let radius = radius.clamp(0, MAX_RADIUS);
    let len = (point_m[0] * point_m[0] + point_m[1] * point_m[1] + point_m[2] * point_m[2]).sqrt();
    if len.is_nan()
        || len <= 0.0
        || len > body.ladder().radius_m() * FAR_EYE_RADII
        || rung >= body.ladder().rungs
    {
        return Vec::new();
    }
    let d = [point_m[0] / len, point_m[1] / len, point_m[2] / len];
    let face = face_of(d);
    let (t, s) = face_coords(face, d);
    let (a, b) = (unbend(t), unbend(s));
    let n_l = body.ladder().cells_per_edge(rung);
    let edge = CHUNK_EDGE as i32;
    let last = (n_l as i32 - 1) / edge;
    let (cx, cy) = (index_of(a, n_l) / edge, index_of(b, n_l) / edge);
    let top = (body.ladder().cells_in_band(rung) as i32 - 1) / edge;
    let mut keys = Vec::new();
    let mut y = cy - radius;
    while y <= cy + radius {
        let mut x = cx - radius;
        while x <= cx + radius {
            if (x >= 0) & (x <= last) & (y >= 0) & (y <= last) {
                let (lo, hi) = vd_terrain::digest::surface_chunk_span(body, face, rung, x, y);
                let mut z = lo;
                while z <= hi.min(top) {
                    keys.push(ChunkKey {
                        face,
                        rung,
                        x,
                        y,
                        z,
                    });
                    z += 1;
                }
            }
            x += 1;
        }
        y += 1;
    }
    keys
}

/// A body-frame position from a delivered row's placement and facing and the delivered eye, all
/// in the origin frame: `Fᵀ · (eye − centre)`. Display arithmetic on delivered data — no pose and no
/// velocity is derived (SL10 clause 7).
#[must_use]
pub fn body_frame_point(eye_m: [f64; 3], centre_m: [f64; 3], facing_xyzw: [f64; 4]) -> [f64; 3] {
    let q = vd_core::glam::DQuat::from_xyzw(
        facing_xyzw[0],
        facing_xyzw[1],
        facing_xyzw[2],
        facing_xyzw[3],
    )
    .normalize();
    let p = q.inverse()
        * vd_core::glam::DVec3::new(
            eye_m[0] - centre_m[0],
            eye_m[1] - centre_m[1],
            eye_m[2] - centre_m[2],
        );
    [p.x, p.y, p.z]
}

/// THE EYE'S GROUND TRUTH (8p, ruling V14 D8-7): the recipe's surface under a body-frame point, the
/// point's height over it, and the biome there. The stamp on every judged picture states these,
/// and the picture gate recomputes them from the state file — two readings of one recipe that must
/// agree (M8-4). The surface is read at rung 0, the true shape; a drawn rung stands within its own
/// bound of it. `None` at the body's centre (no direction) or for a non-finite point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeSurface {
    /// The recipe's surface radius along the point's direction, in metres.
    pub surface_m: f64,
    /// The point's height over that surface, in metres (negative underground).
    pub altitude_m: f64,
    /// The biome of the column under the point.
    pub biome: vd_terrain::strata::Biome,
}

/// A finite length greater than zero — one expression, so a NaN reads as "no", never as a branch.
fn positive(x: f64) -> bool {
    x.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater)
}

#[must_use]
pub fn eye_surface(body: &BodyDefinition, point_m: [f64; 3]) -> Option<EyeSurface> {
    let len = (point_m[0] * point_m[0] + point_m[1] * point_m[1] + point_m[2] * point_m[2]).sqrt();
    if !positive(len) {
        return None;
    }
    let dir = [
        Gf::from_f64(point_m[0] / len),
        Gf::from_f64(point_m[1] / len),
        Gf::from_f64(point_m[2] / len),
    ];
    let surface = vd_terrain::height::height_m(body, dir, 0);
    Some(EyeSurface {
        surface_m: surface.to_f64(),
        altitude_m: len - surface.to_f64(),
        biome: vd_terrain::height::biome_at(body, dir, surface),
    })
}

/// THE RULER'S ANGULAR SIZE (8p, ruling V14 D8-7): the ball's radius over its distance from the eye,
/// `tan(2°)` (pinned by a test). A subject that subtends the same angle at every stand is readable
/// in every picture — MEASURED: on the ground a 1 m ball 29 m off, from 60 km up an 8.7 km ball
/// 245 km off — and the stamp states both numbers, so the owner reads the scale off the ball and
/// the gate reads the projection off its pixels. The two bytes of the probe saturate at 65 535
/// cells; a ball farther than that at its rung reads as 65 535, and the gate's distance check
/// would say so.
pub const RULER_TAN_HALF_ANGLE: f64 = 0.034_920_769_491_747_67;

/// The ruler: a ball of known size hovering over the recipe's surface where the eye's centre ray
/// meets it, in the body's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ruler {
    /// The ball's centre in the body's frame, in metres: the hit point lifted by TWO radii along
    /// the radial, so the ball hovers one radius clear of the ground. A ball resting on a slope has
    /// its uphill side cut off by the ground, and a gate that measures its disc would then measure
    /// the slope; a ball clear of the ground is the whole disc, and its shadow still falls on the
    /// ground under it, which is the orienter.
    pub centre_m: [f64; 3],
    /// The ball's radius, in metres.
    pub radius_m: f64,
    /// The ball's centre's distance from the eye, in metres.
    pub distance_m: f64,
}

/// Bisection steps refining the hit between the last sample above the surface and the first below:
/// twelve halve one cell to a four-thousandth of it.
const RULER_REFINE_STEPS: u32 = 12;

/// Where the centre ray from `eye_m` along `forward` meets the recipe's surface AT THE DRAWN RUNG
/// (so the ball rests on the drawn ground, not on a finer shape it is not standing on), marched one
/// cell at a time out to `reach_m` and refined by bisection. `None` when the ray leaves without
/// touching ground within reach (an eye looking at the sky), for a rung the ladder lacks, or for a
/// degenerate eye or direction. The ball's radius is the HIT's distance along the ray times
/// [`RULER_TAN_HALF_ANGLE`], never smaller than half a cell.
#[must_use]
pub fn ruler_on_surface(
    body: &BodyDefinition,
    eye_m: [f64; 3],
    forward: [f64; 3],
    rung: u8,
    reach_m: f64,
) -> Option<Ruler> {
    let eye = vd_core::glam::DVec3::from_array(eye_m);
    let fwd = vd_core::glam::DVec3::from_array(forward).normalize_or_zero();
    if rung >= body.ladder().rungs || !positive(eye.length()) || !positive(fwd.length()) {
        return None;
    }
    let cell = f64::from(vd_seed::ladder::cell_m(rung));
    let below = |t: f64| -> bool {
        let p = eye + fwd * t;
        let len = p.length();
        let dir = [
            Gf::from_f64(p.x / len),
            Gf::from_f64(p.y / len),
            Gf::from_f64(p.z / len),
        ];
        len <= vd_terrain::height::height_m(body, dir, rung).to_f64()
    };
    // An eye at or under the drawn surface plants nothing: the bracket would close on the eye and
    // the ball would stand at the nose (the refuter's finding 7). The stamp's altitude says why.
    if below(0.0) {
        return None;
    }
    // March: the first sample at or under the surface ends it.
    let mut above = 0.0_f64;
    let mut t = cell;
    while t <= reach_m && !below(t) {
        above = t;
        t += cell;
    }
    if t > reach_m {
        return None;
    }
    let mut lo = above;
    let mut hi = t;
    let mut n = 0;
    while n < RULER_REFINE_STEPS {
        let mid = f64::midpoint(lo, hi);
        // Branchless halving: the reading picks the side as a weight, so no arm depends on where
        // the seed happens to put the crossing (HR5 — a recipe change could not redden this).
        let under = f64::from(u8::from(below(mid)));
        hi += (mid - hi) * under;
        lo += (mid - lo) * (1.0 - under);
        n += 1;
    }
    let hit = eye + fwd * hi;
    let radial = hit.normalize_or_zero();
    let radius_m = (hi * RULER_TAN_HALF_ANGLE).max(cell * 0.5);
    let centre = hit + radial * (2.0 * radius_m);
    Some(Ruler {
        centre_m: centre.to_array(),
        radius_m,
        distance_m: (centre - eye).length(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::Face;
    use vd_terrain::digest::surface_chunk_z;
    use vd_terrain::home::{HOME_PLANET_RADIUS_BITS, HOME_PLANET_SEED, home_planet};

    fn planet() -> RealmId {
        RealmId::Planet(HOME_PLANET_SEED)
    }

    fn surface(generator: u64) -> SurfaceStmt {
        SurfaceStmt {
            frame: FrameRef::PlanetCentered {
                planet_seed: HOME_PLANET_SEED,
            },
            generator,
        }
    }

    fn look() -> Boundary {
        Boundary::Shell {
            r: f64::from_bits(HOME_PLANET_RADIUS_BITS),
        }
    }

    fn lane() -> ChunkLane {
        ChunkLane::new(Box::new(InlineWorkers::default()), 77)
    }

    fn surface_key(body: &BodyDefinition, rung: u8, x: i32, y: i32) -> ChunkKey {
        ChunkKey {
            face: Face::PosX,
            rung,
            x,
            y,
            z: surface_chunk_z(body, Face::PosX, rung, x, y),
        }
    }

    /// A refused statement is counted ONCE: the second frame's statement is ignored, and after a
    /// forget the realm is judged afresh.
    #[test]
    fn a_refused_surface_is_counted_once_until_the_realm_is_forgotten() {
        let body = home_planet();
        let declared = vd_terrain::declared_world_tag(vd_terrain::home::HOME_UNIVERSE_SEED);
        let mut lane = ChunkLane::new(Box::new(InlineWorkers::default()), declared);
        let foreign = SurfaceStmt {
            generator: declared ^ 1,
            frame: FrameRef::PlanetCentered {
                planet_seed: body.seed(),
            },
        };
        let look = Boundary::Shell {
            r: body.ladder().radius_m(),
        };
        lane.state_surface(planet(), &foreign, &look);
        lane.state_surface(planet(), &foreign, &look);
        assert_eq!(lane.counters().foreign_generator, 1);
        assert!(lane.body(planet()).is_none());
        lane.forget(planet());
        lane.state_surface(planet(), &foreign, &look);
        assert_eq!(lane.counters().foreign_generator, 2);
        // A frame without a seed: no body, counted once too.
        let seedless = SurfaceStmt {
            generator: declared,
            frame: FrameRef::SystemSpace { system_seed: 7 },
        };
        let system = RealmId::System(7);
        lane.state_surface(system, &seedless, &look);
        lane.state_surface(system, &seedless, &look);
        assert_eq!(lane.counters().no_body, 1);
    }

    #[test]
    fn a_surface_statement_becomes_the_home_planet_and_a_foreign_one_is_refused() {
        let mut lane = lane();
        lane.state_surface(planet(), &surface(77), &look());
        assert_eq!(lane.body(planet()).map(|b| **b), Some(home_planet()));
        // Stated again: the same body, nothing counted.
        lane.state_surface(planet(), &surface(77), &look());
        assert_eq!(lane.counters(), ChunkCounters::default());
        // A foreign recipe tag: refused, counted, no body.
        let other = RealmId::Planet(5);
        lane.state_surface(other, &surface(78), &look());
        assert!(lane.body(other).is_none());
        assert_eq!(lane.counters().foreign_generator, 1);
        // A frame without a seed, and a radius the ladder refuses: no body. Each on its own realm —
        // a refused realm is counted once and then ignored (see the test above).
        lane.state_surface(
            RealmId::Planet(6),
            &SurfaceStmt {
                frame: FrameRef::SystemSpace { system_seed: 1 },
                generator: 77,
            },
            &look(),
        );
        lane.state_surface(
            RealmId::Planet(7),
            &surface(77),
            &Boundary::Shell { r: -1.0 },
        );
        assert_eq!(lane.counters().no_body, 2);
        // A box look is not a round body.
        lane.state_surface(
            RealmId::Planet(8),
            &surface(77),
            &Boundary::Aabb {
                half: vd_core::glam::DVec3::ONE,
            },
        );
        assert_eq!(lane.counters().no_body, 3);
        lane.forget(planet());
        assert!(lane.body(planet()).is_none());
    }

    #[test]
    fn the_lane_builds_holds_and_releases_chunks_and_never_serves_a_second_rung() {
        let mut lane = lane();
        let body = home_planet();
        // No body yet: refused and counted.
        let k0 = surface_key(&body, 0, 300, 700);
        lane.request(planet(), k0);
        assert_eq!(lane.counters().no_body, 1);
        lane.state_surface(planet(), &surface(77), &look());
        // Outside the ladder: refused.
        lane.request(
            planet(),
            ChunkKey {
                face: Face::PosX,
                rung: 0,
                x: -1,
                y: 0,
                z: 0,
            },
        );
        assert_eq!(lane.counters().outside, 1);
        // The first request sets the realm's rung; a second request of the same key is not queued twice.
        lane.request(planet(), k0);
        lane.request(planet(), k0);
        assert!(lane.holds(planet(), k0));
        assert_eq!(lane.counters().submitted, 1);
        // A second rung: refused (D-TERRAIN-3).
        let k3 = surface_key(&body, 3, 30, 70);
        lane.request(planet(), k3);
        assert_eq!(lane.counters().second_rung, 1);
        assert!(!lane.holds(planet(), k3));
        // Harvest: the chunk is resident, with geometry relative to its origin.
        let ready = lane.poll(8);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].realm, planet());
        assert_eq!(ready[0].geometry.key, k0);
        assert_eq!(lane.resident(planet()), vec![k0]);
        assert_eq!(lane.counters().harvested, 1);
        assert!(lane.poll(8).is_empty());
        // Release the only chunk: the realm's rung is free, and rung 3 is served.
        lane.release(planet(), k0);
        assert!(!lane.holds(planet(), k0));
        lane.request(planet(), k3);
        assert!(lane.holds(planet(), k3));
        // Release while building: never handed out.
        lane.release(planet(), k3);
        assert!(lane.poll(8).is_empty());
        assert!(lane.resident(planet()).is_empty());
        // A bounded harvest: two chunks queued, one per poll.
        let k1 = surface_key(&body, 0, 301, 700);
        lane.request(planet(), k0);
        lane.request(planet(), k1);
        assert_eq!(lane.pending_count(), 2);
        assert_eq!(lane.poll(1).len(), 1);
        assert_eq!(lane.pending_count(), 1);
        assert_eq!(lane.poll(1).len(), 1);
        assert_eq!(lane.pending_count(), 0);
        assert_eq!(lane.resident(planet()).len(), 2);
        // Releasing ONE of two keeps the realm's rung: the other is still held, so rung 3 is refused.
        lane.release(planet(), k0);
        lane.request(planet(), k3);
        assert_eq!(lane.counters().second_rung, 2);
        assert!(!lane.holds(planet(), k3));
        // A release of a chunk never held changes nothing.
        lane.release(planet(), k3);
        assert_eq!(lane.resident(planet()), vec![k1]);
        // Forgetting a realm withdraws what is still building: nothing is ever handed out for it.
        lane.request(planet(), k0);
        lane.forget(planet());
        assert!(lane.poll(8).is_empty());
        assert_eq!(lane.pending_count(), 0);
        // The job and the finished chunk are plain data (their derives are exercised here).
        let job = ChunkJob {
            realm: planet(),
            body: Arc::new(home_planet()),
            key: k0,
        };
        assert!(format!("{:?}", job.clone()).contains("ChunkJob"));
        let mut inline = InlineWorkers::default();
        inline.submit(job);
        let mut out = Vec::new();
        inline.drain(&mut out, 8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].clone(), out[0]);
        assert!(format!("{:?}", out[0]).contains("ChunkReady"));
    }

    /// THE TRIM GATE (ruling S7-10): what the engine draws is the extractor's vertex, to the float's
    /// own rounding — the client moved nothing.
    #[test]
    fn the_geometry_is_the_extractors_positions_around_the_chunks_origin() {
        let body = home_planet();
        let key = surface_key(&body, 0, 300, 700);
        let g = geometry_of(&body, key).expect("in the band");
        let samples = sample_box(&body, key).expect("in the band");
        let mesh = extract(&samples);
        assert_eq!(g.triangles, mesh.triangles);
        assert_eq!(g.vertices.len(), mesh.vertices.len());
        assert_eq!(g.normals.len(), mesh.vertices.len());
        let mut worst = 0.0f64;
        for (v, rel) in mesh.vertices.iter().zip(g.vertices.iter()) {
            let p = vertex_position_m(&body, &samples, *v);
            let mut k = 0;
            while k < 3 {
                let drawn = g.origin_m[k] + f64::from(rel[k]);
                worst = worst.max((drawn - p[k].to_f64()).abs());
                k += 1;
            }
            // Relative to the origin, a chunk is at most 62 m wide: the float holds micrometres.
            assert!(rel.iter().all(|c| c.abs() < 2.0 * CHUNK_EDGE as f32));
        }
        assert!(
            worst < 1e-4,
            "the drawn vertex is the extractor's: {worst} m"
        );
        // The origin is the middle cell's centre, on the body's radius scale.
        let r = (g.origin_m[0] * g.origin_m[0]
            + g.origin_m[1] * g.origin_m[1]
            + g.origin_m[2] * g.origin_m[2])
            .sqrt();
        assert!((r - body.ladder().radius_m()).abs() < 20_000.0, "{r}");
        // Every normal is unit length and points away from the planet on a surface chunk (rock
        // below, air above): the dot with the outward direction is positive for most.
        let mut outward = 0;
        for (n, rel) in g.normals.iter().zip(g.vertices.iter()) {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "{n:?}");
            let p = [
                g.origin_m[0] + f64::from(rel[0]),
                g.origin_m[1] + f64::from(rel[1]),
                g.origin_m[2] + f64::from(rel[2]),
            ];
            // Branchless (HR5): on the earth-like home every normal of this chunk faces outward,
            // so an `if` here would carry an arm no world reaches.
            let dot = f64::from(n[0]) * p[0] + f64::from(n[1]) * p[1] + f64::from(n[2]) * p[2];
            outward += usize::from(dot > 0.0);
        }
        let total = g.normals.len();
        assert!(outward * 2 > total, "{outward} of {total} face outward");
        // A key outside the body: none.
        assert_eq!(
            geometry_of(
                &body,
                ChunkKey {
                    face: Face::PosX,
                    rung: 99,
                    x: 0,
                    y: 0,
                    z: 0
                }
            ),
            None
        );
    }

    #[test]
    fn smooth_normals_average_the_adjoining_faces_and_leave_an_unused_vertex_at_zero() {
        // A square in the a–b plane, cut into two triangles, both facing +c.
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
        ];
        let n = smooth_normals(&v, &[[0, 1, 2], [0, 2, 3]]);
        for nn in n.iter().take(4) {
            assert_eq!(*nn, [0.0, 0.0, 1.0]);
        }
        assert_eq!(
            n[4],
            [0.0, 0.0, 0.0],
            "an unused vertex keeps a zero normal"
        );
    }

    #[test]
    fn the_chunks_around_a_point_are_the_columns_near_it_at_one_rung() {
        let body = home_planet();
        let r = body.ladder().radius_m();
        // A point on the +X face, near its centre: radius 1 → 3 × 3 columns × up to 3 chunks.
        let keys = chunks_around(&body, [r, 1000.0, -500.0], 0, 1);
        assert!(keys.len() >= 9);
        assert!(keys.len() <= 27);
        assert!(keys.iter().all(|k| k.face == Face::PosX));
        assert!(keys.iter().all(|k| k.rung == 0));
        let xs: BTreeSet<i32> = keys.iter().map(|k| k.x).collect();
        assert_eq!(xs.len(), 3);
        // At the coarsest rung, near the (−u, −v) corner of −Y: columns past the face are left out.
        let top = body.ladder().rungs - 1;
        let corner = chunks_around(&body, [-r, -r, -r], top, 2);
        assert!(!corner.is_empty());
        assert!(corner.iter().all(|k| k.x >= 0));
        assert!(corner.iter().all(|k| k.y >= 0));
        // Every key is in the ladder.
        for k in corner.iter().chain(keys.iter()) {
            assert!(vd_terrain::lattice::in_ladder(&body, *k), "{k:?}");
        }
        // The centre, a NaN, a rung the body lacks, or an eye farther than the far bound: nothing.
        assert!(chunks_around(&body, [0.0, 0.0, 0.0], 0, 1).is_empty());
        assert!(chunks_around(&body, [f64::NAN, 0.0, 0.0], 0, 1).is_empty());
        // The radius is clamped: an absurd one is the widest allowed, a negative one is zero.
        let widest = chunks_around(&body, [r, 1000.0, -500.0], 3, MAX_RADIUS);
        assert_eq!(
            chunks_around(&body, [r, 1000.0, -500.0], 3, i32::MAX),
            widest
        );
        assert_eq!(
            chunks_around(&body, [r, 1000.0, -500.0], 3, -5),
            chunks_around(&body, [r, 1000.0, -500.0], 3, 0)
        );
        assert!(!widest.is_empty());
        assert!(chunks_around(&body, [r, 0.0, 0.0], 99, 1).is_empty());
        assert!(chunks_around(&body, [r * FAR_EYE_RADII * 1.01, 0.0, 0.0], 0, 1).is_empty());
        assert!(!chunks_around(&body, [r * FAR_EYE_RADII * 0.99, 0.0, 0.0], 0, 1).is_empty());
    }

    /// THE COLUMN UNDER THE EYE IS UNDER THE EYE (the ground picture's measured defect: the tangents
    /// a direction gives were read as the face parameter, and the column asked for lay 230 km from
    /// the eye). At every rung and away from a face's centre — where the bend is largest — the
    /// middle column's own direction is within one chunk of the point.
    #[test]
    fn the_column_under_a_point_holds_the_point_at_every_rung() {
        use vd_seed::bend::{direction, face_of};
        use vd_seed::ladder::face_param;
        let body = home_planet();
        let r = body.ladder().radius_m();
        let edge = CHUNK_EDGE as i32;
        // The ground picture's own standing point: well off the +X face's centre.
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let point = [d[0] * r, d[1] * r, d[2] * r];
        let mut rung = 0u8;
        while rung < body.ladder().rungs {
            let keys = chunks_around(&body, point, rung, 0);
            assert!(!keys.is_empty(), "rung {rung}");
            let n_l = body.ladder().cells_per_edge(rung);
            for k in &keys {
                assert_eq!(k.face, face_of(d));
                // The chunk's middle cell as a direction, against the point's own.
                let a = face_param(k.x * edge + edge / 2, n_l);
                let b = face_param(k.y * edge + edge / 2, n_l);
                let c = direction(k.face, a, b);
                let cos = c[0] * d[0] + c[1] * d[1] + c[2] * d[2];
                let angle = cos.clamp(-1.0, 1.0).acos();
                // One chunk's own size at that rung (its edge in metres), generously.
                let chunk_m = f64::from(vd_seed::ladder::cell_m(rung)) * f64::from(edge);
                assert!(angle * r <= chunk_m, "rung {rung}: {} m off", angle * r);
            }
            rung += 1;
        }
    }

    #[test]
    fn a_body_frame_point_undoes_the_rows_placement_and_facing() {
        // The planet sits at (100, 0, 0) turned a quarter turn about +Z; the eye at (100, 10, 0).
        let q = vd_core::glam::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let p = body_frame_point([100.0, 10.0, 0.0], [100.0, 0.0, 0.0], [q.x, q.y, q.z, q.w]);
        // In the body's frame the eye is at (10, 0, 0): the inverse quarter turn of (0, 10, 0).
        assert!((p[0] - 10.0).abs() < 1e-9, "{p:?}");
        assert!(p[1].abs() < 1e-9);
        assert!(p[2].abs() < 1e-9);
    }

    #[test]
    fn the_eye_surface_reads_the_recipe_under_the_eye_and_refuses_the_centre() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let read = eye_surface(&body, eye).expect("a direction");
        assert_eq!(eye_surface(&body, eye), Some(read));
        assert!((read.surface_m - surface).abs() < 1e-6, "{read:?}");
        assert!((read.altitude_m - 3.4).abs() < 1e-6, "{read:?}");
        assert_eq!(
            read.biome,
            vd_terrain::height::biome_at(&body, dir, Gf::from_f64(surface))
        );
        assert_eq!(eye_surface(&body, [0.0, 0.0, 0.0]), None);
        assert_eq!(eye_surface(&body, [f64::NAN, 0.0, 0.0]), None);
    }

    #[test]
    fn the_ruler_stands_on_the_drawn_ground_ahead_and_is_absent_for_a_sky_ray() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let up = vd_core::glam::DVec3::from_array(d);
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        let eye = up * (surface + 3.4);
        // A level direction tilted 8° down, like the ground picture's nose.
        let level = up.cross(vd_core::glam::DVec3::Z).normalize();
        let tilt = 8.0_f64.to_radians();
        let nose = (level * tilt.cos() - up * tilt.sin()).normalize();
        let ruler = ruler_on_surface(&body, eye.to_array(), nose.to_array(), 0, 400.0)
            .expect("the ray meets the ground");
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), nose.to_array(), 0, 400.0),
            Some(ruler)
        );
        // The ball hovers one radius clear of the rung-0 surface at its own hit direction: its
        // centre stands two radii over the recipe there.
        let c = vd_core::glam::DVec3::from_array(ruler.centre_m);
        let cd = c.normalize();
        let there = vd_terrain::height::height_m(
            &body,
            [Gf::from_f64(cd.x), Gf::from_f64(cd.y), Gf::from_f64(cd.z)],
            0,
        )
        .to_f64();
        let centre_len = c.length();
        assert!(
            (centre_len - there - 2.0 * ruler.radius_m).abs() < 0.01,
            "{ruler:?}: centre {centre_len} m, surface {there} m"
        );
        assert!(
            (ruler.distance_m - (c - eye).length()).abs() < 1e-9,
            "{ruler:?}"
        );
        assert!(ruler.distance_m > 3.0, "{ruler:?}");
        assert!(ruler.distance_m < 400.0, "{ruler:?}");
        // The size law: the larger of the angular size AT THE HIT and half a cell.
        let hit = c - cd * (2.0 * ruler.radius_m);
        let by_angle = (hit - eye).length() * RULER_TAN_HALF_ANGLE;
        assert!(
            (ruler.radius_m - by_angle.max(0.5)).abs() < 1e-3,
            "{ruler:?} vs {by_angle}"
        );
        // A ray to the sky meets nothing within reach.
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), up.to_array(), 0, 400.0),
            None
        );
        // Straight down from aloft at a coarse rung: the hit is under the eye, one cell's step.
        let aloft = up * (surface + 60_000.0);
        let down = ruler_on_surface(&body, aloft.to_array(), (-up).to_array(), 9, 400_000.0)
            .expect("the ground is below");
        assert!(down.radius_m > 256.0, "{down:?}");
        // The centre stands two radii over the rung-9 surface, which lies within the dropped
        // octaves' bound of the rung-0 surface the eye's height was set from.
        assert!(
            (down.distance_m + 2.0 * down.radius_m - 60_000.0).abs() < 2_000.0,
            "{down:?}"
        );
        let dc = vd_core::glam::DVec3::from_array(down.centre_m);
        let dd = dc.normalize();
        let there9 = vd_terrain::height::height_m(
            &body,
            [Gf::from_f64(dd.x), Gf::from_f64(dd.y), Gf::from_f64(dd.z)],
            9,
        )
        .to_f64();
        let down_len = dc.length();
        assert!(
            (down_len - there9 - 2.0 * down.radius_m).abs() < 1.0,
            "{down:?}: centre {down_len} m, rung-9 surface {there9} m"
        );
        // An eye under the drawn surface plants nothing.
        let buried = up * (surface - 5.0);
        assert_eq!(
            ruler_on_surface(&body, buried.to_array(), nose.to_array(), 0, 400.0),
            None
        );
        // The angular size is two degrees, pinned to the trigonometry it names.
        assert!((RULER_TAN_HALF_ANGLE - 2.0_f64.to_radians().tan()).abs() < 1e-15);
        // Refusals: a rung the ladder lacks, a degenerate eye, a zero direction.
        assert_eq!(
            ruler_on_surface(
                &body,
                eye.to_array(),
                nose.to_array(),
                body.ladder().rungs,
                400.0
            ),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, [0.0, 0.0, 0.0], nose.to_array(), 0, 400.0),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), [0.0, 0.0, 0.0], 0, 400.0),
            None
        );
    }
}
