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
//! **Every rung at once** (slice 8 step 2): the ladder view (`ladder_view`) names the chunks of every
//! ring out to the horizon, coarsest first, and the lane serves them all. The one-rung refusal of
//! slice 7 (`D-TERRAIN-3`) is gone with its flag.
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
use std::sync::{Arc, Mutex};

use vd_core::geometry::Boundary;
use vd_core::glam::DVec3;
use vd_core::look::SurfaceStmt;
use vd_core::pose::{FrameRef, RealmId};
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::extract::{extract, extract_all_edges};
use vd_terrain::lattice::sample_box;
use vd_terrain::position::vertex_position_m;

/// HOW FAR A RUNG SINKS under the next finer one (slice 8 step 3): the recipe's own bound on the
/// gap between the two fields (the octaves the coarser rung drops) plus a cell of each rung for
/// the extractor's own placement, which lies within a vertex's group. Under that, the coarser
/// mesh is certainly below the finer one; zero at rung 0, which is under nothing.
///
/// ★ SINCE THE CAP-ROCK BENCH (slice 8a stage 4) the difference of the two dropped bounds IS the
/// body's own handover step (`vd_terrain::BodyDefinition::step_bound`), which carries the terrace's
/// Lipschitz word and the fade's own step as well as the dropped octaves. The bench amplifies every
/// variation under it, so a sink sized on the octaves alone would be short and a coarse mesh would
/// show through a finer one at a fade edge.
#[must_use]
pub fn sink_m(body: &BodyDefinition, rung: u8) -> f64 {
    if rung == 0 {
        return 0.0;
    }
    let finer = rung - 1;
    body.dropped_bound_m(rung) - body.dropped_bound_m(finer)
        + f64::from(vd_seed::ladder::cell_m(rung))
        + f64::from(vd_seed::ladder::cell_m(finer))
}

/// A finished chunk: integer vertices turned into floats around a floating origin.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkGeometry {
    pub key: ChunkKey,
    /// The chunk's own origin in the realm's frame, in metres: the centre of its middle cell.
    pub origin_m: [f64; 3],
    /// THE MORPH TARGETS (slice 8 step 3), PACKED (step 5, ruling V16): where each vertex stands on
    /// the NEXT COARSER rung's MESH along its own radial ([`ParentMesh`]), as ONE signed metre
    /// along that radial from the vertex — the target lies on the radial by construction (a
    /// radial hit, or the coarser field in the same direction), so the target is `vertex + radial
    /// × morph_m` ([`ChunkGeometry::morph_target`]), exact to float rounding, at a third of the
    /// bytes. The crossfade slides a vertex from its own position to the target across the rung's
    /// band, so the finer surface becomes the coarser before it is dropped: at the band's far edge
    /// the finer vertices lie on the coarser triangles. At the top rung every value is zero.
    pub morph_m: Vec<f32>,
    /// How many SURFACE vertices (within the sink bound of the coarser field) had no parent
    /// triangle on their radial within that bound and read the field instead — a diagnosis
    /// count the stamp carries and the picture gate bounds (MEASURED near zero). A cave's own
    /// vertex, far under the field, reads the field by nature and is not counted.
    pub morph_fallbacks: u32,
    /// How many vertices stand on or past the face's edge and read the field by rule (the
    /// parents are one face's; both faces read the field for the vertex they share).
    pub morph_seam: u32,
    /// THE SINK (slice 8 step 3), PACKED (step 5): [`sink_m`] of the chunk's rung, in metres along
    /// every vertex's radial — ONE number per chunk, the shader's uniform; a vertex's sink vector
    /// is `radial × sink_m` ([`ChunkGeometry::sink_of`]). How far the chunk drops under the next
    /// finer rung's surface nearer than the finer rung's fade-out edge, so a coarser surface never
    /// shows through a finer one (the dropped octaves cut both ways) and the two are continuous
    /// at the edge, where the finer stands on the coarser. A rung-0 chunk sinks nowhere.
    pub sink_m: f32,
    /// Vertices in metres, relative to `origin_m`.
    pub vertices: Vec<[f32; 3]>,
    /// Per-vertex unit normals, derived from the triangles (style, never shape).
    pub normals: Vec<[f32; 3]>,
    /// THE PACKED NORMAL of each vertex (ruling V18): `oct_encode` of `normals`, built by the
    /// worker — the harvest loop is the main thread's, and the encoder's four candidates a vertex
    /// do not belong there (refutation N-2, the same rule as the bounds, P-8). The engine chooses
    /// by rung which form a mesh carries.
    pub packed_normals: Vec<[i16; 2]>,
    /// THE MORPH NORMAL of each vertex (D-TERRAIN-5 item 20): the shade the next coarser rung
    /// draws where this vertex's morph target lies — the parent's smooth normal at the radial's
    /// hit — packed like `packed_normals`; the vertex's own normal where no parent triangle is on
    /// its radial (the field's own, the top rung's, a cave's). The shader blends it with the own
    /// normal across the fade-out band as it blends the positions, so the shade hands over with
    /// the shape. MEASURED without it (§22.2): at 240 m/s 13 % of the pixels at the rung 1→2
    /// handover stepped by up to 48 levels in one frame.
    pub morph_normals: Vec<[i16; 2]>,
    /// THE RADIAL of each vertex (step 5): its unit direction from the body's centre in the
    /// realm's frame, narrowed once to `f32` — the shader needs no centre of the body (MEASURED
    /// with a centre uniform instead: a moving eye rewrote every material and the engine
    /// re-prepared them, 41 → 18 frames a second at 240 m/s). Four signed 16-bit quanta were
    /// MEASURED too: 4 bytes fewer a vertex, and on the hill picture 5 pixels of 701 472 fell on
    /// the neighbouring triangle at a crease (up to 19 of 255 levels) — a change of the picture,
    /// which ruling V16 refuses; the owner holds that option with its numbers.
    pub radials: Vec<[f32; 3]>,
    /// The extractor's triangles, unchanged.
    pub triangles: Vec<[u32; 3]>,
    /// Where the skirt vertices begin: every vertex before this index stands on the surface (one
    /// per surface cell, the extractor's own), every one from it hangs under an edge. The far-rung
    /// splat (D8-8's measurement) draws the surface ones only.
    pub skirt_start: u32,
    /// THE BOX that holds every vertex, its morph target and its sunk position — wherever the
    /// vertex stage can put a vertex, so the engine culls by it. Built by the worker (refutation
    /// P-8: on the main thread it cost two square roots a vertex in the harvest loop). Zero for an
    /// empty geometry.
    pub bounds: ([f32; 3], [f32; 3]),
}

/// The widest angle between a vertex's narrowed radial and the exact one: `f32`'s own rounding,
/// about 1e-7 rad — a third of a micrometre on a target three metres out at the finest rung.
pub const RADIAL_STEP_RAD: f64 = 2.0e-7;

/// The bytes a ground vertex costs the engine at the packed rungs (position 12, packed normal
/// 4, morph normal 4, morph 4, radial 12 — ruling V18's form with item 20); the exact rungs cost
/// eight more.
pub const UPLOAD_VERTEX_BYTES: u64 = 36;

impl ChunkGeometry {
    /// THE BYTES THIS CHUNK UPLOADS (ruling V15 item 2, the harvest's byte budget): its vertices
    /// at the packed stride and its indices at 16 or 32 bits — the engine's own count within a
    /// fifth (the exact rungs' wider normal), enough for a budget.
    #[must_use]
    pub fn upload_bytes(&self) -> u64 {
        let index = if self.vertices.len() <= usize::from(u16::MAX) {
            2
        } else {
            4
        };
        self.vertices.len() as u64 * UPLOAD_VERTEX_BYTES + self.triangles.len() as u64 * 3 * index
    }

    /// The unit radial of vertex `i`: from the body's centre through the vertex, in `f64`.
    fn radial(&self, i: usize) -> DVec3 {
        let v = self.vertices[i];
        (DVec3::from_array(self.origin_m)
            + DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])))
        .normalize()
    }

    /// The morph target of vertex `i`, relative to the origin like the vertex.
    #[must_use]
    pub fn morph_target(&self, i: usize) -> [f32; 3] {
        let v = self.vertices[i];
        let t = DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))
            + self.radial(i) * f64::from(self.morph_m[i]);
        [t.x as f32, t.y as f32, t.z as f32]
    }

    /// Every vertex's morph target, in the vertices' order.
    #[must_use]
    pub fn morph_targets(&self) -> Vec<[f32; 3]> {
        (0..self.vertices.len())
            .map(|i| self.morph_target(i))
            .collect()
    }

    /// The sink vector of vertex `i`: its radial times the chunk's sink.
    #[must_use]
    pub fn sink_of(&self, i: usize) -> [f32; 3] {
        let s = self.radial(i) * f64::from(self.sink_m);
        [s.x as f32, s.y as f32, s.z as f32]
    }

    /// The box over every vertex, its morph target and its sunk position, as the worker builds
    /// it into `bounds`. Zero for an empty geometry.
    #[must_use]
    pub fn measure_bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut take = |p: [f32; 3]| {
            let mut k = 0;
            while k < 3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
                k += 1;
            }
        };
        let mut i = 0;
        while i < self.vertices.len() {
            let v = self.vertices[i];
            let s = self.sink_of(i);
            take(v);
            take(self.morph_target(i));
            take([v[0] - s[0], v[1] - s[1], v[2] - s[2]]);
            i += 1;
        }
        if self.vertices.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        (lo, hi)
    }
}

/// A job for the workers: one chunk of one body.
#[derive(Clone, Debug)]
pub struct ChunkJob {
    pub realm: RealmId,
    pub body: Arc<BodyDefinition>,
    pub key: ChunkKey,
    /// The parent meshes the geomorph reads its targets from, shared by every job of the lane.
    pub parents: Arc<ParentCache>,
    /// THE ORDER (ruling V15): the lower builds first. The wanted set's own order — urgent before
    /// revealed before margin, the coarser rung first, then by parent, the nearest first — so
    /// the workers build neighbours back to back and their parent cache holds the parents.
    pub priority: u32,
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
    /// THE BUILD COUNT (M8-2a, ruling V15): the jobs the workers ran since the start, with or
    /// without a geometry, and the wall time they spent on them (a wait on a sibling's parent
    /// build included) — the workers' own rate, read against the harvest's and the frame's.
    fn built(&self) -> BuildCount;
    /// Move a job still waiting in the queue to `priority` (refutation T-1: a queued job kept
    /// the priority of the frame that first asked for it). A job already taken, or unknown, is
    /// left alone.
    fn reprioritise(&mut self, realm: RealmId, key: ChunkKey, priority: u32);
    /// Move up to `max` finished jobs into `out`.
    fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize);
    /// Withdraw a job the lane no longer wants: one not started is never run, one finished is not
    /// handed out. A job already running finishes and is dropped at the lane's `poll`.
    fn cancel(&mut self, realm: RealmId, key: ChunkKey);
}

/// What the workers ran: jobs finished (with or without a geometry), and the wall nanoseconds
/// spent on them (zero where the host has no clock — the inline workers count jobs only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildCount {
    pub chunks: u64,
    pub nanos: u64,
    /// ★ THE WORST SINGLE BUILD (2026-09-16, the walk-gap measurement): the longest wall time ONE
    /// chunk took, and the chunk it took it on. `nanos` is a SUM, so one thirty-millisecond chunk
    /// hides inside a mean of fourteen and a flight cannot tell a dense chunk from a busy queue.
    /// Zero and `None` where the host has no clock (the inline workers).
    pub peak_nanos: u64,
    pub peak_key: Option<ChunkKey>,
}

/// The inline workers: a job runs on the calling thread at `submit`, and waits in a queue for
/// `drain`. The Tier-A implementation, and a correct one for a single-threaded host.
#[derive(Default)]
pub struct InlineWorkers {
    done: std::collections::VecDeque<ChunkReady>,
    built: u64,
}

impl ChunkWorkers for InlineWorkers {
    fn submit(&mut self, job: ChunkJob) {
        // Branchless (HR5): the lane refuses a key outside the ladder before it submits, so the
        // `None` arm of the seam's own refusal is never reached through the lane.
        self.done.extend(
            geometry_with(&job.body, job.realm, job.key, &job.parents).map(|geometry| ChunkReady {
                realm: job.realm,
                geometry,
            }),
        );
        self.built += 1;
    }

    fn built(&self) -> BuildCount {
        BuildCount {
            chunks: self.built,
            nanos: 0,
            peak_nanos: 0,
            peak_key: None,
        }
    }

    fn reprioritise(&mut self, _realm: RealmId, _key: ChunkKey, _priority: u32) {
        // A job runs at `submit`: nothing waits.
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

/// THE PARENT MESH (slice 8 step 3): a coarser chunk's whole surface — every crossed edge of its
/// box, the halo's included (`extract_all_edges`) — as world positions, with its triangles
/// bucketed by the box's own lattice cell across the face, so a finer vertex's radial finds the
/// triangles under it in one lookup. The geomorph's target is where the finer vertex's radial
/// meets THIS mesh, not the coarser field: the drawn coarser surface is the extractor's mesh,
/// which lies within a cell of the field, and MEASURED with the field as the target the two
/// surfaces crossed each other along every fade-out edge — the coarser mesh's bumps over the
/// finer surface shadowed it (34 dark specks on the hill at 950 m, 1.9 km and 3.8 km). Two
/// finer neighbours read one and the same parent surface for the vertex they share, because
/// every crossing of the parent's box is in it, owned or not.
#[derive(Debug)]
pub struct ParentMesh {
    key: ChunkKey,
    /// The mesh's origin in the body's frame, in metres: the centre of the parent's middle cell.
    /// Positions are `f32` from it (step 5, D-TERRAIN-5 item 10): a chunk of 248 m at rung 2
    /// rounds to 15 µm, the coarsest of 127 km to 8 mm — under a hundredth of a pixel at any
    /// switch distance.
    origin: DVec3,
    positions: Vec<[f32; 3]>,
    /// THE PARENT'S OWN SHADING (D-TERRAIN-5 item 20): its smooth per-vertex normals, packed as
    /// the drawn chunks pack theirs, so a finer vertex that morphs onto a parent triangle can
    /// carry the shade the parent draws there.
    normals: Vec<[i16; 2]>,
    triangles: ParentTriangles,
    /// THE BUCKETS as one table: for the lattice cell `(a, b)` of the parent's box, `−1..=62`
    /// each, the triangles that span it lie in `bucket_tris[bucket_start[c]..bucket_start[c+1]]`
    /// with `c = (a + 1) × 64 + (b + 1)`. A triangle is in every cell its vertices span.
    bucket_start: Vec<u32>,
    bucket_tris: Vec<u32>,
}

/// A parent mesh's triangles at 16 bits where its vertices fit, else at 32.
#[derive(Debug)]
enum ParentTriangles {
    Narrow(Vec<[u16; 3]>),
    Wide(Vec<[u32; 3]>),
}

impl ParentTriangles {
    fn pack(triangles: Vec<[u32; 3]>, vertices: usize) -> ParentTriangles {
        if vertices <= usize::from(u16::MAX) {
            ParentTriangles::Narrow(
                triangles
                    .iter()
                    .map(|t| [t[0] as u16, t[1] as u16, t[2] as u16])
                    .collect(),
            )
        } else {
            ParentTriangles::Wide(triangles)
        }
    }

    fn len(&self) -> usize {
        match self {
            ParentTriangles::Narrow(v) => v.len(),
            ParentTriangles::Wide(v) => v.len(),
        }
    }

    fn get(&self, i: usize) -> [u32; 3] {
        match self {
            ParentTriangles::Narrow(v) => {
                let t = v[i];
                [u32::from(t[0]), u32::from(t[1]), u32::from(t[2])]
            }
            ParentTriangles::Wide(v) => v[i],
        }
    }

    fn bytes(&self) -> usize {
        match self {
            ParentTriangles::Narrow(v) => v.capacity() * std::mem::size_of::<[u16; 3]>(),
            ParentTriangles::Wide(v) => v.capacity() * std::mem::size_of::<[u32; 3]>(),
        }
    }
}

/// The lattice cells of a parent's box along one axis, `−1..=62`.
const PARENT_CELLS: i32 = 64;

/// The bucket of the lattice cell `(a, b)`, or `None` outside the box's cells.
/// The bucket of a cell the mesh's own vertices stand in: the extractor keeps every vertex
/// inside the box (its own test asserts the bound), so the cell is always in the grid, and a
/// vertex outside it is a defect that stops the build here.
fn bucket_of(a: i32, b: i32) -> usize {
    parent_bucket(a, b).expect("the extractor keeps every vertex inside the parent's box")
}

fn parent_bucket(a: i32, b: i32) -> Option<usize> {
    let (a, b) = (a + 1, b + 1);
    ((0..PARENT_CELLS).contains(&a) & (0..PARENT_CELLS).contains(&b))
        .then(|| (a * PARENT_CELLS + b) as usize)
}

impl ParentMesh {
    /// The whole surface of the chunk at `key`; `None` outside the body.
    #[must_use]
    pub fn build(body: &BodyDefinition, key: ChunkKey) -> Option<ParentMesh> {
        let samples = sample_box(body, key)?;
        let mesh = extract_all_edges(&samples);
        let q = i32::from(vd_terrain::VERTEX_QUANTUM as i16);
        let cell_of = |v: [i16; 3]| -> (i32, i32) {
            (i32::from(v[0]).div_euclid(q), i32::from(v[1]).div_euclid(q))
        };
        let half = (CHUNK_EDGE / 2) as i16 * vd_terrain::VERTEX_QUANTUM as i16;
        let o = vertex_position_m(body, &samples, [half, half, half]);
        let origin = DVec3::new(o[0], o[1], o[2]);
        let positions: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|v| {
                let p = vertex_position_m(body, &samples, *v);
                [
                    (p[0] - origin.x) as f32,
                    (p[1] - origin.y) as f32,
                    (p[2] - origin.z) as f32,
                ]
            })
            .collect();
        // The buckets, counted then filled: one pass for the counts, a prefix sum, one pass for
        // the triangles — a flat table, no map and no vector per cell.
        let cells = (PARENT_CELLS * PARENT_CELLS) as usize;
        let mut counts = vec![0u32; cells + 1];
        let spans: Vec<(i32, i32, i32, i32)> = mesh
            .triangles
            .iter()
            .map(|t| {
                let c = [
                    cell_of(mesh.vertices[t[0] as usize]),
                    cell_of(mesh.vertices[t[1] as usize]),
                    cell_of(mesh.vertices[t[2] as usize]),
                ];
                (
                    c[0].0.min(c[1].0).min(c[2].0),
                    c[0].0.max(c[1].0).max(c[2].0),
                    c[0].1.min(c[1].1).min(c[2].1),
                    c[0].1.max(c[1].1).max(c[2].1),
                )
            })
            .collect();
        for (a0, a1, b0, b1) in &spans {
            let mut a = *a0;
            while a <= *a1 {
                let mut b = *b0;
                while b <= *b1 {
                    counts[bucket_of(a, b) + 1] += 1;
                    b += 1;
                }
                a += 1;
            }
        }
        let mut c = 1;
        while c <= cells {
            counts[c] += counts[c - 1];
            c += 1;
        }
        let bucket_start = counts;
        let mut fill = bucket_start.clone();
        let mut bucket_tris = vec![0u32; bucket_start[cells] as usize];
        for (i, (a0, a1, b0, b1)) in spans.iter().enumerate() {
            let mut a = *a0;
            while a <= *a1 {
                let mut b = *b0;
                while b <= *b1 {
                    let c = bucket_of(a, b);
                    bucket_tris[fill[c] as usize] = i as u32;
                    fill[c] += 1;
                    b += 1;
                }
                a += 1;
            }
        }
        let vertices = positions.len();
        let normals = smooth_normals(&positions, &mesh.triangles)
            .iter()
            .map(|n| oct_encode(*n))
            .collect();
        Some(ParentMesh {
            key,
            origin,
            positions,
            normals,
            triangles: ParentTriangles::pack(mesh.triangles, vertices),
            bucket_start,
            bucket_tris,
        })
    }

    /// The key this mesh is the surface of.
    #[must_use]
    pub fn key(&self) -> ChunkKey {
        self.key
    }

    /// How many triangles the surface holds.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// A vertex's position in the body's frame, in `f64`: the origin plus the narrowed offset.
    fn position(&self, i: u32) -> DVec3 {
        let p = self.positions[i as usize];
        self.origin + DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
    }

    /// AN ESTIMATE of the bytes the mesh holds (M8-2a): the vectors' capacities — the positions,
    /// the triangles, the bucket table — what one cache entry costs, within the allocator's own
    /// rounding.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.positions.capacity() * std::mem::size_of::<[f32; 3]>()
            + self.normals.capacity() * std::mem::size_of::<[i16; 2]>()
            + self.triangles.bytes()
            + self.bucket_start.capacity() * std::mem::size_of::<u32>()
            + self.bucket_tris.capacity() * std::mem::size_of::<u32>()
    }

    /// The radius at which the radial along `dir` (a unit vector from the body's centre) meets
    /// this mesh within lattice cell `(a, b)`, nearest to `near_m` when it meets it more than
    /// once (a cave under the surface); `None` when no triangle of that cell is on the radial.
    #[must_use]
    pub fn radial_hit_m(&self, a: i32, b: i32, dir: DVec3, near_m: f64) -> Option<f64> {
        self.radial_hit(a, b, dir, near_m).map(|(h, _)| h)
    }

    /// [`radial_hit_m`](Self::radial_hit_m) with THE PARENT'S SHADE at the hit (item 20): the
    /// parent's smooth normal interpolated over the triangle the radial met, as the parent draws
    /// it there — what a finer vertex morphing onto that triangle must shade like at the band's
    /// far edge.
    #[must_use]
    pub fn radial_hit(&self, a: i32, b: i32, dir: DVec3, near_m: f64) -> Option<(f64, DVec3)> {
        // The nearest hit with its triangle and the crossing's weights; the shade is decoded once,
        // for the winner alone (refutation, 2026-09-12: every candidate was decoded).
        let mut best: Option<(f64, [u32; 3], f64, f64)> = None;
        let range = parent_bucket(a, b).map_or(0..0, |c| {
            self.bucket_start[c] as usize..self.bucket_start[c + 1] as usize
        });
        for i in &self.bucket_tris[range] {
            let t = self.triangles.get(*i as usize);
            let hit = ray_triangle(
                dir,
                self.position(t[0]),
                self.position(t[1]),
                self.position(t[2]),
            );
            // The nearest hit; a tie goes to the lower one, so the answer never depends on the
            // order the triangles were met in.
            best = match (best, hit) {
                (Some((b, _, _, _)), Some((h, u, v)))
                    if ((h - near_m).abs(), h) < ((b - near_m).abs(), b) =>
                {
                    Some((h, t, u, v))
                }
                (None, Some((h, u, v))) => Some((h, t, u, v)),
                (b, _) => b,
            };
        }
        best.map(|(h, t, u, v)| {
            let n = |k: u32| DVec3::from_array(oct_decode(self.normals[k as usize]).map(f64::from));
            let shade = (n(t[0]) * (1.0 - u - v) + n(t[1]) * u + n(t[2]) * v).normalize_or_zero();
            (h, shade)
        })
    }
}

/// THE RADIAL AGAINST A TRIANGLE: the distance from the body's centre at which the ray along the
/// unit `dir` crosses the triangle `p0 p1 p2` (Möller–Trumbore, in metres), `None` when it does
/// not. A crossing on an edge or a corner counts on both sides, so two triangles that share the
/// edge both answer, with the same distance.
#[must_use]
pub fn ray_triangle_m(dir: DVec3, p0: DVec3, p1: DVec3, p2: DVec3) -> Option<f64> {
    ray_triangle(dir, p0, p1, p2).map(|(t, _, _)| t)
}

/// [`ray_triangle_m`] with the crossing's barycentric weights `(u, v)` of `p1` and `p2` (`p0`
/// weighs `1 − u − v`), for the shade the parent draws at the crossing.
#[must_use]
pub fn ray_triangle(dir: DVec3, p0: DVec3, p1: DVec3, p2: DVec3) -> Option<(f64, f64, f64)> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < RAY_EPSILON {
        return None;
    }
    let inv = 1.0 / det;
    // The ray starts at the centre: `s = origin − p0 = −p0`.
    let s = -p0;
    let u = s.dot(h) * inv;
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    let t = e2.dot(q) * inv;
    let inside = (u >= -RAY_SLACK) & (v >= -RAY_SLACK) & (u + v <= 1.0 + RAY_SLACK) & (t > 0.0);
    inside.then_some((t, u, v))
}

/// A determinant under this is a ray parallel to the triangle.
const RAY_EPSILON: f64 = 1e-18;
/// How far past an edge a crossing still counts, in the triangle's own barycentric units: the
/// rounding of a radial that runs exactly along a shared edge, and — since step 5 narrows a
/// parent's positions to `f32` from its origin — a corner moved by that narrowing (8 mm on the
/// coarsest parent's 2 km triangles, 4e-6 of the triangle; 15 µm on 4 m at rung 2, the same
/// share). MEASURED without this: a ray through a parent's own vertex missed every triangle
/// that met there.
const RAY_SLACK: f64 = 1e-5;

/// THE PARENT CACHE: the parent meshes the lane's workers build, shared and bounded — the eight
/// finer chunks under one parent, and the halo users beside them, read one build. A map with a
/// use order (a counter per entry, so a hit is a lookup and never a scan, SL9); the least
/// recently used leaves when the bound is passed. Its content is a pure function of the seed, so
/// which worker builds it, and when, changes nothing. A reader CLAIMS a build in flight: the next
/// reader of the same parent waits for it instead of repeating it (ruling V15: with siblings
/// ordered back to back, four workers miss one parent at once). A claim is a guard: a build that
/// panics drops it, and the drop releases the waiters (refutation T-5). A realm's `forget` bumps
/// the realm's epoch, so a build that lands after it keeps nothing (refutation T-6).
#[derive(Debug)]
pub struct ParentCache {
    store: Mutex<ParentStore>,
    /// Woken when a claimed build lands or is abandoned, for the readers that wait on it.
    landed: std::sync::Condvar,
}

impl Default for ParentCache {
    fn default() -> ParentCache {
        ParentCache::with_capacity(PARENT_CACHE_ENTRIES)
    }
}

#[derive(Debug)]
struct Entry {
    mesh: Arc<ParentMesh>,
    /// The use counter's value at the last read.
    used: u64,
}

#[derive(Debug, Default)]
struct ParentStore {
    map: BTreeMap<(RealmId, ChunkKey), Entry>,
    /// The entries by their last use: the first is the least recently used.
    by_use: BTreeMap<u64, (RealmId, ChunkKey)>,
    use_seq: u64,
    /// The parents a reader has claimed and builds now.
    building: BTreeSet<(RealmId, ChunkKey)>,
    /// A realm's epoch: `forget` bumps it, and a claim from before keeps nothing.
    epochs: BTreeMap<RealmId, u64>,
    /// How many meshes the cache keeps (the library's own bound; the engine lifts it and sets
    /// the byte budget instead).
    capacity: usize,
    /// THE BYTE BUDGET (refutation of step 5's second half, finding 2): the meshes' bytes the
    /// cache keeps, a true bound — a count of entries at an estimated size was not one.
    budget_bytes: usize,
    /// Under the byte budget the cache never trims below this many meshes: one working set of
    /// the workers, so a budget too small for the parents in flight stalls nothing.
    floor: usize,
    /// The bytes of the meshes held now (`ParentMesh::bytes`, summed on keep and trim).
    held_bytes: usize,
    /// The counts (M8-2a): claims served from the map, claims that built, claims that waited on
    /// a build in flight (a wait counts once per claim, however many wakes it took).
    stats: ParentStats,
}

/// What the parent cache did since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParentStats {
    pub hits: u64,
    pub builds: u64,
    pub waits: u64,
}

/// How many parent meshes the cache keeps by default (the library's tests and a bare lane). The
/// engine sizes it from its memory budget (`ParentCache::set_capacity`).
pub const PARENT_CACHE_ENTRIES: usize = 48;

/// What a claim found: the mesh (a hit), or the guard of the right to build it (a miss, claimed).
#[derive(Debug)]
pub enum Claim<'a> {
    Hit(Arc<ParentMesh>),
    Build(ClaimGuard<'a>),
}

/// THE RIGHT TO BUILD one parent, held until `finish` (the mesh, or `None` to abandon). A guard
/// dropped any other way — a panic in the build — abandons the claim and wakes the waiters.
#[derive(Debug)]
pub struct ClaimGuard<'a> {
    cache: &'a ParentCache,
    realm: RealmId,
    key: ChunkKey,
    epoch: u64,
    done: bool,
}

impl ClaimGuard<'_> {
    /// End the claim: keep the mesh (or abandon with `None`) and wake the readers waiting on it.
    pub fn finish(mut self, mesh: Option<Arc<ParentMesh>>) {
        self.done = true;
        self.cache.land(self.realm, self.key, self.epoch, mesh);
    }
}

impl Drop for ClaimGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.cache.land(self.realm, self.key, self.epoch, None);
        }
    }
}

impl ParentCache {
    /// A cache that keeps `capacity` meshes (at least one).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> ParentCache {
        ParentCache {
            store: Mutex::new(ParentStore {
                capacity: capacity.max(1),
                budget_bytes: usize::MAX,
                floor: 1,
                ..ParentStore::default()
            }),
            landed: std::sync::Condvar::new(),
        }
    }

    /// Resize: the oldest leave when the new bound is smaller than what is held.
    pub fn set_capacity(&self, capacity: usize) {
        let mut store = self.lock();
        store.capacity = capacity.max(1);
        trim(&mut store);
    }

    /// How many meshes the cache keeps.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.lock().capacity
    }

    /// BOUND THE CACHE BY BYTES: keep meshes while their bytes fit `bytes`, never fewer than
    /// `floor` of them (at least one), and lift the count bound out of the way. The oldest leave
    /// at once when what is held passes the new budget.
    pub fn set_budget_bytes(&self, bytes: usize, floor: usize) {
        let mut store = self.lock();
        store.capacity = usize::MAX;
        store.budget_bytes = bytes;
        store.floor = floor.max(1);
        trim(&mut store);
    }

    /// The byte budget (unbounded until `set_budget_bytes`).
    #[must_use]
    pub fn budget_bytes(&self) -> usize {
        self.lock().budget_bytes
    }

    /// The bytes of the meshes held now.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.lock().held_bytes
    }

    /// The parent mesh of `key` in `realm`: the cached one, a build in flight waited for, or a
    /// fresh build kept for the next reader. `None` for a key outside the body.
    #[must_use]
    pub fn get(
        &self,
        realm: RealmId,
        body: &BodyDefinition,
        key: ChunkKey,
    ) -> Option<Arc<ParentMesh>> {
        match self.claim(realm, key) {
            Claim::Hit(found) => Some(found),
            Claim::Build(guard) => {
                let built = ParentMesh::build(body, key).map(Arc::new);
                guard.finish(built.clone());
                built
            }
        }
    }

    /// Claim `key`: a hit returns at once (and refreshes its use); a miss another reader builds
    /// WAITS, then reads a hit; a miss nobody builds is claimed, and the guard must `finish`.
    pub fn claim(&self, realm: RealmId, key: ChunkKey) -> Claim<'_> {
        let mut store = self.lock();
        let mut waited = false;
        loop {
            if store.map.contains_key(&(realm, key)) {
                let found = touch(&mut store, realm, key);
                store.stats.hits += 1;
                store.stats.waits += u64::from(waited);
                return Claim::Hit(found);
            }
            if !store.building.contains(&(realm, key)) {
                store.building.insert((realm, key));
                store.stats.builds += 1;
                store.stats.waits += u64::from(waited);
                let epoch = store.epochs.get(&realm).copied().unwrap_or(0);
                return Claim::Build(ClaimGuard {
                    cache: self,
                    realm,
                    key,
                    epoch,
                    done: false,
                });
            }
            waited = true;
            store = self
                .landed
                .wait(store)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The counts since the start.
    #[must_use]
    pub fn stats(&self) -> ParentStats {
        self.lock().stats
    }

    /// A claim lands: the mesh is kept when the realm's epoch is the claim's, and the waiters
    /// wake either way.
    fn land(&self, realm: RealmId, key: ChunkKey, epoch: u64, mesh: Option<Arc<ParentMesh>>) {
        {
            let mut store = self.lock();
            store.building.remove(&(realm, key));
            let current = store.epochs.get(&realm).copied().unwrap_or(0);
            if let Some(mesh) = mesh.filter(|_| current == epoch) {
                keep(&mut store, realm, key, mesh);
            }
        }
        self.landed.notify_all();
    }

    /// Keep a mesh; the oldest leaves when the bound is passed.
    pub fn insert(&self, realm: RealmId, key: ChunkKey, mesh: Arc<ParentMesh>) {
        let mut store = self.lock();
        keep(&mut store, realm, key, mesh);
    }

    /// Drop every mesh of a realm and refuse the builds in flight for it: its body may be stated
    /// anew with another seed. A walk of the store — a forget is a realm leaving the window, not
    /// a per-frame event.
    pub fn forget(&self, realm: RealmId) {
        {
            let mut store = self.lock();
            store.map.retain(|(r, _), _| *r != realm);
            store.by_use.retain(|_, (r, _)| *r != realm);
            store.building.retain(|(r, _)| *r != realm);
            store.held_bytes = store.map.values().map(|e| e.mesh.bytes()).sum();
            *store.epochs.entry(realm).or_insert(0) += 1;
        }
        self.landed.notify_all();
    }

    /// How many meshes the cache holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    /// Whether the cache holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ParentStore> {
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Refresh a held entry's use and hand out its mesh: two map operations, never a scan.
fn touch(store: &mut ParentStore, realm: RealmId, key: ChunkKey) -> Arc<ParentMesh> {
    store.use_seq += 1;
    let now = store.use_seq;
    let entry = store.map.get_mut(&(realm, key)).expect("held");
    let before = entry.used;
    entry.used = now;
    let mesh = Arc::clone(&entry.mesh);
    store.by_use.remove(&before);
    store.by_use.insert(now, (realm, key));
    mesh
}

/// Keep a mesh in the store (a held key is replaced and refreshed) and trim to the bound.
fn keep(store: &mut ParentStore, realm: RealmId, key: ChunkKey, mesh: Arc<ParentMesh>) {
    store.use_seq += 1;
    let now = store.use_seq;
    store.held_bytes += mesh.bytes();
    if let Some(old) = store.map.insert((realm, key), Entry { mesh, used: now }) {
        store.by_use.remove(&old.used);
        store.held_bytes -= old.mesh.bytes();
    }
    store.by_use.insert(now, (realm, key));
    trim(store);
}

/// The least recently used leave until the store holds its capacity AND its bytes fit the
/// budget — the byte bound stops at the floor, so a working set always stays.
fn trim(store: &mut ParentStore) {
    while store.map.len() > store.capacity || over_budget(store) {
        let (_, key) = store.by_use.pop_first().expect("as many uses as entries");
        let gone = store.map.remove(&key).expect("held");
        store.held_bytes -= gone.mesh.bytes();
    }
}

/// Whether the held bytes pass the budget with more than the floor held.
fn over_budget(store: &ParentStore) -> bool {
    store.held_bytes > store.budget_bytes && store.map.len() > store.floor
}

/// THE COARSE KEY (the shadow ladder, D8-8's shadow cost): the chunk `steps` rungs up that holds
/// this chunk's volume — each axis halved a step at a time, as `parent_keys` halves one step.
/// `None` past the body's top rung or off the coarser rung's grid.
#[must_use]
pub fn coarse_key(body: &BodyDefinition, key: ChunkKey, steps: u8) -> Option<ChunkKey> {
    let rung = key.rung.checked_add(steps)?;
    if rung >= body.ladder().rungs {
        return None;
    }
    let edge = CHUNK_EDGE as i32;
    let last = (body.ladder().cells_per_edge(rung) as i32 - 1) / edge;
    let top = vd_terrain::digest::top_chunk_z(body, rung);
    let div = 1i32 << steps;
    let (x, y, z) = (
        key.x.div_euclid(div),
        key.y.div_euclid(div),
        key.z.div_euclid(div),
    );
    let inside = (x >= 0) & (x <= last) & (y >= 0) & (y <= last) & (z >= 0) & (z <= top);
    inside.then_some(ChunkKey {
        face: key.face,
        rung,
        x,
        y,
        z,
    })
}

/// THE PARENTS a chunk's geomorph reads: none at the top rung; else the chunk at the next
/// coarser rung that holds this chunk's footprint, and its neighbours on every side this chunk
/// FACES — across the face, the side its halo lies on (a lower half's halo stands over the
/// parent's own halo column, whose radial crossings no box can build: the groups past the box
/// do not exist, so that column's surface is the lateral neighbour's, which holds it as its own
/// last column; MEASURED before this, a halo vertex found only a cave 33 m down); radially, the
/// side it faces (a lower half looks down, an upper half up), because the coarser surface
/// stands within its gap bound of the finer one and may cross into the next chunk. Up to eight
/// keys, only those inside the face and the band; a halo across a face seam reads the field.
#[must_use]
pub fn parent_keys(body: &BodyDefinition, key: ChunkKey) -> Vec<ChunkKey> {
    let rungs = body.ladder().rungs;
    if key.rung + 1 >= rungs {
        return Vec::new();
    }
    let rung = key.rung + 1;
    let edge = CHUNK_EDGE as i32;
    let last = (body.ladder().cells_per_edge(rung) as i32 - 1) / edge;
    let top = vd_terrain::digest::top_chunk_z(body, rung);
    let side = |i: i32| -> i32 { if i.rem_euclid(2) == 0 { -1 } else { 1 } };
    let (px, py, pz) = (
        key.x.div_euclid(2),
        key.y.div_euclid(2),
        key.z.div_euclid(2),
    );
    let mut out = Vec::new();
    for dx in [0, side(key.x)] {
        for dy in [0, side(key.y)] {
            for dz in [0, side(key.z)] {
                let (x, y, z) = (px + dx, py + dy, pz + dz);
                let inside =
                    (x >= 0) & (x <= last) & (y >= 0) & (y <= last) & (z >= 0) & (z <= top);
                if inside {
                    out.push(ChunkKey {
                        face: key.face,
                        rung,
                        x,
                        y,
                        z,
                    });
                }
            }
        }
    }
    out
}

/// The parent's lattice cell `(a, b)` under a finer vertex: the finer chunk's face cell in
/// quanta, halved (two finer cells to a coarser one), less the parent box's origin — exact in
/// integers, so a shared vertex reads the same cell from both its chunks.
fn parent_cell(key: ChunkKey, parent: ChunkKey, v: [i16; 3]) -> (i32, i32) {
    let edge = i64::from(CHUNK_EDGE as i32);
    let q = i64::from(vd_terrain::VERTEX_QUANTUM as i16);
    let cell = |fine: i32, coarse: i32, quanta: i16| -> i32 {
        let global = i64::from(fine) * edge * q + i64::from(quanta);
        let local = global - i64::from(coarse) * edge * q * 2;
        local.div_euclid(q * 2) as i32
    };
    (cell(key.x, parent.x, v[0]), cell(key.y, parent.y, v[1]))
}

/// Run the recipe and the extractor for one chunk and turn the result into engine floats; `None`
/// for a key outside the body (the lane never submits one, so this is the seam's own refusal).
/// The geomorph's targets come from a fresh parent cache: the lane's workers share one through
/// [`geometry_with`].
#[must_use]
pub fn geometry_of(body: &BodyDefinition, key: ChunkKey) -> Option<ChunkGeometry> {
    // A fresh, private cache: the realm named in it is nobody's.
    geometry_with(body, RealmId::Planet(0), key, &ParentCache::default())
}

/// [`geometry_of`] with the lane's parent cache, for the chunk of `realm`.
///
/// A chunk's build is TWO STEPS, and this is the pair run together: the SAMPLE step
/// ([`vd_terrain::lattice::sample_box`]) reads the recipe into a box of cells, and the GEOMETRY
/// step ([`geometry_from`]) turns that box into a mesh. They are named apart because a second
/// builder may do the sample step elsewhere — the card computes the same box byte for byte
/// (`vd_terrain::gpu`) — and hand the box to the same geometry step (ruling F9 item 2).
#[must_use]
pub fn geometry_with(
    body: &BodyDefinition,
    realm: RealmId,
    key: ChunkKey,
    parents: &ParentCache,
) -> Option<ChunkGeometry> {
    let samples = sample_box(body, key)?;
    geometry_from(body, realm, key, &samples, parents)
}

/// THE GEOMETRY STEP of a chunk's build: the mesh, the vertices, the morph targets and the normals
/// from a box of cells somebody already sampled. The box is the recipe's own output, whoever
/// computed it.
#[must_use]
pub fn geometry_from(
    body: &BodyDefinition,
    realm: RealmId,
    key: ChunkKey,
    samples: &vd_terrain::lattice::SampleBox,
    parents: &ParentCache,
) -> Option<ChunkGeometry> {
    let mesh = extract(samples);
    let half = (CHUNK_EDGE / 2) as i16 * vd_terrain::VERTEX_QUANTUM as i16;
    let origin = vertex_position_m(body, samples, [half, half, half]);
    let origin_m = [origin[0], origin[1], origin[2]];
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(mesh.vertices.len());
    let mut morph: Vec<f32> = Vec::with_capacity(mesh.vertices.len());
    let mut radials: Vec<[f32; 3]> = Vec::with_capacity(mesh.vertices.len());
    // The parent's shade at each vertex's target, where a parent triangle was hit (item 20).
    let mut shades: Vec<Option<DVec3>> = Vec::with_capacity(mesh.vertices.len());
    let sink_len = sink_m(body, key.rung);
    // The coarser rung's surface along each vertex's radial: where the radial meets a parent
    // mesh, nearest the vertex's own radius; the coarser FIELD where no parent triangle is on
    // it (a cave's own vertex, or a coarser surface past both parents); the top rung morphs to
    // itself.
    let coarser = key
        .rung
        .saturating_add(1)
        .min(body.ladder().rungs.saturating_sub(1));
    let parent_meshes: Vec<Arc<ParentMesh>> = parent_keys(body, key)
        .into_iter()
        .filter_map(|p| parents.get(realm, body, p))
        .collect();
    // How far the coarser surface can stand from this vertex: the sink of the coarser rung.
    let reach_m = sink_m(body, coarser);
    let mut fallbacks = 0u32;
    let mut seam = 0u32;
    // THE FACE SEAM: a vertex on or past the face's edge (the halo across a seam, or the shared
    // edge itself) reads the coarser FIELD, from both faces alike — the parents are this face's
    // own, and the two faces would read two meshes for one vertex.
    let face_quanta = i64::from(body.ladder().cells_per_edge(key.rung))
        * i64::from(CHUNK_EDGE as i32)
        * i64::from(vd_terrain::VERTEX_QUANTUM as i16);
    let on_seam = |v: [i16; 3]| -> bool {
        let ga = i64::from(key.x)
            * i64::from(CHUNK_EDGE as i32)
            * i64::from(vd_terrain::VERTEX_QUANTUM as i16)
            + i64::from(v[0]);
        let gb = i64::from(key.y)
            * i64::from(CHUNK_EDGE as i32)
            * i64::from(vd_terrain::VERTEX_QUANTUM as i16)
            + i64::from(v[1]);
        (ga <= 0) | (ga >= face_quanta) | (gb <= 0) | (gb >= face_quanta)
    };
    for v in &mesh.vertices {
        let p = vertex_position_m(body, samples, *v);
        vertices.push([
            (p[0] - origin[0]) as f32,
            (p[1] - origin[1]) as f32,
            (p[2] - origin[2]) as f32,
        ]);
        let abs = vd_core::glam::DVec3::new(p[0], p[1], p[2]);
        let len = abs.length();
        let dir = abs / len;
        let target_m = if coarser == key.rung {
            len
        } else {
            let at_seam = on_seam(*v);
            seam += u32::from(at_seam);
            let on_mesh = parent_meshes
                .iter()
                .filter(|_| !at_seam)
                .fold(None, |best: Option<(f64, DVec3)>, pm| {
                    let (a, b) = parent_cell(key, pm.key(), *v);
                    let hit = pm.radial_hit(a, b, dir, len);
                    match (best, hit) {
                        (Some((b, _)), Some((h, s)))
                            if ((h - len).abs(), h) < ((b - len).abs(), b) =>
                        {
                            Some((h, s))
                        }
                        (None, Some(h)) => Some(h),
                        (b, _) => b,
                    }
                })
                // A hit farther than the two surfaces can stand apart is another surface (a
                // cave under this one): the field, then.
                .filter(|(h, _)| (h - len).abs() <= reach_m);
            shades.push(on_mesh.map(|(_, s)| s));
            let on_mesh = on_mesh.map(|(h, _)| h);
            // The coarser field, read only where no hit stands: the target then, and the
            // judge of what the vertex is — a SURFACE vertex with no parent triangle on its
            // radial is a fallback (counted; the gate bounds it), a vertex far under the field
            // is a cave's own and reads the field by nature.
            let field = on_mesh.map_or_else(
                || vd_terrain::height::height_m(body, [dir.x, dir.y, dir.z], coarser),
                |_| len,
            );
            let missing = on_mesh.is_none() & !at_seam;
            fallbacks += u32::from(missing & ((field - len).abs() <= reach_m));
            on_mesh.unwrap_or(field)
        };
        // The target as one metre along the radial from the vertex, and the radial itself (step 5).
        morph.push((target_m - len) as f32);
        radials.push([dir.x as f32, dir.y as f32, dir.z as f32]);
    }
    let normals = smooth_normals(&vertices, &mesh.triangles);
    let packed_normals: Vec<[i16; 2]> = normals.iter().map(|n| oct_encode(*n)).collect();
    // The top rung morphs to itself and pushes no shade; every other rung one per vertex.
    let morph_normals: Vec<[i16; 2]> = if shades.is_empty() {
        packed_normals.clone()
    } else {
        shades
            .iter()
            .zip(packed_normals.iter())
            .map(|(shade, own)| {
                shade.map_or(*own, |s| oct_encode([s.x as f32, s.y as f32, s.z as f32]))
            })
            .collect()
    };
    let mut geometry = ChunkGeometry {
        key,
        origin_m,
        morph_m: morph,
        morph_fallbacks: fallbacks,
        morph_seam: seam,
        sink_m: sink_len as f32,
        vertices,
        normals,
        packed_normals,
        morph_normals,
        triangles: mesh.triangles,
        radials,
        bounds: ([0.0; 3], [0.0; 3]),
        skirt_start: 0,
    };
    let drop_m = f64::from(SKIRT_CELLS) * f64::from(vd_seed::ladder::cell_m(key.rung));
    add_skirts(&mut geometry, drop_m);
    geometry.bounds = geometry.measure_bounds();
    Some(geometry)
}

/// How deep a skirt hangs under a chunk's edge, in cells of its rung.
pub const SKIRT_CELLS: u32 = 2;

/// THE SKIRTS: a strip hanging `drop_m` radially under every boundary edge of the chunk's mesh —
/// an edge one triangle alone uses, which is where a neighbour's quads meet this chunk's. Two
/// neighbours state their shared vertices from one set of quanta, but the engine adds each
/// chunk's own origin to its own single-precision offsets, and the two sums differ by a rounding
/// (a tenth of a millimetre at a kilometre): a hairline crack no eye can see, until a pixel's
/// centre falls into it. MEASURED on the hill stand: one pixel of nothing at 970 m, where four
/// rung-1 chunks meet, the same pixel on two flights. A skirt faces outward (away from its
/// triangle's third corner), keeps its edge's normals so the light is continuous, and morphs and
/// sinks with its edge (its bottom's targets are its top's, dropped the same), so the crack is
/// covered at every distance. The rounding is the engine's own floating origin at work; the skirt
/// is the standard cure, two cells deep, under the surface everywhere but in the crack.
fn add_skirts(g: &mut ChunkGeometry, drop_m: f64) {
    g.skirt_start = g.vertices.len() as u32;
    // Boundary edges: each edge as an ordered pair (lower index first) with its use count, the
    // triangle's third corner, and the edge's own direction in that triangle.
    let mut uses: BTreeMap<(u32, u32), (u32, u32, bool)> = BTreeMap::new();
    for t in &g.triangles {
        let mut i = 0;
        while i < 3 {
            let (a, b, c) = (t[i], t[(i + 1) % 3], t[(i + 2) % 3]);
            let key = (a.min(b), a.max(b));
            let e = uses.entry(key).or_insert((0, c, a < b));
            e.0 += 1;
            i += 1;
        }
    }
    let o = DVec3::from_array(g.origin_m);
    let at =
        |v: [f32; 3]| -> DVec3 { DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])) };
    let mut skirt_tris: Vec<[u32; 3]> = Vec::new();
    for ((lo, hi), (count, c, forward)) in uses {
        if count != 1 {
            continue;
        }
        // The edge as its triangle winds it: a → b.
        let (a, b) = if forward { (lo, hi) } else { (hi, lo) };
        let (pa, pb, pc) = (
            at(g.vertices[a as usize]),
            at(g.vertices[b as usize]),
            at(g.vertices[c as usize]),
        );
        // A skirt vertex is its top dropped along the radial; its target drops by the same, so
        // the metre along the radial from the vertex to the target is its top's (step 5).
        let bottom = |i: u32| -> ([f32; 3], f32) {
            let p = at(g.vertices[i as usize]);
            let dir = (o + p).normalize();
            let q = p - dir * drop_m;
            ([q.x as f32, q.y as f32, q.z as f32], g.morph_m[i as usize])
        };
        let (qa, ma) = bottom(a);
        let (qb, mb) = bottom(b);
        let a2 = g.vertices.len() as u32;
        let b2 = a2 + 1;
        g.vertices.push(qa);
        g.vertices.push(qb);
        g.morph_m.push(ma);
        g.morph_m.push(mb);
        g.radials.push(g.radials[a as usize]);
        g.radials.push(g.radials[b as usize]);
        g.normals.push(g.normals[a as usize]);
        g.normals.push(g.normals[b as usize]);
        g.packed_normals.push(g.packed_normals[a as usize]);
        g.packed_normals.push(g.packed_normals[b as usize]);
        g.morph_normals.push(g.morph_normals[a as usize]);
        g.morph_normals.push(g.morph_normals[b as usize]);
        // The strip faces away from the triangle's third corner: the winding whose normal points
        // from the edge's middle away from that corner.
        let n = (pb - pa).cross(at(qb) - pa);
        let away = (pa + pb) * 0.5 - pc;
        let outward = n.dot(away) >= 0.0;
        if outward {
            skirt_tris.push([a, b, b2]);
            skirt_tris.push([a, b2, a2]);
        } else {
            skirt_tris.push([a, b2, b]);
            skirt_tris.push([a, a2, b2]);
        }
    }
    g.triangles.extend(skirt_tris);
}

/// THE PACKED NORMAL (step 5's second half, ruling V18): a unit normal as two signed 16-bit
/// numbers on the octahedron — the unit sphere folded onto a square, so two coordinates name every
/// direction (Cigolle et al., "A Survey of Efficient Representations for Independent Unit
/// Vectors"). The widest angle between a normal and its unpacked form is under
/// `OCT_NORMAL_STEP_RAD`. Four bytes a vertex against twelve; the shader unfolds it.
///
/// **Example.** The hill's crease at (641, 300) has a normal 31° off the radial; packed and
/// unpacked it is 31° off by less than a hundredth of a degree, and the light on it moves by a
/// fraction of one level of 255.
#[must_use]
pub fn oct_encode(n: [f32; 3]) -> [i16; 2] {
    let sum = n[0].abs() + n[1].abs() + n[2].abs();
    // A zero normal (a vertex no triangle uses) folds to the square's centre: straight up.
    let (mut x, mut y) = if sum > 0.0 {
        (n[0] / sum, n[1] / sum)
    } else {
        (0.0, 0.0)
    };
    if n[2] < 0.0 {
        let (fx, fy) = ((1.0 - y.abs()) * sign_of(x), (1.0 - x.abs()) * sign_of(y));
        x = fx;
        y = fy;
    }
    // THE PRECISE ROUNDING (Cigolle et al., "oct16P"): the nearest of the four quanta around the
    // folded point is the one whose unfolded normal lies closest to the true one — a rounding of
    // the coordinates alone misses it by up to a cell, because the fold is not a scaling.
    // MEASURED: the plain rounding left one limb pixel of the orbit stand three levels off (the
    // lighting there divides by a near-zero view angle, so it reads the normal's error a
    // hundredfold); the precise one halves the widest angle.
    let (qx, qy) = (x * f32::from(i16::MAX), y * f32::from(i16::MAX));
    let mut best = [snorm16(x), snorm16(y)];
    let mut best_err = f64::MAX;
    for cx in [qx.floor(), qx.ceil()] {
        for cy in [qy.floor(), qy.ceil()] {
            let candidate = [clamp_i16(cx), clamp_i16(cy)];
            if sine_error(n, oct_decode(candidate)) < best_err {
                best_err = sine_error(n, oct_decode(candidate));
                best = candidate;
            }
        }
    }
    best
}

/// The squared sine of the angle between two near-unit vectors, from the cross product in
/// 64-bit: linear in the angle, so it tells two candidates a ten-thousandth of a radian apart.
/// (A dot product near one cannot: the 32-bit components' rounding hides half a milliradian.)
/// Every candidate lies within a quantum of the true normal, so no far-side guard is needed
/// (refutation N-1: a guard no input reaches is a red coverage gate).
fn sine_error(a: [f32; 3], b: [f32; 3]) -> f64 {
    let a = DVec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]));
    let b = DVec3::new(f64::from(b[0]), f64::from(b[1]), f64::from(b[2]));
    let c = a.cross(b);
    c.dot(c)
}

/// A quantum already rounded, clamped to the signed 16-bit range.
fn clamp_i16(q: f32) -> i16 {
    q.clamp(f32::from(-i16::MAX), f32::from(i16::MAX)) as i16
}

/// The unit normal a packed pair names: the shader's own steps, for the tests and the bound.
#[must_use]
pub fn oct_decode(e: [i16; 2]) -> [f32; 3] {
    // The GPU's own rule for a signed 16-bit quantum: divided by 32 767, and −32 768 reads −1.
    let x = (f32::from(e[0]) / f32::from(i16::MAX)).max(-1.0);
    let y = (f32::from(e[1]) / f32::from(i16::MAX)).max(-1.0);
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).clamp(0.0, 1.0);
    let ux = x + if x >= 0.0 { -t } else { t };
    let uy = y + if y >= 0.0 { -t } else { t };
    let len = (ux * ux + uy * uy + z * z).sqrt();
    [ux / len, uy / len, z / len]
}

/// The widest angle between a unit normal and its packed form, in radians (MEASURED in the
/// tests over a real chunk's normals: about 5e-5; the 16-bit square's cell is 2 / 65 534).
pub const OCT_NORMAL_STEP_RAD: f64 = 1.0e-4;

/// `+1` for a non-negative number, `−1` otherwise (the octahedron's fold keeps a zero's side).
fn sign_of(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// A number in `[−1, 1]` as a signed 16-bit quantum, rounded to nearest.
fn snorm16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
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
    /// ★ A surface statement with NO CHARTER (slice 8b stage 1): the realm states it is seed-shaped
    /// and states none of its physical facts, so the lane refuses THIS realm's ground and counts it.
    /// One realm's ground is missing; the world stands.
    pub no_charter: u64,
    /// Surface statements accepted WITH a charter in hand — the presence counter beside the
    /// refusal, so "nobody ships a charter" and "everybody does" are two different readings.
    pub charters_held: u64,
    /// A request for a key outside the body's ladder.
    pub outside: u64,
    /// Jobs submitted to the workers.
    pub submitted: u64,
    /// Chunks harvested by the engine.
    pub harvested: u64,
    /// Polls that returned the whole cap (M8-2a): a harvest that fills its cap every frame is the
    /// wall, not the workers.
    pub harvest_full: u64,
}

/// The lane: the bodies it knows, the chunks it holds, the workers it drives.
pub struct ChunkLane {
    workers: Box<dyn ChunkWorkers>,
    /// This client's declared recipe tag: a realm's surface must state the same one.
    declared: u64,
    bodies: BTreeMap<RealmId, Arc<BodyDefinition>>,
    /// The charter each body's realm stated, held beside the body. Nothing reads it yet: stage 3 is
    /// where the relief law takes it, and the record is carried now because appending a word after
    /// the freeze is impossible.
    charters: BTreeMap<RealmId, vd_core::look::BodyCharter>,
    /// Realms whose surface statement was refused: counted ONCE, never re-read each frame.
    refused: BTreeSet<RealmId>,
    resident: BTreeSet<(RealmId, ChunkKey)>,
    pending: BTreeSet<(RealmId, ChunkKey)>,
    /// Resident plus pending chunks per realm, so a release is a lookup, not a scan (SL9).
    held: BTreeMap<RealmId, usize>,
    counters: ChunkCounters,
    /// The parent meshes the workers share.
    parents: Arc<ParentCache>,
}

impl ChunkLane {
    /// A lane over `workers`, for a client whose declared recipe tag is `declared`.
    #[must_use]
    pub fn new(workers: Box<dyn ChunkWorkers>, declared: u64) -> ChunkLane {
        ChunkLane {
            workers,
            declared,
            bodies: BTreeMap::new(),
            charters: BTreeMap::new(),
            refused: BTreeSet::new(),
            resident: BTreeSet::new(),
            pending: BTreeSet::new(),
            held: BTreeMap::new(),
            counters: ChunkCounters::default(),
            parents: Arc::new(ParentCache::default()),
        }
    }

    /// The counters.
    #[must_use]
    pub fn counters(&self) -> ChunkCounters {
        self.counters
    }

    /// What the workers have built (M8-2a).
    #[must_use]
    pub fn built(&self) -> BuildCount {
        self.workers.built()
    }

    /// What the parent cache did (M8-2a).
    #[must_use]
    pub fn parent_stats(&self) -> ParentStats {
        self.parents.stats()
    }

    /// How many chunks are still building: the instrument a picture gate waits on for "the whole
    /// wanted set landed", beside `terrain_chunks_drawn` for "the first one did".
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// State a realm's surface: its statement, its CHARTER and its look. The body is built from the
    /// seed in the statement's frame and the look shell's radius. A statement with a foreign recipe
    /// tag, NO CHARTER, a frame without a seed, or a radius the ladder refuses is counted and the
    /// realm gets no body. Stating the same surface twice keeps the body; a refused realm is counted
    /// once and then ignored until it is forgotten (a refusal used to be re-counted and the body
    /// re-derived on every frame — the refuter's finding).
    ///
    /// ★ THE CHARTER IS REQUIRED (slice 8b §4.1/§4.2 rule 3). A shard that cannot derive its charter
    /// states no surface, so in the shipped path the two always arrive together. A surface that
    /// arrives alone is a fault, and the lane says so with a counter instead of inventing a gravity.
    /// The charter is HELD, not yet read: stage 3 is where the relief law takes it.
    pub fn state_surface(
        &mut self,
        realm: RealmId,
        surface: &SurfaceStmt,
        charter: Option<&vd_core::look::BodyCharter>,
        look: &Boundary,
    ) {
        if self.bodies.contains_key(&realm) | self.refused.contains(&realm) {
            return;
        }
        if surface.generator != self.declared {
            self.counters.foreign_generator += 1;
            self.refused.insert(realm);
            return;
        }
        let Some(charter) = charter else {
            self.counters.no_charter += 1;
            self.refused.insert(realm);
            return;
        };
        // ★ THE CLIENT BUILDS THE BODY FROM THE SAME INTEGERS THE SHARD USED (slice 8b stage 3).
        // The two words come off the realm's own charter, not off anything the client derives, so
        // the client's chunks equal the shard's byte for byte and the no-drift gate stays a
        // measurement that could fail.
        let facts = vd_terrain::BodyFacts::new(charter.gravity_mm_s2, charter.bulk_density_kgm3);
        let body = match (surface.frame, look) {
            (FrameRef::PlanetCentered { planet_seed }, Boundary::Shell { r }) => {
                BodyDefinition::from_seed(planet_seed, *r, facts)
            }
            _ => None,
        };
        match body {
            Some(body) => {
                self.bodies.insert(realm, Arc::new(body));
                self.charters.insert(realm, *charter);
                self.counters.charters_held += 1;
            }
            None => {
                self.counters.no_body += 1;
                self.refused.insert(realm);
            }
        }
    }

    /// The charter the lane holds for a realm — what the realm stated about itself, in whole
    /// numbers. Nothing in the shipped path reads it yet (stage 3 is where the relief law does);
    /// the dev state reads it so a flight can prove a client received one.
    #[must_use]
    pub fn charter(&self, realm: RealmId) -> Option<&vd_core::look::BodyCharter> {
        self.charters.get(&realm)
    }

    /// The body the lane holds for a realm.
    #[must_use]
    pub fn body(&self, realm: RealmId) -> Option<&Arc<BodyDefinition>> {
        self.bodies.get(&realm)
    }

    /// Forget a realm: its body and every chunk of it.
    pub fn forget(&mut self, realm: RealmId) {
        self.bodies.remove(&realm);
        self.charters.remove(&realm);
        self.refused.remove(&realm);
        self.held.remove(&realm);
        self.parents.forget(realm);
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

    /// Whether a chunk has ARRIVED (harvested, resident) — the release hold's question (step 2): a
    /// chunk still building is held, not arrived.
    #[must_use]
    pub fn is_resident(&self, realm: RealmId, key: ChunkKey) -> bool {
        self.resident.contains(&(realm, key))
    }

    /// Whether the lane holds or is building a chunk.
    #[must_use]
    pub fn holds(&self, realm: RealmId, key: ChunkKey) -> bool {
        self.resident.contains(&(realm, key)) | self.pending.contains(&(realm, key))
    }

    /// The parent cache the workers share (the engine sizes it to its workers).
    #[must_use]
    pub fn parents(&self) -> &Arc<ParentCache> {
        &self.parents
    }

    /// Ask for a chunk, at `priority` (the lower builds first: the wanted set's own order). A realm
    /// without a body or a key outside its ladder is refused and counted; a chunk already held or
    /// building is not queued twice. Any rung, beside any other (step 2).
    pub fn request(&mut self, realm: RealmId, key: ChunkKey, priority: u32) {
        let Some(body) = self.bodies.get(&realm) else {
            self.counters.no_body += 1;
            return;
        };
        if !vd_terrain::lattice::in_ladder(body, key) {
            self.counters.outside += 1;
            return;
        }
        if self.resident.contains(&(realm, key)) {
            return;
        }
        if self.pending.contains(&(realm, key)) {
            self.workers.reprioritise(realm, key, priority);
            return;
        }
        self.pending.insert((realm, key));
        *self.held.entry(realm).or_insert(0) += 1;
        self.counters.submitted += 1;
        self.workers.submit(ChunkJob {
            realm,
            body: Arc::clone(body),
            key,
            parents: Arc::clone(&self.parents),
            priority,
        });
    }

    /// Harvest up to `max` finished chunks. A chunk released while it was building is dropped
    /// here, never handed out.
    pub fn poll(&mut self, max: usize) -> Vec<ChunkReady> {
        self.poll_within(max, u64::MAX)
    }

    /// THE HARVEST WITHIN A BYTE BUDGET (ruling V15 item 2): finished chunks one at a time until
    /// `max` of them or until their upload bytes reach `budget_bytes` — a frame of small far
    /// chunks takes more of them, a frame of large near ones fewer, so no frame hitches on a bad
    /// mix. The first chunk always comes, whatever its size. The harvest counts as FULL when
    /// either bound stopped it.
    pub fn poll_within(&mut self, max: usize, budget_bytes: u64) -> Vec<ChunkReady> {
        let mut out = Vec::new();
        let mut bytes = 0u64;
        while out.len() < max && bytes < budget_bytes {
            let before = out.len();
            self.workers.drain(&mut out, 1);
            if out.len() == before {
                break;
            }
            bytes += out[before].geometry.upload_bytes();
        }
        self.counters.harvest_full +=
            u64::from((max > 0) & ((out.len() == max) | (bytes >= budget_bytes)));
        out.retain(|ready| self.pending.remove(&(ready.realm, ready.geometry.key)));
        for ready in &out {
            self.resident.insert((ready.realm, ready.geometry.key));
        }
        self.counters.harvested += out.len() as u64;
        out
    }

    /// Drop a chunk: resident or still building (withdrawn from the workers). A lookup per realm,
    /// never a scan of every chunk held (SL9).
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
            }
        }
    }

    /// Every chunk still building, with its realm — what the engine walks to withdraw the ones no
    /// longer wanted (the refuter's finding: only DRAWN chunks were released, so a moving eye left
    /// every stale job in the workers' queue and `pending` never drained).
    #[must_use]
    pub fn pending_all(&self) -> Vec<(RealmId, ChunkKey)> {
        self.pending.iter().copied().collect()
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
    let dir = [point_m[0] / len, point_m[1] / len, point_m[2] / len];
    let surface = vd_terrain::height::height_m(body, dir, 0);
    Some(EyeSurface {
        surface_m: surface,
        altitude_m: len - surface,
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
/// THE MARCH'S OWN REACH, in cells of its rung: a ray that has met no ground within this many cells
/// plants no ball. The reach a caller states is the ladder's, hundreds of kilometres; at one metre
/// a cell that is a half-million-step march on the render thread for a ray that points at the sky
/// (the refuter's finding). Four thousand cells is 4 km on the ground and 17 000 km at the top
/// rung — every stand's ray meets ground well inside it, and a ray that does not gets no ruler.
pub const RULER_MAX_CELLS: f64 = 4096.0;

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
    let reach_m = reach_m.min(cell * RULER_MAX_CELLS);
    let below = |t: f64| -> bool {
        let p = eye + fwd * t;
        let len = p.length();
        let dir = [p.x / len, p.y / len, p.z / len];
        len <= vd_terrain::height::height_m(body, dir, rung)
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

    /// The home planet's charter as its own realm states it: the words the census draws, and
    /// ABSENCE for the words a later stage draws.
    ///
    /// ★ THE TWO RELIEF WORDS ARE THE HOME PLANET'S OWN (slice 8b stage 3), because the lane now
    /// BUILDS THE BODY FROM THEM. They were the design's hand figures (9 821 and 5 514) while
    /// nobody read them; the moment the relief law read them the fixture's body stopped being the
    /// home planet, which is the measurement that says the charter reaches the shape.
    fn charter() -> vd_core::look::BodyCharter {
        vd_core::look::BodyCharter {
            gravity_mm_s2: vd_terrain::home::HOME_PLANET_GRAVITY_MM_S2,
            bulk_density_kgm3: vd_terrain::home::HOME_PLANET_BULK_DENSITY_KGM3,
            escape_velocity_mps: 11_190,
            insolation_q12: 3_065,
            t_eq_mk: 236_795,
            t_surface_mk: None,
            bond_albedo_q12: 1_228,
            mu_q8: Some(7_168),
            scale_height_m: Some(7_161),
            p_surf_pa: None,
            tau_vis_q12: None,
            tau_ir_q12: None,
            day_s: None,
            obliquity_cos_q1024: None,
            water_km3: None,
            sea_offset_mm: None,
            elastic_thickness_m: None,
            ecc_q16: 1_130,
            year_s: 34_766_100,
            flags: vd_core::look::CHARTER_FLAG_HAS_AIR | vd_core::look::CHARTER_FLAG_SOLID_SURFACE,
        }
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
        lane.state_surface(planet(), &foreign, Some(&charter()), &look);
        lane.state_surface(planet(), &foreign, Some(&charter()), &look);
        assert_eq!(lane.counters().foreign_generator, 1);
        assert!(lane.body(planet()).is_none());
        lane.forget(planet());
        lane.state_surface(planet(), &foreign, Some(&charter()), &look);
        assert_eq!(lane.counters().foreign_generator, 2);
        // A frame without a seed: no body, counted once too.
        let seedless = SurfaceStmt {
            generator: declared,
            frame: FrameRef::SystemSpace { system_seed: 7 },
        };
        let system = RealmId::System(7);
        lane.state_surface(system, &seedless, Some(&charter()), &look);
        lane.state_surface(system, &seedless, Some(&charter()), &look);
        assert_eq!(lane.counters().no_body, 1);
    }

    #[test]
    fn a_surface_statement_becomes_the_home_planet_and_a_foreign_one_is_refused() {
        let mut lane = lane();
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
        assert_eq!(lane.body(planet()).map(|b| **b), Some(home_planet()));
        assert_eq!(
            lane.charter(planet()),
            Some(&charter()),
            "the charter is held beside the body"
        );
        // Stated again: the same body, nothing counted but the one charter already held.
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
        assert_eq!(
            lane.counters(),
            ChunkCounters {
                charters_held: 1,
                ..ChunkCounters::default()
            }
        );
        // A foreign recipe tag: refused, counted, no body.
        let other = RealmId::Planet(5);
        lane.state_surface(other, &surface(78), Some(&charter()), &look());
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
            Some(&charter()),
            &look(),
        );
        lane.state_surface(
            RealmId::Planet(7),
            &surface(77),
            Some(&charter()),
            &Boundary::Shell { r: -1.0 },
        );
        assert_eq!(lane.counters().no_body, 2);
        // A box look is not a round body.
        lane.state_surface(
            RealmId::Planet(8),
            &surface(77),
            Some(&charter()),
            &Boundary::Aabb {
                half: vd_core::glam::DVec3::ONE,
            },
        );
        assert_eq!(lane.counters().no_body, 3);
        lane.forget(planet());
        assert!(lane.body(planet()).is_none());
        assert!(
            lane.charter(planet()).is_none(),
            "a forgotten realm's charter goes with its body"
        );
    }

    /// ★ G-REFUSE (slice 8b stage 1): a realm that states a surface and NO CHARTER is refused, and
    /// the counter says so. The observed-failing control of the A1 crossing — without it a client
    /// would build a planet from numbers nobody authored.
    #[test]
    fn a_surface_without_a_charter_is_refused_and_counted() {
        let mut lane = lane();
        lane.state_surface(planet(), &surface(77), None, &look());
        assert_eq!(lane.counters().no_charter, 1, "the refusal is counted");
        assert!(lane.body(planet()).is_none(), "no body is built");
        assert!(lane.charter(planet()).is_none());
        // Counted ONCE: a refused realm is ignored until it is forgotten.
        lane.state_surface(planet(), &surface(77), None, &look());
        assert_eq!(lane.counters().no_charter, 1);
        assert_eq!(lane.counters().charters_held, 0);
        // Forgotten, then stated WITH a charter: the body is built and the charter is held.
        lane.forget(planet());
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
        assert_eq!(lane.body(planet()).map(|b| **b), Some(home_planet()));
        assert_eq!(lane.charter(planet()), Some(&charter()));
        assert_eq!(lane.counters().charters_held, 1);
        assert_eq!(lane.counters().no_charter, 1);
    }

    #[test]
    fn the_lane_builds_holds_and_releases_chunks_of_every_rung() {
        let mut lane = lane();
        let body = home_planet();
        // No body yet: refused and counted.
        let k0 = surface_key(&body, 0, 300, 700);
        lane.request(planet(), k0, 0);
        assert_eq!(lane.counters().no_body, 1);
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
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
            0,
        );
        assert_eq!(lane.counters().outside, 1);
        // A second request of the same key is not queued twice.
        lane.request(planet(), k0, 0);
        lane.request(planet(), k0, 0);
        assert!(lane.holds(planet(), k0));
        assert!(!lane.is_resident(planet(), k0));
        assert_eq!(lane.counters().submitted, 1);
        // A second rung beside the first: served (step 2 — every rung at once).
        let k3 = surface_key(&body, 3, 30, 70);
        lane.request(planet(), k3, 0);
        assert!(lane.holds(planet(), k3));
        assert_eq!(lane.counters().submitted, 2);
        // Harvest: both chunks are resident, each with geometry relative to its origin.
        let ready = lane.poll(8);
        assert_eq!(ready.len(), 2);
        assert_eq!(ready[0].realm, planet());
        let mut got: Vec<ChunkKey> = ready.iter().map(|r| r.geometry.key).collect();
        got.sort();
        let mut want = vec![k0, k3];
        want.sort();
        assert_eq!(got, want);
        assert_eq!(lane.resident(planet()), want);
        assert!(lane.is_resident(planet(), k0));
        assert!(lane.is_resident(planet(), k3));
        assert_eq!(lane.counters().harvested, 2);
        // The build count and the harvest cap (M8-2a): two built, no poll filled its cap of 8.
        assert_eq!(
            lane.built(),
            BuildCount {
                chunks: 2,
                nanos: 0,
                peak_nanos: 0,
                peak_key: None
            }
        );
        assert_eq!(lane.counters().harvest_full, 0);
        // Two rung-0 chunks read their parents through the cache: some built, none waited; the
        // cache is the lane's own.
        assert!(lane.parent_stats().builds > 0);
        assert_eq!(lane.parent_stats().waits, 0);
        assert_eq!(lane.parents().stats(), lane.parent_stats());
        // A request of a chunk already resident changes nothing.
        lane.request(planet(), k0, 9);
        assert_eq!(lane.counters().submitted, 2);
        // A poll with a cap of zero harvests nothing and counts no full cap.
        assert!(lane.poll(0).is_empty());
        assert_eq!(lane.counters().harvest_full, 0);
        assert!(lane.poll(8).is_empty());
        // Release both; a chunk released while building is never handed out.
        lane.release(planet(), k0);
        lane.release(planet(), k3);
        assert!(!lane.holds(planet(), k0));
        lane.request(planet(), k3, 0);
        assert!(lane.holds(planet(), k3));
        assert!(!lane.is_resident(planet(), k3));
        lane.release(planet(), k3);
        assert!(lane.poll(8).is_empty());
        assert!(lane.resident(planet()).is_empty());
        // A bounded harvest: two chunks queued, one per poll.
        let k1 = surface_key(&body, 0, 301, 700);
        lane.request(planet(), k0, 0);
        lane.request(planet(), k1, 0);
        assert_eq!(lane.pending_count(), 2);
        let mut building = lane.pending_all();
        building.sort();
        assert_eq!(building, vec![(planet(), k0), (planet(), k1)]);
        assert_eq!(lane.poll(1).len(), 1);
        assert_eq!(lane.pending_count(), 1);
        assert_eq!(lane.poll(1).len(), 1);
        assert_eq!(lane.pending_count(), 0);
        // Both polls filled their cap of one; a re-request of a pending chunk re-keys it (the
        // inline workers have nothing waiting, so this is the seam's no-op).
        assert_eq!(lane.counters().harvest_full, 2);
        lane.request(planet(), k3, 7);
        lane.request(planet(), k3, 3);
        assert_eq!(lane.counters().submitted, 6);
        assert_eq!(lane.resident(planet()).len(), 2);
        // Releasing ONE of two keeps the other; another rung is served beside it.
        lane.release(planet(), k0);
        lane.request(planet(), k3, 0);
        assert!(lane.holds(planet(), k3));
        lane.release(planet(), k3);
        // A release of a chunk never held changes nothing.
        lane.release(planet(), k3);
        assert_eq!(lane.resident(planet()), vec![k1]);
        // Forgetting a realm withdraws what is still building: nothing is ever handed out for it.
        lane.request(planet(), k0, 0);
        lane.forget(planet());
        assert!(lane.poll(8).is_empty());
        assert_eq!(lane.pending_count(), 0);
        // The job and the finished chunk are plain data (their derives are exercised here).
        let job = ChunkJob {
            realm: planet(),
            body: Arc::new(home_planet()),
            key: k0,
            parents: Arc::new(ParentCache::default()),
            priority: 0,
        };
        assert!(format!("{:?}", job.clone()).contains("ChunkJob"));
        let mut inline = InlineWorkers::default();
        inline.submit(job);
        let mut out = Vec::new();
        inline.drain(&mut out, 8);
        assert_eq!(out.len(), 1);
        assert_eq!(inline.built().chunks, 1);
        assert!(format!("{:?}", inline.built()).contains("BuildCount"));
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
        assert_eq!(&g.triangles[..mesh.triangles.len()], &mesh.triangles[..]);
        assert!(g.vertices.len() >= mesh.vertices.len());
        assert_eq!(g.normals.len(), g.vertices.len());
        let mut worst = 0.0f64;
        for (v, rel) in mesh.vertices.iter().zip(g.vertices.iter()) {
            let p = vertex_position_m(&body, &samples, *v);
            let mut k = 0;
            while k < 3 {
                let drawn = g.origin_m[k] + f64::from(rel[k]);
                worst = worst.max((drawn - p[k]).abs());
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
    fn a_radial_meets_a_triangle_inside_it_and_on_its_edge_and_misses_it_beside() {
        let p0 = DVec3::new(100.0, -1.0, -1.0);
        let p1 = DVec3::new(100.0, 3.0, -1.0);
        let p2 = DVec3::new(100.0, -1.0, 3.0);
        // Through the inside: at the plane, 100 m out.
        let t = ray_triangle_m(DVec3::X, p0, p1, p2).expect("inside");
        assert!((t - 100.0).abs() < 1e-9, "{t}");
        // Along an edge (y = −1): still a crossing.
        let along = DVec3::new(100.0, -1.0, 1.0).normalize();
        assert!(ray_triangle_m(along, p0, p1, p2).is_some());
        // Beside the triangle: none. Parallel to it: none. Behind the centre: none.
        assert_eq!(
            ray_triangle_m(DVec3::new(100.0, 5.0, 5.0).normalize(), p0, p1, p2),
            None
        );
        assert_eq!(ray_triangle_m(DVec3::Y, p0, p1, p2), None);
        assert_eq!(ray_triangle_m(DVec3::NEG_X, p0, p1, p2), None);
    }

    #[test]
    fn two_triangles_on_one_radial_answer_the_nearer_one_and_a_tie_the_lower() {
        // A hand-built parent: two level triangles on the +x radial, at 100 m and 130 m, both
        // in bucket (0, 0).
        let tri = |r: f32| -> [[f32; 3]; 3] { [[r, -1.0, -1.0], [r, 3.0, -1.0], [r, -1.0, 3.0]] };
        let (near, far) = (tri(100.0), tri(130.0));
        // Both triangles in the bucket of cell (0, 0) and no other: the table's starts step from
        // 0 to 2 right after that cell.
        let one_bucket = |a: i32, b: i32, tris: Vec<u32>| -> (Vec<u32>, Vec<u32>) {
            let c = parent_bucket(a, b).expect("inside the box");
            let mut start = vec![0u32; (PARENT_CELLS * PARENT_CELLS) as usize + 1];
            let mut i = c + 1;
            while i < start.len() {
                start[i] = tris.len() as u32;
                i += 1;
            }
            (start, tris)
        };
        let (bucket_start, bucket_tris) = one_bucket(0, 0, vec![0, 1]);
        let pm = ParentMesh {
            key: ChunkKey {
                face: Face::PosX,
                rung: 1,
                x: 0,
                y: 0,
                z: 0,
            },
            origin: DVec3::ZERO,
            positions: vec![near[0], near[1], near[2], far[0], far[1], far[2]],
            normals: vec![oct_encode([1.0, 0.0, 0.0]); 6],
            triangles: ParentTriangles::pack(vec![[0, 1, 2], [3, 4, 5]], 6),
            bucket_start: bucket_start.clone(),
            bucket_tris: bucket_tris.clone(),
        };
        // Asked near 128: the far one; near 102: the near one; at 115 (a tie): the lower.
        assert_eq!(pm.radial_hit_m(0, 0, DVec3::X, 128.0), Some(130.0));
        assert_eq!(pm.radial_hit_m(0, 0, DVec3::X, 102.0), Some(100.0));
        assert_eq!(pm.radial_hit_m(0, 0, DVec3::X, 115.0), Some(100.0));
        // The same triangles the other way round meet the same tie the same way.
        let swapped = ParentMesh {
            key: pm.key,
            origin: DVec3::ZERO,
            positions: vec![far[0], far[1], far[2], near[0], near[1], near[2]],
            // The far triangle shaded along Y, the near one along X (item 20).
            normals: vec![
                oct_encode([0.0, 1.0, 0.0]),
                oct_encode([0.0, 1.0, 0.0]),
                oct_encode([0.0, 1.0, 0.0]),
                oct_encode([1.0, 0.0, 0.0]),
                oct_encode([1.0, 0.0, 0.0]),
                oct_encode([1.0, 0.0, 0.0]),
            ],
            triangles: ParentTriangles::pack(vec![[0, 1, 2], [3, 4, 5]], 6),
            bucket_start,
            bucket_tris,
        };
        assert_eq!(swapped.radial_hit_m(0, 0, DVec3::X, 115.0), Some(100.0));
        // THE PARENT'S SHADE AT THE HIT (item 20): the near triangle's X, the far one's Y.
        let (near_hit, near_shade) = swapped.radial_hit(0, 0, DVec3::X, 102.0).expect("near");
        assert_eq!(near_hit, 100.0);
        assert!((near_shade - DVec3::X).length() < 1e-3, "{near_shade:?}");
        let (far_hit, far_shade) = swapped.radial_hit(0, 0, DVec3::X, 128.0).expect("far");
        assert_eq!(far_hit, 130.0);
        assert!((far_shade - DVec3::Y).length() < 1e-3, "{far_shade:?}");
    }

    #[test]
    fn a_vertex_on_the_face_edge_reads_the_field_from_both_faces() {
        let body = home_planet();
        let cache = ParentCache::default();
        // A chunk at the face's low x edge: its vertices at the edge read the field.
        let edge = surface_key(&body, 0, 0, 700);
        let g = geometry_with(&body, planet(), edge, &cache).expect("the edge chunk");
        assert!(g.morph_seam > 0, "no seam vertex on an edge chunk");
        let (fallbacks, vertices) = (g.morph_fallbacks, g.vertices.len() as u32);
        assert!(
            fallbacks * 100 < vertices,
            "{fallbacks} fallbacks of {vertices}"
        );
        let samples = sample_box(&body, edge).expect("in the band");
        let mesh = extract(&samples);
        let o = DVec3::from_array(g.origin_m);
        let mut checked = 0;
        for (v, m) in mesh.vertices.iter().zip(g.morph_targets().iter()) {
            if v[0] <= 0 {
                let t = o + DVec3::new(f64::from(m[0]), f64::from(m[1]), f64::from(m[2]));
                let dir = t.normalize();
                let field = vd_terrain::height::height_m(&body, [dir.x, dir.y, dir.z], 1);
                let radius = t.length();
                assert!((radius - field).abs() < 0.01, "{radius} vs {field}");
                checked += 1;
            }
        }
        assert!(checked > 0);
        // A chunk in the middle of the face has no seam vertex.
        let mid = geometry_with(&body, planet(), surface_key(&body, 0, 300, 700), &cache)
            .expect("a middle chunk");
        assert_eq!(mid.morph_seam, 0);
    }

    #[test]
    fn the_parent_mesh_stands_within_the_sink_of_its_field_on_every_radial() {
        // THE SINK'S CLAIM, MEASURED: along every radial of a chunk's own vertices, the parent
        // mesh's hit and the coarser field stand apart by less than the two cells the sink adds
        // to the gap bound — the extractor's placement, read on the surface a radial meets, not
        // only at a vertex.
        let body = home_planet();
        let cache = ParentCache::default();
        let mut worst: f64 = 0.0;
        let mut measured = 0;
        let mut over = 0usize;
        let mut outside: Vec<f64> = Vec::new();
        for (x, y) in [(300, 700), (301, 700), (150, 350), (2506, 1781)] {
            let rung = if x > 1000 {
                5
            } else if x < 200 {
                1
            } else {
                0
            };
            let key = surface_key(&body, rung, x, y);
            let coarser = rung + 1;
            let samples = sample_box(&body, key).expect("in the band");
            let mesh = extract(&samples);
            let parents: Vec<Arc<ParentMesh>> = parent_keys(&body, key)
                .into_iter()
                .filter_map(|p| cache.get(planet(), &body, p))
                .collect();
            for v in mesh.vertices.iter().step_by(7) {
                let p = vertex_position_m(&body, &samples, *v);
                let pv = DVec3::new(p[0], p[1], p[2]);
                let len = pv.length();
                let dir = pv / len;
                let field = vd_terrain::height::height_m(&body, [dir.x, dir.y, dir.z], coarser);
                // The hit nearest the FIELD's own radius: the coarser surface, not a cave.
                let hit = parents.iter().fold(None, |best: Option<f64>, pm| {
                    let (a, b) = parent_cell(key, pm.key(), *v);
                    match (best, pm.radial_hit_m(a, b, dir, field)) {
                        (Some(b), Some(h)) if (h - field).abs() < (b - field).abs() => Some(h),
                        (None, Some(h)) => Some(h),
                        (b, _) => b,
                    }
                });
                // Branchless (HR5): a radial with no hit adds nothing.
                let met = hit.is_some();
                let gap = (hit.unwrap_or(field) - field).abs();
                let cells = f64::from(vd_seed::ladder::cell_m(coarser))
                    + f64::from(vd_seed::ladder::cell_m(rung));
                over += usize::from(gap > cells);
                if gap > cells {
                    outside.push(hit.unwrap_or(field) - field);
                }
                worst = worst.max(gap);
                measured += usize::from(met);
            }
        }
        // ★ THE ONE RADIAL THAT DOES NOT, RE-MEASURED 2026-09-18 on slice 8b stage 3 (THE RELIEF
        // LAW): 4 475 of the 4 476 radials stand inside the sink; ONE stands outside it, at 31.29 m.
        //
        // The number moves at every stage that moves the ground, and it has moved six times now:
        // SIX of 4 142 radials (worst 210.26 m) before the spectrum; NINETEEN of 5 284 (worst
        // 170.28 m) at 8a stage 1; THREE of 3 242 (worst 136.21 m) at stage 2; THIRTEEN of 6 180
        // (worst 128.91 m) at stage 3; FIFTEEN of 5 630 (worst 197.08 m) at stage 4; ONE of 4 476
        // (worst 31.29 m) here, where the relief law halved the mountains and the caves under this
        // box with them. The RADIAL COUNT is
        // the field's own roughness read back: a
        // smoother column carries more vertices over the parent's cells that answer a radial at
        // all, and the factor lowers the fine half of nine columns in ten.
        //
        // ★ THE RESIDUE IS THE SAME CLASS, and it is MEASURED, not argued — the two discriminators
        // are ASSERTED below and each could fail:
        //   1. EVERY ONE of the hits stands BELOW its field. The fifteen signed gaps are all
        //      negative, the deepest −197.08 m. A misplaced parent mesh would stand on
        //      both sides of the field and at the CELL's own scale (3 m at rung 0, 6 m at rung 1).
        //   2. Every gap sits inside the body's own CAVE BAND (8 m to 370 m under the surface).
        // So the radial slips past the hill into a cave and reports the cave's depth, not the
        // extractor's placement. The sink is a claim about the SURFACE, and a cave is not the surface.
        //
        // ⚠ OWED, THE OWNER'S CALL: whether a morph target should follow its parent into a cave at
        // all. Until it is answered this test states the MEASURED residue by name, so it goes red
        // the moment the residue grows.
        assert_eq!(
            over, 1,
            "{over} of {measured} radials outside the sink, the worst {worst:.2} m"
        );
        assert!(
            worst < 32.0,
            "the worst radial stands {worst:.2} m from its field"
        );
        assert!(measured > 1000, "{measured}");
        // The two discriminators, as assertions: the residue is a radial in a CAVE, never a mesh in
        // the wrong place.
        let mut below = 0;
        let mut in_band = 0;
        for gap in &outside {
            below += usize::from(*gap < 0.0);
            in_band += usize::from((gap.abs() >= 8.0) & (gap.abs() <= 370.0));
        }
        assert_eq!(
            below, over,
            "every residue stands BELOW its field: {outside:?}"
        );
        assert_eq!(in_band, over, "and inside the cave band: {outside:?}");
    }

    #[test]
    fn a_parent_mesh_answers_the_radials_over_its_cells_nearest_the_asked_radius() {
        let body = home_planet();
        let key = surface_key(&body, 1, 150, 350);
        let pm = ParentMesh::build(&body, key).expect("the parent");
        assert_eq!(pm.key(), key);
        assert!(pm.triangle_count() > 0);
        // Every vertex of the parent's own mesh is met by its own radial, at its own radius,
        // within the cell it lies in (a vertex is on its triangles).
        let samples = sample_box(&body, key).expect("in the band");
        let mesh = extract_all_edges(&samples);
        let q = i32::from(vd_terrain::VERTEX_QUANTUM as i16);
        let mut met = 0;
        for v in mesh.vertices.iter().step_by(41) {
            let p = vertex_position_m(&body, &samples, *v);
            let p = DVec3::new(p[0], p[1], p[2]);
            let (a, b) = (i32::from(v[0]).div_euclid(q), i32::from(v[1]).div_euclid(q));
            let r = p.length();
            let hit = pm.radial_hit_m(a, b, p / r, r).expect("its own radial");
            assert!((hit - r).abs() < 1e-3, "{hit} vs {r}");
            met += 1;
        }
        assert!(met > 50);
        // A cell with no triangle: no hit. Out of the body: no parent.
        assert_eq!(pm.radial_hit_m(1000, 1000, DVec3::X, 1.0), None);
        let outside = ChunkKey { z: -1, ..key };
        assert!(ParentMesh::build(&body, outside).is_none());
    }

    #[test]
    fn the_parent_cache_shares_a_build_and_forgets_the_oldest() {
        let body = home_planet();
        let cache = ParentCache::default();
        assert!(cache.is_empty());
        let key = surface_key(&body, 1, 150, 350);
        let a = cache.get(planet(), &body, key).expect("built");
        let b = cache.get(planet(), &body, key).expect("cached");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 1);
        // The same key inserted again keeps one entry (and one place in the order).
        cache.insert(planet(), key, Arc::clone(&a));
        assert_eq!(cache.len(), 1);
        // A key outside the body builds nothing and keeps nothing.
        assert!(
            cache
                .get(planet(), &body, ChunkKey { z: -1, ..key })
                .is_none()
        );
        assert_eq!(cache.len(), 1);
        // A hit refreshes its place: after the cap is passed by fresh keys, the hit one stays
        // and the oldest untouched one leaves.
        let second = ChunkKey { x: 151, ..key };
        let b2 = cache.get(planet(), &body, second).expect("built");
        let _ = cache.get(planet(), &body, key).expect("hit, refreshed");
        let mut i = 0;
        while i < PARENT_CACHE_ENTRIES - 1 {
            cache.insert(
                planet(),
                ChunkKey {
                    x: 20_000 + i as i32,
                    ..key
                },
                Arc::clone(&a),
            );
            i += 1;
        }
        assert_eq!(cache.len(), PARENT_CACHE_ENTRIES);
        assert!(
            cache
                .get(planet(), &body, key)
                .is_some_and(|c| Arc::ptr_eq(&c, &a))
        );
        assert!(
            cache
                .get(planet(), &body, second)
                .is_some_and(|c| !Arc::ptr_eq(&c, &b2))
        );
        // The capacity is settable: a smaller bound drops the oldest at once.
        assert_eq!(cache.capacity(), PARENT_CACHE_ENTRIES);
        cache.set_capacity(PARENT_CACHE_ENTRIES + 10);
        assert_eq!(cache.capacity(), PARENT_CACHE_ENTRIES + 10);
        cache.set_capacity(PARENT_CACHE_ENTRIES);
        assert_eq!(cache.len(), PARENT_CACHE_ENTRIES);
        cache.set_capacity(0);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.len(), 1);
        cache.set_capacity(PARENT_CACHE_ENTRIES);
        // The bound: one more than the cap, and the first one is gone.
        let mut i = 0;
        while i < PARENT_CACHE_ENTRIES {
            cache.insert(
                planet(),
                ChunkKey {
                    x: 10_000 + i as i32,
                    ..key
                },
                Arc::clone(&a),
            );
            i += 1;
        }
        assert_eq!(cache.len(), PARENT_CACHE_ENTRIES);
        assert!(
            cache
                .get(planet(), &body, key)
                .is_some_and(|c| !Arc::ptr_eq(&c, &a))
        );
        // Forgetting a realm drops its meshes and nobody else's.
        cache.insert(RealmId::Planet(1), key, Arc::clone(&a));
        cache.forget(planet());
        assert_eq!(cache.len(), 1);
        cache.forget(RealmId::Planet(1));
        assert!(cache.is_empty());
    }

    #[test]
    fn the_parents_are_the_coarser_chunk_and_its_neighbour_on_the_side_faced() {
        let body = home_planet();
        let top = body.ladder().rungs - 1;
        assert!(parent_keys(&body, surface_key(&body, top, 3, 3)).is_empty());
        let even = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 301,
            y: 700,
            z: 276,
        };
        // x odd faces +x, y even faces −y, z even faces down: eight parents.
        let ps = parent_keys(&body, even);
        assert_eq!(ps.len(), 8);
        assert_eq!((ps[0].rung, ps[0].x, ps[0].y, ps[0].z), (1, 150, 350, 138));
        let xs: BTreeSet<i32> = ps.iter().map(|p| p.x).collect();
        let ys: BTreeSet<i32> = ps.iter().map(|p| p.y).collect();
        let zs: BTreeSet<i32> = ps.iter().map(|p| p.z).collect();
        assert_eq!(xs, [150, 151].into_iter().collect());
        assert_eq!(ys, [349, 350].into_iter().collect());
        assert_eq!(zs, [137, 138].into_iter().collect());
        let odd = ChunkKey { z: 277, ..even };
        let zs: BTreeSet<i32> = parent_keys(&body, odd).iter().map(|p| p.z).collect();
        assert_eq!(zs, [138, 139].into_iter().collect(), "an odd z faces up");
        // At the band's floor, the face's edge and the band's top the outside is left out.
        assert_eq!(parent_keys(&body, ChunkKey { z: 0, ..even }).len(), 4);
        assert_eq!(parent_keys(&body, ChunkKey { x: 0, ..even }).len(), 4);
        let top_z = vd_terrain::digest::top_chunk_z(&body, 0);
        let top_parent = vd_terrain::digest::top_chunk_z(&body, 1);
        let at_top = ChunkKey { z: top_z, ..even };
        let ps = parent_keys(&body, at_top);
        assert!(ps.iter().all(|p| p.z <= top_parent));
        // The parent cell under a finer vertex: the finer chunk's cell halved, in the parent's
        // box — a low halo vertex of an even chunk reads the parent's halo column −1.
        assert_eq!(parent_cell(even, ps[0], [0, 0, 0]), (31, 0));
        let even_x = ChunkKey { x: 300, ..even };
        let parent = parent_keys(&body, even_x)[0];
        assert_eq!(parent_cell(even_x, parent, [-256, 0, 0]), (-1, 0));
        assert_eq!(parent_cell(even, parent, [62 * 256, 0, 0]), (62, 0));
    }

    #[test]
    fn the_morph_targets_stand_on_the_parent_mesh_and_neighbours_share_them() {
        let body = home_planet();
        let cache = ParentCache::default();
        let left = surface_key(&body, 0, 300, 700);
        let right = ChunkKey { x: 301, ..left };
        let gl = geometry_with(&body, planet(), left, &cache).expect("left");
        let gr = geometry_with(&body, planet(), right, &cache).expect("right");
        // Near nothing fell back to the field.
        let fallbacks = gl.morph_fallbacks;
        let vertices = gl.vertices.len() as u32;
        assert!(
            fallbacks * 100 < vertices,
            "{fallbacks} fallbacks of {vertices}"
        );
        // The two chunks share their parent builds: eight parents each, twelve at most in all.
        assert!(cache.len() <= 12, "{}", cache.len());
        // Every target lies ON a parent triangle: the radial through it meets the parent mesh
        // at its own radius.
        let parents: Vec<Arc<ParentMesh>> = parent_keys(&body, left)
            .into_iter()
            .filter_map(|p| cache.get(planet(), &body, p))
            .collect();
        let samples = sample_box(&body, left).expect("in the band");
        let mesh = extract(&samples);
        let o = DVec3::from_array(gl.origin_m);
        let mut checked = 0;
        let mut deep = 0u32;
        let mut deepest: f64 = 0.0;
        for (v, m) in mesh.vertices.iter().zip(gl.morph_targets().iter()) {
            // The exact radial of the vertex (the target's own is rounded through f32).
            let p = vertex_position_m(&body, &samples, *v);
            let dir = DVec3::new(p[0], p[1], p[2]).normalize();
            let t = o + DVec3::new(f64::from(m[0]), f64::from(m[1]), f64::from(m[2]));
            let r = t.length();
            let on = parents.iter().any(|pm| {
                let (a, b) = parent_cell(left, pm.key(), *v);
                pm.radial_hit_m(a, b, dir, r)
                    .is_some_and(|h| (h - r).abs() < 0.01)
            });
            checked += u32::from(on);
            // ★ A VERTEX IN A CAVE READS THE FIELD BY RULE, and the builder does not count it as a
            // fallback (`missing & ((field − len).abs() <= reach_m)`). The test names the same rule
            // through the same function, so the two can never drift apart.
            let len = DVec3::new(p[0], p[1], p[2]).length();
            let field = vd_terrain::height::height_m(&body, [dir.x, dir.y, dir.z], left.rung + 1);
            let reach = sink_m(&body, left.rung + 1);
            if !on & ((field - len).abs() > reach) {
                deep += 1;
                deepest = deepest.min(len - field);
            }
        }
        // The extractor's own vertices; the skirts' targets are their tops' dropped.
        let targets = mesh.vertices.len() as u32;
        // ★ RE-MEASURED 2026-09-17 on slice 8a stage 4 (the cap-rock bench): the targets of this
        // box lie on a parent triangle but for TWO, which stand IN A CAVE; 0 are recorded fallbacks
        // and 0 stand on a box seam. None is none of the four — the claim is whole. (Stage 3 read
        // 8 572 of 8 579 with SEVEN in a cave; the bench moves the ground, so it moves which
        // vertices fall through it.)
        //
        // ★ THE CAVE ARM IS NAMED HERE FOR THE FIRST TIME, and a PROBE found it rather than an
        // argument. The builder's own rule is explicit: a vertex whose coarser field stands farther
        // than the SINK away is "a cave's own and reads the field by nature", so it is neither a
        // parent hit nor a counted fallback. The seven stand 26.42 m to 45.85 m UNDER their coarser
        // field — inside the body's own cave band — and their targets sit EXACTLY on that field
        // (a gap of 0.000 m). Six have no parent hit in their cell at all; the seventh's only hit is
        // 27.49 m below, which is the cave's far wall.
        //
        // This box held no such vertex on the earlier ground, so the two-way claim happened to hold.
        // The test now reads the SAME `sink_m` the builder reads, so the two can never drift apart.
        //
        // (2026-09-18, slice 8b stage 3, THE RELIEF LAW: the box now holds FOUR cave vertices, not
        // two, and the deepest stands 26.57 m under its field — still inside the body's own cave
        // band. The mountains halved, so more of this box's ground sits at the depth the caves are
        // carved at.)
        //
        // (2026-09-15, the earlier reading: 4 489 of 4 493 on a parent triangle, four fallbacks.
        // Before the crust rule the same box held 8 189 targets and two fell into a CAVE MOUTH in
        // the parent.)
        let seam = gl.morph_seam;
        assert_eq!(
            checked + fallbacks + seam + deep,
            targets,
            "{checked} on the parent of {targets}, {fallbacks} fallbacks, {seam} on a seam, {deep} in a cave"
        );
        assert_eq!(seam, 0, "this box holds no seam vertex");
        assert_eq!(deep, 4, "the cave vertices of this box");
        assert!(
            (-28.0..=-26.0).contains(&deepest),
            "the deepest cave vertex stands {deepest:.2} m under its field"
        );
        // A vertex the two chunks share (the same world position) has the same target.
        let ol = DVec3::from_array(gl.origin_m);
        let or = DVec3::from_array(gr.origin_m);
        let mut shared = 0;
        let mut right_at: BTreeMap<[i64; 3], DVec3> = BTreeMap::new();
        for (v, m) in gr.vertices.iter().zip(gr.morph_targets().iter()) {
            let p = or + DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            let k = [
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.z * 1000.0).round() as i64,
            ];
            right_at.insert(
                k,
                or + DVec3::new(f64::from(m[0]), f64::from(m[1]), f64::from(m[2])),
            );
        }
        for (v, m) in gl.vertices.iter().zip(gl.morph_targets().iter()) {
            let p = ol + DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            let k = [
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.z * 1000.0).round() as i64,
            ];
            if let Some(tr) = right_at.get(&k) {
                let tl = ol + DVec3::new(f64::from(m[0]), f64::from(m[1]), f64::from(m[2]));
                assert!((tl - *tr).length() < 0.01, "{tl} vs {tr}");
                shared += 1;
            }
        }
        assert!(shared > 10, "{shared} shared vertices");
    }

    #[test]
    fn the_skirts_hang_two_cells_under_every_boundary_edge_facing_outward() {
        let body = home_planet();
        let key = surface_key(&body, 0, 300, 700);
        let samples = sample_box(&body, key).expect("in the band");
        let mesh = extract(&samples);
        let g = geometry_of(&body, key).expect("the chunk");
        // The boundary edges of the extractor's mesh, by a count of their uses.
        let mut uses: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for t in &mesh.triangles {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let boundary = uses.values().filter(|c| **c == 1).count();
        assert!(boundary > 100, "{boundary} boundary edges");
        assert_eq!(g.vertices.len(), mesh.vertices.len() + 2 * boundary);
        assert_eq!(g.triangles.len(), mesh.triangles.len() + 2 * boundary);
        // Every skirt vertex hangs two cells under a top vertex along its radial, keeps its
        // normal, and its morph target is the top's target dropped the same.
        let o = DVec3::from_array(g.origin_m);
        let drop = 2.0;
        let mut i = mesh.vertices.len();
        while i < g.vertices.len() {
            let q = o + DVec3::from(g.vertices[i].map(f64::from));
            // The top: the vertex of the base mesh at the same radial, two cells up.
            let top = g.vertices[..mesh.vertices.len()]
                .iter()
                .enumerate()
                .find(|(_, v)| {
                    let p = o + DVec3::from(v.map(f64::from));
                    (p - q).length() < drop + 0.01
                        && (p - q).normalize().dot(p.normalize()) > 0.999_999
                        && ((p - q).length() - drop).abs() < 0.01
                })
                .map(|(j, _)| j)
                .expect("a top vertex over the skirt vertex");
            assert_eq!(g.normals[i], g.normals[top]);
            let (si, st) = (
                DVec3::from(g.sink_of(i).map(f64::from)),
                DVec3::from(g.sink_of(top).map(f64::from)),
            );
            assert!((si - st).length() < 1e-3);
            let mt = o + DVec3::from(g.morph_target(top).map(f64::from));
            let ms = o + DVec3::from(g.morph_target(i).map(f64::from));
            assert!(((mt - ms).length() - drop).abs() < 0.01);
            i += 1;
        }
        // Every skirt triangle faces away from the chunk: its normal points away from the base
        // triangle's third corner (checked through the winding rule the builder applied).
        let mut outward = 0;
        for t in &g.triangles[mesh.triangles.len()..] {
            let p = |k: u32| o + DVec3::from(g.vertices[k as usize].map(f64::from));
            let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            // The strip is radial: its normal is across the radial, never along it.
            let radial = p(t[0]).normalize();
            let along = n.normalize().dot(radial);
            assert!(along.abs() < 0.2, "{along}");
            outward += 1;
        }
        assert_eq!(outward, 2 * boundary);
    }

    #[test]
    fn the_sink_is_the_gap_bound_plus_a_cell_of_each_rung_and_rung_zero_sinks_nowhere() {
        let body = home_planet();
        assert_eq!(sink_m(&body, 0), 0.0);
        let s1 = sink_m(&body, 1);
        let gap = body.dropped_bound_m(1) - body.dropped_bound_m(0);
        assert!((s1 - (gap + 2.0 + 1.0)).abs() < 1e-9, "{s1}");
        assert!(sink_m(&body, 3) > sink_m(&body, 1));
        // On a chunk: rung 0 carries zeros; rung 1 carries its radial times the sink.
        let g0 = geometry_of(&body, surface_key(&body, 0, 300, 700)).expect("rung 0");
        assert_eq!(g0.sink_m, 0.0);
        assert!((0..g0.vertices.len()).all(|i| g0.sink_of(i) == [0.0, 0.0, 0.0]));
        let g1 = geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1");
        assert_eq!(g1.morph_m.len(), g1.vertices.len());
        assert!((f64::from(g1.sink_m) - s1).abs() < 1e-3);
        let o = vd_core::glam::DVec3::from_array(g1.origin_m);
        for (i, v) in g1.vertices.iter().enumerate() {
            let s = g1.sink_of(i);
            let pv =
                o + vd_core::glam::DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            let sv = vd_core::glam::DVec3::new(f64::from(s[0]), f64::from(s[1]), f64::from(s[2]));
            assert!((sv.length() - s1).abs() < 1e-3, "{} vs {s1}", sv.length());
            assert!(sv.normalize().dot(pv.normalize()) > 0.999_999);
        }
    }

    /// THE PACKING IS EXACT (step 5, ruling V16): the target the vector form carried — the hit's
    /// radial distance times the radial, less the origin, narrowed once — and the target the
    /// metre form reconstructs — the narrowed vertex plus its radial times the metre — differ by
    /// the narrowing alone. MEASURED here over every vertex of two chunks, with the bound the
    /// gate holds: under a tenth of a millimetre, a thousandth of the finest cell.
    #[test]
    fn the_morph_metre_reconstructs_the_target_vector_to_float_rounding() {
        let body = home_planet();
        let mut widest = 0.0f64;
        let mut counted = 0;
        for key in [
            surface_key(&body, 0, 300, 700),
            surface_key(&body, 1, 150, 350),
        ] {
            let g = geometry_of(&body, key).expect("the chunk");
            let o = vd_core::glam::DVec3::from_array(g.origin_m);
            for (i, v) in g.vertices.iter().enumerate() {
                // The vector form's target, as step 3 computed it: from the absolute position.
                let abs = o + vd_core::glam::DVec3::new(
                    f64::from(v[0]),
                    f64::from(v[1]),
                    f64::from(v[2]),
                );
                let len = abs.length();
                let dir = abs / len;
                let target_m = len + f64::from(g.morph_m[i]);
                let t = dir * target_m - o;
                let vector_form = [t.x as f32, t.y as f32, t.z as f32];
                let metre_form = g.morph_target(i);
                let mut k = 0;
                while k < 3 {
                    widest =
                        widest.max((f64::from(vector_form[k]) - f64::from(metre_form[k])).abs());
                    k += 1;
                }
                counted += 1;
            }
        }
        assert!(counted > 5_000, "{counted} vertices");
        assert!(widest < 1e-4, "the two forms differ by {widest} m");
    }

    /// The angle between two unit vectors, in 64-bit from the cross product (an `acos` of a
    /// 32-bit dot reads a rounding of 6e-8 as 3.5e-4 rad).
    fn angle_between(a: [f32; 3], b: [f32; 3]) -> f64 {
        let a = DVec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]));
        let b = DVec3::new(f64::from(b[0]), f64::from(b[1]), f64::from(b[2]));
        a.cross(b).length().atan2(a.dot(b))
    }

    /// The coarse key two rungs up holds the fine chunk's volume (each axis quartered), agrees
    /// with two parent steps, and is refused past the top rung or off the coarser grid.
    #[test]
    fn the_coarse_key_holds_the_fine_chunk_and_stops_at_the_top() {
        let body = home_planet();
        let fine = surface_key(&body, 1, 150, 350);
        let coarse = coarse_key(&body, fine, 2).expect("two rungs up");
        assert_eq!((coarse.rung, coarse.x, coarse.y), (3, 37, 87));
        assert_eq!(coarse.z, fine.z.div_euclid(4));
        // One step agrees with the parent that holds the chunk.
        let one = coarse_key(&body, fine, 1).expect("one rung up");
        assert!(parent_keys(&body, fine).contains(&one));
        // Zero steps is the key itself.
        assert_eq!(coarse_key(&body, fine, 0), Some(fine));
        // Past the top rung: refused.
        let rungs = body.ladder().rungs;
        assert_eq!(coarse_key(&body, fine, rungs), None);
        assert_eq!(coarse_key(&body, fine, u8::MAX), None);
        // Off the grid: a key below zero stays below zero after the halving.
        let off = ChunkKey { z: -8, ..fine };
        assert_eq!(coarse_key(&body, off, 2), None);
    }

    /// The upload bytes: the packed stride times the vertices plus the indices at 16 bits for a
    /// chunk under 65 536 vertices, and at 32 bits past it.
    #[test]
    fn the_upload_bytes_follow_the_stride_and_the_index_width() {
        let body = home_planet();
        let g = geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1");
        assert_eq!(
            g.upload_bytes(),
            g.vertices.len() as u64 * UPLOAD_VERTEX_BYTES + g.triangles.len() as u64 * 6
        );
        let mut wide = ChunkGeometry {
            key: g.key,
            origin_m: g.origin_m,
            morph_m: Vec::new(),
            morph_fallbacks: 0,
            morph_seam: 0,
            sink_m: 0.0,
            vertices: vec![[0.0; 3]; 70_000],
            normals: Vec::new(),
            packed_normals: Vec::new(),
            morph_normals: Vec::new(),
            radials: Vec::new(),
            triangles: vec![[0, 1, 2]],
            bounds: ([0.0; 3], [0.0; 3]),
            skirt_start: 0,
        };
        assert_eq!(wide.upload_bytes(), 70_000 * UPLOAD_VERTEX_BYTES + 12);
        wide.vertices.truncate(3);
        assert_eq!(wide.upload_bytes(), 3 * UPLOAD_VERTEX_BYTES + 6);
    }

    /// THE BYTE BUDGET (ruling V15 item 2): a budget under one chunk's bytes hands out exactly
    /// one chunk and counts the harvest as full; a budget past every chunk's bytes hands out what
    /// the count allows and counts full only at the count; an empty queue counts nothing.
    #[test]
    fn the_harvest_stops_within_its_byte_budget() {
        let mut lane = lane();
        let body = home_planet();
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
        for x in 0..3 {
            lane.request(planet(), surface_key(&body, 0, 302 + x, 700), 0);
        }
        let one = lane.poll_within(8, 1);
        assert_eq!(one.len(), 1);
        assert!(one[0].geometry.upload_bytes() > 1);
        assert_eq!(lane.counters().harvest_full, 1);
        let rest = lane.poll_within(8, u64::MAX);
        assert_eq!(rest.len(), 2);
        assert_eq!(lane.counters().harvest_full, 1);
        assert!(lane.poll_within(8, 1).is_empty());
        assert_eq!(lane.counters().harvest_full, 1);
        // The count bound within a wide budget still counts full.
        for x in 0..2 {
            lane.request(planet(), surface_key(&body, 0, 310 + x, 700), 0);
        }
        assert_eq!(lane.poll_within(2, u64::MAX).len(), 2);
        assert_eq!(lane.counters().harvest_full, 2);
    }

    /// The skirt start: every vertex before it is the surface's, every one from it a skirt copy of
    /// a surface vertex (its radial and normal equal), and the split is not at either end.
    #[test]
    fn the_skirt_start_splits_the_surface_from_the_skirt() {
        let body = home_planet();
        let g = geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1");
        let start = g.skirt_start as usize;
        assert!(start > 0);
        assert!(start < g.vertices.len());
        let mut copied = 0usize;
        for i in start..g.vertices.len() {
            let same =
                (0..start).any(|j| g.normals[j] == g.normals[i] && g.radials[j] == g.radials[i]);
            copied += usize::from(same);
        }
        assert_eq!(copied, g.vertices.len() - start);
    }

    /// Every vertex a triangle uses has a unit normal. A zero one draws dark on the exact path
    /// (the shader's normalise of a zero) and LIT under the packed one (a zero folds to the
    /// square's centre, which unfolds to "up") — a whole-range change no tolerance allows. So
    /// the extractor may never hand a used vertex a zero normal: asserted over every rung of the
    /// home planet at four columns. (The black specks once blamed on this were the overlay's
    /// tick readout, refutation N-3; no zero normal was ever measured.)
    #[test]
    fn no_used_vertex_of_a_chunk_has_a_zero_normal() {
        let body = home_planet();
        let mut zero = 0usize;
        let mut total = 0usize;
        for rung in 0..body.ladder().rungs {
            for (x, y) in [(150, 350), (151, 350), (150, 351), (37, 88)] {
                let g = geometry_of(&body, surface_key(&body, rung, x >> rung, y >> rung))
                    .expect("in the ladder");
                assert_eq!(g.packed_normals.len(), g.normals.len());
                let mut used = vec![false; g.vertices.len()];
                for t in &g.triangles {
                    for i in t {
                        used[*i as usize] = true;
                    }
                }
                for (n, u) in g.normals.iter().zip(used.iter()) {
                    total += usize::from(*u);
                    zero += usize::from(*u) * usize::from(*n == [0.0, 0.0, 0.0]);
                }
            }
        }
        assert_eq!(
            zero, 0,
            "{zero} of {total} used vertices carry a zero normal"
        );
    }

    /// THE PACKED NORMAL (ruling V18): every normal of a real chunk unpacks within
    /// `OCT_NORMAL_STEP_RAD` of itself; the square's four quadrants, the lower hemisphere's fold,
    /// the axes and a zero normal each round-trip.
    #[test]
    fn the_packed_normal_unpacks_within_its_step_on_every_vertex() {
        let body = home_planet();
        let g = geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1");
        let mut widest = 0.0f64;
        let mut widest_plain = 0.0f64;
        // The chunk's normals and twelve synthetic ones (both hemispheres, the axes, the
        // quadrants), so the fold's both arms run below.
        let s = 1.0 / 3.0f32.sqrt();
        let samples = [
            [s, s, s],
            [-s, s, s],
            [s, -s, s],
            [-s, -s, s],
            [s, s, -s],
            [-s, s, -s],
            [s, -s, -s],
            [-s, -s, -s],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for n in g.normals.iter().copied().chain(samples) {
            let back = oct_decode(oct_encode(n));
            let angle = angle_between(n, back);
            assert!(
                angle <= OCT_NORMAL_STEP_RAD,
                "{n:?} -> {back:?}: {angle} rad"
            );
            widest = widest.max(angle);
            // The plain rounding of the folded coordinates, for the comparison the precise
            // search must win.
            let sum = n[0].abs() + n[1].abs() + n[2].abs();
            let (mut x, mut y) = (n[0] / sum, n[1] / sum);
            if n[2] < 0.0 {
                let (fx, fy) = ((1.0 - y.abs()) * sign_of(x), (1.0 - x.abs()) * sign_of(y));
                x = fx;
                y = fy;
            }
            widest_plain = widest_plain.max(angle_between(n, oct_decode([snorm16(x), snorm16(y)])));
        }
        // MEASURED on this chunk: 4.2e-5 rad precise against 6.3e-5 plain.
        eprintln!(
            "the packed normal's widest angle over a real chunk: {widest:.3e} rad (plain rounding \
             {widest_plain:.3e})"
        );
        // Strict: the precise search wins on this chunk by a third (4.2e-5 against 6.3e-5).
        assert!(widest < widest_plain, "{widest} >= {widest_plain}");
        assert!(widest <= OCT_NORMAL_STEP_RAD, "{widest} rad");
        assert_eq!(oct_encode([0.0, 0.0, 0.0]), [0, 0]);
        assert_eq!(oct_decode([0, 0]), [0.0, 0.0, 1.0]);
        assert_eq!(oct_encode([0.0, 0.0, -1.0]), [i16::MAX, i16::MAX]);
        assert_eq!(snorm16(2.0), i16::MAX);
        assert_eq!(snorm16(-2.0), -i16::MAX);
    }

    /// THE VERTEX'S RADIAL (step 5): every vertex's narrowed radial stands within
    /// `RADIAL_STEP_RAD` of the exact one, and a skirt vertex carries its top's.
    #[test]
    fn the_narrowed_radial_stands_within_its_step_of_the_exact_one() {
        let body = home_planet();
        let g = geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1");
        assert_eq!(g.radials.len(), g.vertices.len());
        let mut widest = 0.0f64;
        for i in 0..g.vertices.len() {
            let exact = g.radial(i);
            let r = g.radials[i];
            let narrowed =
                DVec3::new(f64::from(r[0]), f64::from(r[1]), f64::from(r[2])).normalize();
            widest = widest.max(exact.dot(narrowed).clamp(-1.0, 1.0).acos());
        }
        assert!(widest <= RADIAL_STEP_RAD, "{widest} rad");
    }

    #[test]
    fn the_morph_targets_stand_on_the_coarser_surface_and_the_top_rung_is_its_own() {
        let body = home_planet();
        let key = surface_key(&body, 0, 300, 700);
        let g = geometry_of(&body, key).expect("the golden chunk");
        assert_eq!(g.morph_m.len(), g.vertices.len());
        // THE BOUNDS hold every vertex, every target and every sunk position, on a chunk that
        // sinks (rung 1) as on one that does not; the worker's box is the measured box; an empty
        // geometry has none.
        for g in [
            &g,
            &geometry_of(&body, surface_key(&body, 1, 150, 350)).expect("rung 1"),
        ] {
            let (lo, hi) = g.bounds;
            assert_eq!(g.bounds, g.measure_bounds());
            for (i, v) in g.vertices.iter().enumerate() {
                let t = g.morph_target(i);
                let s = g.sink_of(i);
                let mut k = 0;
                while k < 3 {
                    assert!(lo[k] <= v[k]);
                    assert!(v[k] <= hi[k]);
                    assert!(lo[k] <= t[k]);
                    assert!(t[k] <= hi[k]);
                    assert!(lo[k] <= v[k] - s[k]);
                    assert!(v[k] - s[k] <= hi[k]);
                    k += 1;
                }
            }
        }
        let empty = ChunkGeometry {
            key,
            origin_m: g.origin_m,
            morph_m: Vec::new(),
            morph_fallbacks: 0,
            morph_seam: 0,
            sink_m: 0.0,
            vertices: Vec::new(),
            normals: Vec::new(),
            packed_normals: Vec::new(),
            morph_normals: Vec::new(),
            radials: Vec::new(),
            triangles: Vec::new(),
            bounds: ([0.0; 3], [0.0; 3]),
            skirt_start: 0,
        };
        assert_eq!(empty.measure_bounds(), ([0.0; 3], [0.0; 3]));
        assert!(empty.morph_targets().is_empty());
        // The gap between a vertex and its target is radial and within the octave dropped between
        // rung 0 and rung 1 — the recipe's own bound — plus a cell of each rung for the two
        // extractors' placement, and at least one target differs.
        let bound = (body.dropped_bound_m(1) - body.dropped_bound_m(0)) + 3.0;
        let o = vd_core::glam::DVec3::from_array(g.origin_m);
        let mut moved = 0;
        let mut on_surface = 0;
        for (v, m) in g.vertices.iter().zip(g.morph_targets().iter()) {
            let pv =
                o + vd_core::glam::DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            let pm =
                o + vd_core::glam::DVec3::new(f64::from(m[0]), f64::from(m[1]), f64::from(m[2]));
            let radial = pv.normalize();
            let gap = pm - pv;
            let along = gap.dot(radial);
            let across = (gap - radial * along).length();
            assert!(across < 0.01, "{across} m across the radial");
            // The bound holds for a SURFACE vertex; a vertex of a sealed cave under the surface
            // (the world has them) morphs to the parent's surface above it, as far as that is.
            let surface = vd_terrain::height::height_m(&body, [radial.x, radial.y, radial.z], 0);
            let on = (pv.length() - surface).abs() < 2.0;
            on_surface += i32::from(on);
            // Branchless (HR5): a cave vertex's limit is beyond any target.
            let limit = bound + 0.01 + (1.0 - f64::from(u8::from(on))) * 1.0e9;
            assert!(along.abs() <= limit, "{along} m along, bound {bound}");
            moved += i32::from(on & (along.abs() > 1e-3));
        }
        assert!(on_surface > 100, "{on_surface}");
        assert!(moved > 0);
        let top = body.ladder().rungs - 1;
        // The top rung is ONE CHUNK a face edge since 2026-09-15, so the face's only column is (0, 0).
        let top_key = surface_key(&body, top, 0, 0);
        let g = geometry_of(&body, top_key).expect("the top chunk");
        assert!(g.morph_m.iter().all(|m| *m == 0.0));
        assert_eq!(g.morph_targets(), g.vertices);
    }

    #[test]
    fn the_eye_surface_reads_the_recipe_under_the_eye_and_refuses_the_centre() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
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
            vd_terrain::height::biome_at(&body, dir, surface)
        );
        assert_eq!(eye_surface(&body, [0.0, 0.0, 0.0]), None);
        assert_eq!(eye_surface(&body, [f64::NAN, 0.0, 0.0]), None);
    }

    #[test]
    fn the_ruler_stands_on_the_drawn_ground_ahead_and_is_absent_for_a_sky_ray() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let up = vd_core::glam::DVec3::from_array(d);
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        let eye = up * (surface + 3.4);
        // A level direction tilted 8° down, like the ground picture's nose.
        let level = up.cross(vd_core::glam::DVec3::Z).normalize();
        let tilt = 8.0_f64.to_radians();
        let nose = (level * tilt.cos() - up * tilt.sin()).normalize();
        // ★ THE REACH IS 1 200 m, RE-MEASURED 2026-09-16 on slice 8a stage 2 (the ridged band).
        // The eye stands 3.4 m over its own ground and the nose points 8° down, so the ray falls
        // 55.7 m over 400 m — which used to be enough. The ridge lifted the crest the eye stands on
        // and steepened the ground ahead of it, so the ray now meets the surface at 1 113.8 m, and a
        // 400 m march finds nothing. MEASURED: 400, 600 and 800 m answer NOTHING; 1 200 m and every
        // longer reach answer the same 1 113.8 m hit. The ruler is the picture's own size reference,
        // so a hill the eye looks over is exactly the case it must answer.
        let reach_m = 1_200.0;
        let ruler = ruler_on_surface(&body, eye.to_array(), nose.to_array(), 0, reach_m)
            .expect("the ray meets the ground");
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), nose.to_array(), 0, reach_m),
            Some(ruler)
        );
        // The ball hovers one radius clear of the rung-0 surface at its own hit direction: its
        // centre stands two radii over the recipe there.
        let c = vd_core::glam::DVec3::from_array(ruler.centre_m);
        let cd = c.normalize();
        let there = vd_terrain::height::height_m(&body, [cd.x, cd.y, cd.z], 0);
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
        assert!(ruler.distance_m < reach_m, "{ruler:?}");
        // The size law: the larger of the angular size AT THE HIT and half a cell.
        let hit = c - cd * (2.0 * ruler.radius_m);
        let by_angle = (hit - eye).length() * RULER_TAN_HALF_ANGLE;
        assert!(
            (ruler.radius_m - by_angle.max(0.5)).abs() < 1e-3,
            "{ruler:?} vs {by_angle}"
        );
        // A ray to the sky meets nothing within reach — and a reach of the whole ladder is cut to
        // the march's own, so the answer comes in thousands of steps, not a half-million.
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), up.to_array(), 0, 400.0),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, eye.to_array(), up.to_array(), 0, 466_000.0),
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
        let there9 = vd_terrain::height::height_m(&body, [dd.x, dd.y, dd.z], 9);
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

#[cfg(test)]
mod claim_tests {
    use super::*;
    use vd_core::pose::RealmId;
    use vd_seed::bend::Face;

    fn planet() -> RealmId {
        RealmId::Planet(7)
    }

    /// The guard of a claim that builds; `None` for a hit (both arms run in the test below).
    fn guard_of(claim: Claim<'_>) -> Option<ClaimGuard<'_>> {
        match claim {
            Claim::Build(guard) => Some(guard),
            Claim::Hit(_) => None,
        }
    }

    fn key_at(body: &BodyDefinition, x: i32) -> ChunkKey {
        ChunkKey {
            face: Face::PosX,
            rung: 1,
            x,
            y: 5,
            z: vd_terrain::digest::surface_chunk_z(body, Face::PosX, 1, x, 5),
        }
    }

    /// A waiter that hands a message back just before it claims, so the test knows it is at the
    /// door; then the claimant parks a moment (the lib never sleeps — its clippy rule — and this
    /// test parks only to give the waiter its way to the condition variable).
    fn waiter(
        cache: &Arc<ParentCache>,
        key: ChunkKey,
    ) -> (
        std::sync::mpsc::Receiver<()>,
        std::thread::JoinHandle<Option<usize>>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cache = Arc::clone(cache);
        let handle = std::thread::spawn(move || {
            tx.send(()).expect("the test listens");
            match cache.claim(planet(), key) {
                Claim::Hit(m) => Some(m.triangle_count()),
                Claim::Build(guard) => {
                    guard.finish(None);
                    None
                }
            }
        });
        (rx, handle)
    }

    /// SINGLE FLIGHT (ruling V15): a parent claimed by one reader is waited for by the next, and
    /// read as a hit when it lands; an abandoned claim hands the build to the waiter; a guard
    /// dropped without a finish abandons too (a panic's path); a forgotten realm refuses a late
    /// landing.
    #[test]
    fn a_claimed_parent_is_built_once_and_a_waiter_reads_the_landing() {
        let body = vd_terrain::home::home_planet();
        let cache = Arc::new(ParentCache::with_capacity(4));
        let key = key_at(&body, 3);
        // The first claim is a build.
        let guard = guard_of(cache.claim(planet(), key)).expect("a cold cache builds");
        assert!(format!("{guard:?}").contains("ClaimGuard"));
        assert_eq!(
            cache.stats(),
            ParentStats {
                hits: 0,
                builds: 1,
                waits: 0
            }
        );
        // A second reader waits on the claim; it lands and the waiter reads a hit.
        let (at_the_door, waiting) = waiter(&cache, key);
        at_the_door.recv().expect("the waiter reports");
        std::thread::park_timeout(std::time::Duration::from_millis(30));
        let mesh = Arc::new(ParentMesh::build(&body, key).expect("in the ladder"));
        assert!(mesh.bytes() > mesh.triangle_count() * 12);
        guard.finish(Some(Arc::clone(&mesh)));
        assert_eq!(
            waiting.join().expect("the waiter returns"),
            Some(mesh.triangle_count())
        );
        assert_eq!(cache.len(), 1);
        let after = cache.stats();
        assert_eq!((after.hits, after.builds), (1, 1));
        assert!(after.waits <= 1);
        // A hit is a Debug too, and it carries no guard.
        let hit = cache.claim(planet(), key);
        assert!(format!("{hit:?}").contains("Hit"));
        assert!(guard_of(hit).is_none());
        // An abandoned claim: the waiter wakes with nothing kept and claims the build itself.
        let other = key_at(&body, 4);
        let guard = guard_of(cache.claim(planet(), other)).expect("a miss builds");
        let (at_the_door, waiting) = waiter(&cache, other);
        at_the_door.recv().expect("the waiter reports");
        std::thread::park_timeout(std::time::Duration::from_millis(30));
        guard.finish(None);
        assert_eq!(waiting.join().expect("the waiter returns"), None);
        assert_eq!(cache.len(), 1);
        // A guard dropped without a finish abandons the claim (the panic's path).
        let third = key_at(&body, 5);
        drop(guard_of(cache.claim(planet(), third)).expect("a miss builds"));
        assert!(guard_of(cache.claim(planet(), third)).is_some());
        assert_eq!(cache.len(), 1);
        // A forgotten realm refuses the landing of a claim from before the forget.
        let stale = guard_of(cache.claim(planet(), other)).expect("a miss builds");
        cache.forget(planet());
        assert!(cache.is_empty());
        stale.finish(Some(Arc::clone(&mesh)));
        assert!(cache.is_empty());
        // `get` after the forget builds and keeps under the new epoch.
        assert!(cache.get(planet(), &body, other).is_some());
        assert_eq!(cache.len(), 1);
        assert_eq!(
            BuildCount::default(),
            BuildCount {
                chunks: 0,
                nanos: 0,
                peak_nanos: 0,
                peak_key: None
            }
        );
    }
}

#[cfg(test)]
mod parent_shrink_tests {
    use super::*;
    use vd_core::pose::RealmId;

    fn planet() -> RealmId {
        RealmId::Planet(7)
    }

    /// The parent mesh's packing (step 5): a parent of the home planet holds its triangles at 16
    /// bits and fewer bytes than the old form; a synthetic mesh past 65 535 vertices holds them at
    /// 32; a cell outside the box has no bucket; the bucket table indexes every triangle of a
    /// cell exactly as the map did.
    /// THE BYTE BUDGET (refutation of step 5's second half, finding 2): the cache bounds the
    /// bytes it holds, never fewer than the floor; a replaced key counts once; a forgotten realm
    /// returns its bytes.
    #[test]
    fn the_byte_budget_bounds_the_held_bytes_and_the_floor_keeps_a_working_set() {
        let body = vd_terrain::home::home_planet();
        let key = |x: i32| ChunkKey {
            face: vd_seed::bend::Face::PosX,
            rung: 1,
            x,
            y: 5,
            z: vd_terrain::digest::surface_chunk_z(&body, vd_seed::bend::Face::PosX, 1, x, 5),
        };
        let mesh = Arc::new(ParentMesh::build(&body, key(3)).expect("in the ladder"));
        let one = mesh.bytes();
        let cache = ParentCache::with_capacity(100);
        assert_eq!(cache.budget_bytes(), usize::MAX);
        assert_eq!(cache.held_bytes(), 0);
        // Two and a half meshes of budget: the third insert trims the oldest.
        cache.set_budget_bytes(one * 5 / 2, 1);
        assert_eq!(cache.capacity(), usize::MAX);
        cache.insert(planet(), key(3), Arc::clone(&mesh));
        cache.insert(planet(), key(4), Arc::clone(&mesh));
        assert_eq!((cache.len(), cache.held_bytes()), (2, 2 * one));
        cache.insert(planet(), key(5), Arc::clone(&mesh));
        assert_eq!((cache.len(), cache.held_bytes()), (2, 2 * one));
        assert!(cache.get(planet(), &body, key(3)).is_some());
        // A held key replaced counts once.
        cache.insert(planet(), key(5), Arc::clone(&mesh));
        assert_eq!((cache.len(), cache.held_bytes()), (2, 2 * one));
        // Under the budget the floor holds: a zero budget with a floor of two keeps two.
        cache.set_budget_bytes(0, 2);
        assert_eq!((cache.len(), cache.held_bytes()), (2, 2 * one));
        cache.set_budget_bytes(0, 0);
        assert_eq!((cache.len(), cache.held_bytes()), (1, one));
        // A forgotten realm returns its bytes.
        cache.forget(planet());
        assert_eq!((cache.len(), cache.held_bytes()), (0, 0));
        // The count bound still trims on its own when it is the smaller one.
        let counted = ParentCache::with_capacity(1);
        counted.insert(planet(), key(3), Arc::clone(&mesh));
        counted.insert(planet(), key(4), Arc::clone(&mesh));
        assert_eq!((counted.len(), counted.held_bytes()), (1, one));
    }

    #[test]
    fn a_parent_mesh_packs_its_triangles_and_indexes_its_buckets_flat() {
        let body = vd_terrain::home::home_planet();
        let key = ChunkKey {
            face: vd_seed::bend::Face::PosX,
            rung: 1,
            x: 3,
            y: 5,
            z: vd_terrain::digest::surface_chunk_z(&body, vd_seed::bend::Face::PosX, 1, 3, 5),
        };
        let mesh = ParentMesh::build(&body, key).expect("in the ladder");
        assert!(format!("{:?}", mesh.triangles).contains("Narrow"));
        // ★ THE BOUND IS THE PACKING'S, NOT THE TERRAIN'S (re-stated 2026-09-15; the INDEX split out
        // 2026-09-17, slice 8a stage 3). A 400 kB ceiling was a statement about this box's own
        // ground: on the extended ladder's body the box is cave-riddled and holds tens of thousands
        // of triangles, while its neighbours along the face hold 8 000 to 11 000.
        //
        // What the packing claims is THIRTY-TWO BYTES A TRIANGLE for the mesh itself — the
        // positions, the normals and the 16-bit triangle rows — BESIDE the bucket index, which is a
        // fixed 4 097-word table plus one word per bucket entry and belongs to the GRID, not to the
        // triangles. Stating the two together made the claim depend on the triangle COUNT: this box
        // held 7 992 triangles on stage 3's ground and 255 796 bytes, which is 32.007 a triangle
        // with the index folded in and 29.96 without it. The index is now subtracted by name, and
        // the claim is the packing's again.
        //
        // The readings are taken BEFORE the assert: an argument inside the message is only evaluated
        // when the assert fails, which is a region no green run can cover (HR5).
        let index_bytes = (mesh.bucket_start.capacity() + mesh.bucket_tris.capacity())
            * std::mem::size_of::<u32>();
        let mesh_bytes = mesh.bytes() - index_bytes;
        let mesh_tris = mesh.triangle_count();
        assert!(
            mesh_bytes < mesh_tris * 32,
            "{mesh_bytes} bytes for {mesh_tris} triangles, beside a {index_bytes}-byte index"
        );
        assert_eq!(
            mesh.bucket_start.len(),
            (PARENT_CELLS * PARENT_CELLS) as usize + 1
        );
        assert_eq!(
            *mesh.bucket_start.last().expect("the end"),
            mesh.bucket_tris.len() as u32
        );
        // Every triangle stands in at least one bucket, and every bucket entry names a triangle.
        assert!(
            mesh.bucket_tris
                .iter()
                .all(|i| (*i as usize) < mesh.triangle_count())
        );
        assert!(mesh.bucket_tris.len() >= mesh.triangle_count());
        assert_eq!(parent_bucket(-2, 0), None);
        assert_eq!(parent_bucket(0, 63), None);
        assert_eq!(parent_bucket(-1, -1), Some(0));
        assert_eq!(parent_bucket(62, 62), Some(64 * 64 - 1));
        // A hit on the surface through a cell the mesh spans (some cell of the grid answers with
        // the vertex's own radius), none through a cell outside the grid.
        let p = mesh.position(0);
        let dir = p.normalize();
        assert!(mesh.radial_hit_m(-2, -2, dir, p.length()).is_none());
        let mut hit = false;
        let mut a = -1;
        while a <= 62 {
            let mut b = -1;
            while b <= 62 {
                hit |= mesh
                    .radial_hit_m(a, b, dir, p.length())
                    .is_some_and(|m| (m - p.length()).abs() < 1.0e-3);
                b += 1;
            }
            a += 1;
        }
        assert!(hit, "no cell of the grid answers the ray through vertex 0");
        // The wide form: a synthetic mesh of one triangle whose vertex count passes 16 bits.
        let wide = ParentTriangles::pack(vec![[0, 1, 70_000]], 70_001);
        assert!(format!("{wide:?}").contains("Wide"));
        assert_eq!(wide.len(), 1);
        assert_eq!(wide.get(0), [0, 1, 70_000]);
        assert_eq!(wide.bytes(), 12);
        let narrow = ParentTriangles::pack(vec![[0, 1, 2]], 3);
        assert_eq!(narrow.get(0), [0, 1, 2]);
        assert_eq!(narrow.bytes(), 6);
        assert!(format!("{narrow:?}").contains("Narrow"));
    }
}
