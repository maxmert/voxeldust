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
use vd_terrain::artifact::ZField;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::extract::{extract, extract_all_edges};
use vd_terrain::lattice::sample_box;
use vd_terrain::position::vertex_position_m;

use crate::artifact_book::{ArtifactCache, FieldPick, SharedField};

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

/// ★ THE STAND-IN'S SINK for a chunk of `rung` (slice R1c): the sink of the rung that stands in
/// for it while it is late — the next coarser one — or nought at the top rung, which has none and
/// enters whole ([`crate::arrival::arrival_stamp`]).
#[must_use]
pub fn stand_in_sink_m(body: &BodyDefinition, rung: u8) -> f64 {
    let top = body.ladder().rungs.saturating_sub(1);
    let has = u8::from(rung < top);
    f64::from(has) * sink_m(body, rung.saturating_add(1).min(top))
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
    /// ★ THE STAND-IN'S SINK (the far-water design §10, slice R1c): the next coarser rung's sink,
    /// nought at the top rung — where the arrival ramp starts a late chunk
    /// ([`crate::arrival`]), so the chunk's boxes hold that start too.
    pub stand_in_sink_m: f32,
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
    /// ★ THE SURFACE OF EACH VERTEX (ruling B75 item 4; [`crate::surface::vertex_rock`]): the bare rock's share in 1/255
    /// at the vertex's own slope, then at its morph target's (the parent's shade there), then two spare bytes. The shader
    /// blends the two across the fade-out band as it blends the shade, so the rock hands over with the shape.
    pub surface: Vec<[u8; 4]>,
    /// ★ THE HEIGHT OF EACH VERTEX over the ladder radius, metres (ruling B75 item 4): what the drawing holds against the
    /// realm's snow line; its morph target stands `morph_m` higher along the same radial.
    pub heights: Vec<f32>,
    /// The extractor's triangles, unchanged.
    pub triangles: Vec<[u32; 3]>,
    /// Where the skirt vertices begin: every vertex before this index stands on the surface (one
    /// per surface cell, the extractor's own), every one from it hangs under an edge. The far-rung
    /// splat (D8-8's measurement) draws the surface ones only.
    pub skirt_start: u32,
    /// ★ THE FIELD THE CHUNK WAS BUILT ON (the coast flight's hole instrument, 2026-09-20): the
    /// pyramid level it read (zero for the rows, the tiles), or [`NO_FIELD`] for the recipe's own
    /// relief or a box sampled elsewhere (the card).
    pub field_level: u32,
    /// ★ THE ARTIFACT THE CHUNK WAS BUILT ON (the coast flight's stale builds, 2026-09-20): the
    /// head's digest, or `None` for the recipe's own relief. The lane refuses a chunk built on
    /// another artifact than the one it holds for the realm when the chunk lands.
    pub built_on: Option<[u64; 2]>,
    /// ★ BUILT ON THE ROUGH GROUND (ruling B86 item 7, the safety net): the chunk, or a parent its morph read, stood on
    /// the field under the fine ground because a fine tile had not landed. The lane builds it again once every tile
    /// it and its parents read is here ([`ChunkLane::upgradable`]).
    pub rough: bool,
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
/// 4, morph normal 4, morph 4 — ruling V18's form with item 20 — and the surface 4 and the height 4,
/// ruling B75 item 4; ★ no radial since ruling B89: the shaders read it from the body's centre); the
/// exact rungs cost eight more.
pub const UPLOAD_VERTEX_BYTES: u64 = 32;

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

    /// ★ THE ARRIVAL RAMP'S DEEPEST START for vertex `i` (slice R1c): its morph target sunk by
    /// the whole stand-in's sink — where a late chunk's vertex starts when it lands whole.
    #[must_use]
    pub fn ramp_start(&self, i: usize) -> [f32; 3] {
        let t = self.morph_target(i);
        let s = self.radial(i) * f64::from(self.stand_in_sink_m);
        [t[0] - s.x as f32, t[1] - s.y as f32, t[2] - s.z as f32]
    }

    /// The box over every vertex, its morph target, its sunk position and its ramp's deepest
    /// start, as the worker builds it into `bounds`. Zero for an empty geometry.
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
            take(self.ramp_start(i));
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
    /// ★ THE ARTIFACT the realm shipped (slice 8c stage C4c), shared by pointer: the build reads
    /// its field for this rung and for the parents' rung. `None` for a realm that shipped none,
    /// whose chunks are the recipe's own coarse relief — the pre-artifact shape, never a shipped
    /// path once every solid body ships. A job with an artifact is the CPU builders' alone: the
    /// card holds no `Z` (ruling F9, the card builder parked).
    pub artifact: Option<Arc<ArtifactCache>>,
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
            geometry_with(
                &job.body,
                job.artifact.as_deref(),
                job.realm,
                job.key,
                &job.parents,
            )
            .map(|geometry| ChunkReady {
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
    /// ★ Built on the rough ground (B86 item 7): a fine reader builds it again once the parent's fine tiles land.
    rough: bool,
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
    pub fn build(
        body: &BodyDefinition,
        artifact: Option<&ArtifactCache>,
        key: ChunkKey,
    ) -> Option<ParentMesh> {
        let field = field_of(body, artifact, key)?;
        let rough = artifact.zip(body.macro_lattice()).is_some_and(|(a, lattice)| a.is_rough(&lattice, key));
        let samples = sample_box(body, field.as_deref().map(|f| f as &dyn ZField), key)?;
        // The parent's surface with its sealed hollows as rock, as the drawn chunk reads its own.
        let samples = sealed_hollows(body, &samples).unwrap_or(samples);
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
            rough,
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
        artifact: Option<&ArtifactCache>,
        key: ChunkKey,
    ) -> Option<Arc<ParentMesh>> {
        // ★ A ROUGH MESH IS NOT A FINE READER'S (B86 item 7): where the parent's fine tiles are all here now, a mesh
        // built on the rough ground before is built again.
        let fine = artifact
            .zip(body.macro_lattice())
            .is_some_and(|(a, lattice)| !a.is_rough(&lattice, key));
        match self.claim_for(realm, key, fine) {
            Claim::Hit(found) => Some(found),
            Claim::Build(guard) => {
                let built = ParentMesh::build(body, artifact, key).map(Arc::new);
                guard.finish(built.clone());
                built
            }
        }
    }

    /// Claim `key`: a hit returns at once (and refreshes its use); a miss another reader builds
    /// WAITS, then reads a hit; a miss nobody builds is claimed, and the guard must `finish`.
    pub fn claim(&self, realm: RealmId, key: ChunkKey) -> Claim<'_> {
        self.claim_for(realm, key, false)
    }

    /// [`ParentCache::claim`] for a reader that wants the parent on the fine ground when `fine`: a held mesh built on
    /// the rough ground is then a miss, built again in its place.
    pub fn claim_for(&self, realm: RealmId, key: ChunkKey, fine: bool) -> Claim<'_> {
        let mut store = self.lock();
        let mut waited = false;
        loop {
            let usable = store.map.get(&(realm, key)).is_some_and(|e| !(fine & e.mesh.rough));
            if usable {
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

/// THE PARENT HIT NEAREST A RADIUS along a vertex's radial: over the parents the vertex reads (none
/// on the face seam, whose vertex reads the field from both faces alike), the radial's crossing
/// with a parent triangle nearest `near_m`, with the parent's shade there; a tie goes to the lower.
fn nearest_parent_hit(
    parent_meshes: &[Arc<ParentMesh>],
    at_seam: bool,
    key: ChunkKey,
    v: [i16; 3],
    dir: DVec3,
    near_m: f64,
) -> Option<(f64, DVec3)> {
    parent_meshes
        .iter()
        .filter(|_| !at_seam)
        .fold(None, |best: Option<(f64, DVec3)>, pm| {
            let (a, b) = parent_cell(key, pm.key(), v);
            let hit = pm.radial_hit(a, b, dir, near_m);
            match (best, hit) {
                (Some((b, _)), Some((h, s)))
                    if ((h - near_m).abs(), h) < ((b - near_m).abs(), b) =>
                {
                    Some((h, s))
                }
                (None, Some(h)) => Some(h),
                (b, _) => b,
            }
        })
}

/// THE COARSER FIELD'S OWN NORMAL along a direction at a rung (2026-09-24): the surface's slope read
/// one coarser cell to each side along two tangents, the normal turned to face out along the
/// radial. `None` where the field holds no row there (a tile not here).
fn field_normal(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    dir: DVec3,
    rung: u8,
) -> Option<DVec3> {
    let step = f64::from(vd_seed::ladder::cell_m(rung)) / body.radius_m();
    let t1 = dir.any_orthonormal_vector();
    let t2 = dir.cross(t1);
    let at = |d: DVec3| -> Option<DVec3> {
        let u = d.normalize();
        coarser_field_m(body, artifact, u.to_array(), rung).map(|h| u * h)
    };
    // All four sides read, or none: one combined answer, no early exit per side.
    at(dir + t1 * step)
        .zip(at(dir - t1 * step))
        .zip(at(dir + t2 * step).zip(at(dir - t2 * step)))
        .map(|((a1, a2), (b1, b2))| {
            let n = (a1 - a2).cross(b1 - b2).normalize_or_zero();
            n * n.dot(dir).signum()
        })
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
    geometry_with(body, None, RealmId::Planet(0), key, &ParentCache::default())
}

/// ★ THE FIELD A CHUNK READS (slice 8c stage C4c): `Some(None)` without an artifact (the recipe's
/// own coarse relief), `Some(Some(field))` when the artifact's field for this chunk is whole, and
/// `None` while it is not — the build makes nothing, and the coarser rung stands (ruling F9). A
/// body with no macro lattice (too small for one) reads the recipe whatever it was shipped.
#[must_use]
pub fn field_of(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    key: ChunkKey,
) -> Option<Option<SharedField>> {
    let Some(artifact) = artifact else {
        return Some(None);
    };
    let Some(lattice) = body.macro_lattice() else {
        return Some(None);
    };
    match artifact.field_for(&lattice, key) {
        FieldPick::Ready(field) | FieldPick::Rough(field) => Some(Some(field)),
        FieldPick::Waiting => None,
    }
}

/// The coarser surface along a direction, for a vertex with no parent triangle on its radial: the
/// artifact's field at that rung (`None` where the field holds no row there), or the recipe's own
/// relief without an artifact.
fn coarser_field_m(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    dir: [f64; 3],
    rung: u8,
) -> Option<f64> {
    let Some(artifact) = artifact else {
        return Some(vd_terrain::height::height_m(body, dir, rung));
    };
    let Some(lattice) = body.macro_lattice() else {
        return Some(vd_terrain::height::height_m(body, dir, rung));
    };
    let field = artifact.field_at_rung(&lattice, rung)?;
    vd_terrain::height::height_field_m(body, &*field, dir, rung)
}

/// ★ THE SEALED HOLLOWS ARE DRAWN AS ROCK (ruling B22 step 2; the owner, 2026-09-28: *"do we need
/// to render what's underneath the surface? … there are no caves entrances visible"*).
///
/// The recipe carves its caverns and tubes only in a depth band under each column's own surface,
/// never shallower than the body's drawn minimum (8 to 30 m, `vd_terrain::body::Caves`), so every
/// hollow has rock over it and no entrance. From any eye outside the rock its walls are never seen,
/// yet they were meshed and drawn like the ground: MEASURED at the owner's stand (the covered share,
/// `VD_TERRAIN_COVERED=1`), 41 % of all the ground's triangles, 86 % at rung 2 — and the builders'
/// dearest vertices, because a cave vertex meets no parent triangle and reads the coarser field five
/// times. So the DRAWN mesh takes a hollow cell with rock over it in its own column as rock. The
/// world keeps its caves: the cells the server holds and collides against are the recipe's, and
/// only the copy the extractor reads here is sealed.
///
/// The test is exact for today's ground — a height field with hollows in a band under it, no
/// overhang — and errs only toward drawing: a hollow whose roof lies in the chunk above, beyond
/// the box, keeps its walls (they are sealed all the same, and hidden). Overhangs (8f) and a dug
/// opening (the edit system) will need the hollow's reach to the open air instead
/// (`docs/design/DEFERRED.md`, the speed step).
///
/// Example. Twelve metres under the belt's plain a tube four metres wide runs through a rung-2
/// chunk. The box sees rock over each of its cells, so the extractor meets no surface there and
/// the chunk is built and drawn as the plain alone; the shard's own cells still hold the tube.
#[must_use]
pub fn sealed_hollows(
    body: &BodyDefinition,
    samples: &vd_terrain::lattice::SampleBox,
) -> Option<vd_terrain::lattice::SampleBox> {
    let rung = samples.key.rung;
    let carves = vd_terrain::carve::tubes_carve_at(body, rung)
        | vd_terrain::carve::caverns_carve_at(body, rung);
    let hollow = carves & has_sealed_hollow(samples);
    hollow.then(|| {
        let mut sealed = samples.clone();
        seal_hollows(&mut sealed);
        sealed
    })
}

/// The box's columns, each from its top cell down: `(a, b)` over the box's cells, halo included.
const BOX_LO: i32 = -vd_terrain::lattice::HALO;
const BOX_HI: i32 = CHUNK_EDGE as i32;

/// Whether some column of the box holds a hollow cell (air) under a rock cell of its own.
fn has_sealed_hollow(samples: &vd_terrain::lattice::SampleBox) -> bool {
    use vd_terrain::lattice::SampleBox;
    (BOX_LO..=BOX_HI).any(|b| {
        (BOX_LO..=BOX_HI).any(|a| {
            let mut rock_over = false;
            (BOX_LO..=BOX_HI).rev().any(|c| {
                let air = samples.cells[SampleBox::index(a, b, c)].gap >= 0;
                let sealed = air & rock_over;
                rock_over |= !air;
                sealed
            })
        })
    })
}

/// Every hollow cell under a rock cell of its own column becomes rock just inside the surface
/// (gap −1), of the stratum over it.
fn seal_hollows(samples: &mut vd_terrain::lattice::SampleBox) {
    use vd_terrain::lattice::SampleBox;
    for b in BOX_LO..=BOX_HI {
        for a in BOX_LO..=BOX_HI {
            let mut rock: Option<vd_terrain::chunk::Cell> = None;
            for c in (BOX_LO..=BOX_HI).rev() {
                let i = SampleBox::index(a, b, c);
                let cell = samples.cells[i];
                match (cell.gap >= 0, rock) {
                    (true, Some(over)) => {
                        samples.cells[i] = vd_terrain::chunk::Cell {
                            stratum: over.stratum,
                            gap: -1,
                        };
                    }
                    (false, _) => rock = Some(cell),
                    (true, None) => {}
                }
            }
        }
    }
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
    artifact: Option<&ArtifactCache>,
    realm: RealmId,
    key: ChunkKey,
    parents: &ParentCache,
) -> Option<ChunkGeometry> {
    let field = field_of(body, artifact, key)?;
    let samples = sample_box(body, field.as_deref().map(|f| f as &dyn ZField), key)?;
    let mut geometry = geometry_from(body, artifact, realm, key, &samples, parents)?;
    geometry.field_level = field.as_deref().map_or(NO_FIELD, |f| f.level());
    geometry.built_on = artifact.map(|a| a.head.digest);
    geometry.rough |= artifact.zip(body.macro_lattice()).is_some_and(|(a, lattice)| a.is_rough(&lattice, key));
    Some(geometry)
}

/// THE GEOMETRY STEP of a chunk's build: the mesh, the vertices, the morph targets and the normals
/// from a box of cells somebody already sampled. The box is the recipe's own output, whoever
/// computed it.
#[must_use]
pub fn geometry_from(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    realm: RealmId,
    key: ChunkKey,
    samples: &vd_terrain::lattice::SampleBox,
    parents: &ParentCache,
) -> Option<ChunkGeometry> {
    // The drawn mesh reads the box with its sealed hollows as rock (see [`sealed_hollows`]).
    let sealed = sealed_hollows(body, samples);
    let samples = sealed.as_ref().unwrap_or(samples);
    let mesh = extract(samples);
    let half = (CHUNK_EDGE / 2) as i16 * vd_terrain::VERTEX_QUANTUM as i16;
    let origin = vertex_position_m(body, samples, [half, half, half]);
    let origin_m = [origin[0], origin[1], origin[2]];
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(mesh.vertices.len());
    let mut morph: Vec<f32> = Vec::with_capacity(mesh.vertices.len());
    let mut radials: Vec<[f32; 3]> = Vec::with_capacity(mesh.vertices.len());
    let mut heights: Vec<f32> = Vec::with_capacity(mesh.vertices.len());
    let ladder_radius_m = body.ladder().radius_m();
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
        .filter_map(|p| parents.get(realm, body, artifact, p))
        .collect();
    // ★ ROUGH (B86 item 7): built on a rough view, or a parent's mesh stood on the rough ground.
    let rough = artifact.is_some_and(ArtifactCache::is_rough_view) | parent_meshes.iter().any(|p| p.rough);
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
            // A hit farther than the two surfaces can stand apart is another surface (a cave
            // under this one): the field, then.
            let on_mesh = nearest_parent_hit(&parent_meshes, at_seam, key, *v, dir, len)
                .filter(|(h, _)| (h - len).abs() <= reach_m);
            let hit_shade = on_mesh.map(|(_, s)| s);
            let on_mesh = on_mesh.map(|(h, _)| h);
            // The coarser field, read only where no hit stands: the target then, and the
            // judge of what the vertex is — a SURFACE vertex with no parent triangle on its
            // radial is a fallback (counted; the gate bounds it), a vertex far under the field
            // is a cave's own and reads the field by nature.
            // With an artifact whose field holds no row on this radial (a tile not here), the
            // vertex keeps its own radius: no target, no morph, counted below as a fallback.
            let field = on_mesh.map_or_else(
                || coarser_field_m(body, artifact, [dir.x, dir.y, dir.z], coarser).unwrap_or(len),
                |_| len,
            );
            let missing = on_mesh.is_none() & !at_seam;
            fallbacks += u32::from(missing & ((field - len).abs() <= reach_m));
            // ★ THE COARSER SHADE AT A FIELD TARGET (2026-09-24; the owner, low over the belt's
            // highland: "the underneath of the first rung is visible when flying"). A vertex that
            // morphs onto the coarser FIELD — a cave's own roof, wall or floor, which the coarser
            // rung does not hold; a steep riser whose coarser surface stands past the reach; a
            // vertex on the face seam — lands on the coarser ground at the band's far edge, so it
            // must SHADE like that ground there: the parent triangle the radial meets nearest the
            // target, else the field's own normal at it. MEASURED before (`ray_speck`, the 494 m
            // stand): such vertices kept their OWN normals, and the caves of rung 0 (26 m to 62 m
            // under the ground) closed onto the ground lit −0.70 to +0.02 against the ground's
            // +0.40, winning the depth test in dark tears along the ring's far edge (950 m, and
            // rung 1's at 1.9 km). A vertex whose field holds no row keeps its own normal.
            // (A hit within the reach carries its own shade; the field is read only without one.)
            let shade = hit_shade.or_else(|| {
                nearest_parent_hit(&parent_meshes, at_seam, key, *v, dir, field)
                    .filter(|(h, _)| (h - field).abs() <= reach_m)
                    .map(|(_, s)| s)
                    .or_else(|| field_normal(body, artifact, dir, coarser))
            });
            shades.push(shade);
            on_mesh.unwrap_or(field)
        };
        // The target as one metre along the radial from the vertex, and the radial itself (step 5).
        morph.push((target_m - len) as f32);
        radials.push([dir.x as f32, dir.y as f32, dir.z as f32]);
        heights.push((len - ladder_radius_m) as f32);
    }
    let normals = smooth_normals(&vertices, &mesh.triangles);
    // ★ THE ROCK AT EACH VERTEX AND AT ITS TARGET (ruling B75 item 4): the own slope, and the slope of the parent's shade
    // where the vertex morphs onto a parent triangle — the top rung's vertex is its own target.
    let surface: Vec<[u8; 4]> = normals
        .iter()
        .zip(&radials)
        .enumerate()
        .map(|(i, (normal, radial))| {
            let own = crate::surface::vertex_rock(*normal, *radial);
            let target = shades.get(i).copied().flatten().map_or(own, |s| {
                crate::surface::vertex_rock([s.x as f32, s.y as f32, s.z as f32], *radial)
            });
            [own, target, 0, 0]
        })
        .collect();
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
    // ★ EVERY GROUND TRIANGLE IS DRAWN (2026-09-28, ruling B22): no water is drawn until the
    // water fill, so no triangle is cut. Example: the belt's sea floor, 6.9 km down, shows dry.
    let mut geometry = ChunkGeometry {
        key,
        origin_m,
        morph_m: morph,
        morph_fallbacks: fallbacks,
        morph_seam: seam,
        sink_m: sink_len as f32,
        stand_in_sink_m: stand_in_sink_m(body, key.rung) as f32,
        vertices,
        normals,
        packed_normals,
        morph_normals,
        surface,
        heights,
        triangles: mesh.triangles,
        radials,
        bounds: ([0.0; 3], [0.0; 3]),
        skirt_start: 0,
        field_level: NO_FIELD,
        built_on: None,
        rough,
    };
    let drop_m = skirt_drop_m(body, key.rung);
    let ceiling_m = f64::from(SKIRT_CELLS) * f64::from(vd_seed::ladder::cell_m(key.rung));
    add_skirts(&mut geometry, drop_m, ceiling_m);
    geometry.bounds = geometry.measure_bounds();
    Some(geometry)
}

/// The least a skirt hangs under a chunk's edge, in cells of its rung ([`skirt_drop_m`]).
pub const SKIRT_CELLS: u32 = 2;

/// ★ HOW DEEP A SKIRT HANGS (2026-09-24; the owner, flying very low at 29 km/s over the belt's
/// highland: *"I see cracks from time to time, only when I fly veeeery low"* — a long straight dark
/// line with light bits in it): as deep as the COARSER rung sinks under this one, plus one of this
/// rung's cells, and never less than [`SKIRT_CELLS`] of them; the top rung, which has no coarser rung,
/// hangs the least. Where a finer chunk is still building, the coarser rung shows in its place SUNK
/// under the finer ground by its sink (coarse before fine, SL8), so at the edge of a DRAWN finer chunk
/// the ground steps down to that sunk stand-in, and the skirt is what closes the step. MEASURED before
/// (`bins/examples/skirt_gap`, the 494 m eye over the highland): at two cells the skirt bottoms stood
/// OVER the sunk coarser ground by up to 1.50 m at rung 0, 3.37 m at rung 1, 7.98 m at rung 2 and
/// 25.36 m at rung 3 — the sink less the skirt, a crack a grazing eye sees into — while 400 to 2 450
/// wanted chunks stood urgent in the owner's flight. Branchless (HR5).
#[must_use]
pub fn skirt_drop_m(body: &BodyDefinition, rung: u8) -> f64 {
    let rungs = body.ladder().rungs;
    let cell = f64::from(vd_seed::ladder::cell_m(rung));
    let coarser = rung.saturating_add(1).min(rungs.saturating_sub(1));
    let has_coarser = f64::from(u8::from(rung.saturating_add(1) < rungs));
    (f64::from(SKIRT_CELLS) * cell).max((sink_m(body, coarser) + cell) * has_coarser)
}

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
fn add_skirts(g: &mut ChunkGeometry, drop_m: f64, ceiling_m: f64) {
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
        // ★ THE SKIRT HANGS INTO THE ROCK ITS EDGE STANDS ON (2026-09-24; the owner: will the
        // deeper skirt "not break the future terrain editing?"): `drop_m` DOWN under a surface
        // that faces the sky — deep enough to meet the sunk coarser rung where a neighbour is
        // still building — and only `ceiling_m` UP over one that faces the body's centre (a
        // cave's roof, a dug tunnel's ceiling), so it never hangs into open space as a curtain at a
        // chunk's border and cannot pierce a roof thicker than that. MEASURED with the deep drop
        // both ways (`skirt_gap`): ceiling skirts of rungs 2 and 3 rose up to 62 m, through shallow
        // caves' roofs toward the sky. The side is the edge's own triangle's, read along the radial
        // at the edge (a wall reads "down").
        let rise = (pb - pa)
            .cross(pc - pa)
            .dot((o + (pa + pb) * 0.5).normalize());
        let hang = if rise >= 0.0 { drop_m } else { -ceiling_m };
        // A skirt vertex is its top moved along the radial into that rock; its target moves by the
        // same, so the metre along the radial from the vertex to the target is its top's (step 5).
        let bottom = |i: u32| -> ([f32; 3], f32) {
            let p = at(g.vertices[i as usize]);
            let dir = (o + p).normalize();
            let q = p - dir * hang;
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
        g.surface.push(g.surface[a as usize]);
        g.surface.push(g.surface[b as usize]);
        g.heights.push(g.heights[a as usize]);
        g.heights.push(g.heights[b as usize]);
        // The strip faces away from the triangle's third corner: the winding whose normal points
        // from the edge's middle away from that corner.
        // The two windings are picked by index (a bool is 0 or 1), so both are one expression. The
        // first winding is taken when its normal already points away from the third corner; on a
        // ground triangle the test reads the second, and only a sheer wall's edge reads zero and
        // takes the first (DEFERRED: a sheer wall's skirt can face away from its wall).
        let n = (pb - pa).cross(at(qb) - pa);
        let away = (pa + pb) * 0.5 - pc;
        let back = usize::from(n.dot(away) < 0.0);
        skirt_tris.push([[a, b, b2], [a, b2, b]][back]);
        skirt_tris.push([[a, b2, a2], [a, a2, b2]][back]);
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
    /// ★ A request whose artifact field is not whole yet (slice 8c stage C4c): the level still
    /// assembling, or a tile the stencil reads not here. Not queued; asked again next frame; the
    /// coarser rung stands (ruling F9).
    pub awaiting_artifact: u64,
    /// ★ A realm whose artifact ARRIVED after chunks of it were built from the recipe's own
    /// relief (slice 8c stage C4c): those chunks and parents are dropped and rebuilt on the field.
    /// The one visible change is a recipe-to-artifact step at login, ledgered.
    pub artifact_rebuilds: u64,
    /// ★ CHUNKS HARVESTED WITH NO TRIANGLE (the coast flight's hole instrument, 2026-09-20): a
    /// chunk whose asked slice held no ground counts as built and drawn everywhere else, and the
    /// eye sees through the ground there. Named on the stamp too ([`ChunkLane::take_empty`]).
    pub empty_chunks: u64,
    /// ★ STALE BUILDS REFUSED (the coast flight, 2026-09-20): a chunk that landed built on
    /// another artifact than the lane holds for its realm — a job the workers were already
    /// running on the recipe's relief when the head arrived, whose result landed under the fresh
    /// request's pending entry and stood resident, empty, where the ground is. MEASURED: a third
    /// of the 717 hole columns at rest were such chunks. The lane drops it and the next request
    /// builds it on the field.
    pub stale_builds: u64,
    /// ★ CHUNKS HARVESTED ON THE ROUGH GROUND (ruling B86 item 7, the safety net): a fine tile they read had not
    /// landed; each is built again once the tiles are here.
    pub rough_builds: u64,
    /// ★ ROUGH CHUNKS ASKED AGAIN ON THE FINE GROUND, their tiles landed (the rough one stands until the fine lands).
    pub rough_upgrades: u64,
}

/// How many refused requests and empty chunks a frame NAMES on the stamp (the counters count them
/// all).
pub const NAMED_PER_FRAME: usize = 12;

/// One refused request, named: the realm, the key, and the tiles its stencil still waits for
/// (none when the head itself is not here at the stated digest).
pub type Refusal = (RealmId, ChunkKey, Vec<(u8, u32, u32)>);

/// The field level of a chunk built on no field (`ChunkGeometry::field_level`).
pub const NO_FIELD: u32 = u32::MAX;

/// How many slices either way of an empty chunk [`ChunkLane::hole_columns`] looks for a sibling
/// with ground: a column's span is the column bound over the ground, a few slices at most.
pub const HOLE_SIBLING_SLICES: i32 = 4;

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
    /// ★ THE ARTIFACTS the realms shipped (slice 8c stage C4c), by pointer, refreshed when the
    /// book's epoch for a realm changes.
    artifacts: BTreeMap<RealmId, Arc<ArtifactCache>>,
    /// ★ THE SNOW LINE of each realm whose artifact is whole and whose body has snow (ruling B75 item 4;
    /// [`crate::surface::snow_line_of`]): the drawing paints snow over it.
    snow_lines: BTreeMap<RealmId, vd_terrain::surface::SnowLine>,
    /// ★ THE ARTIFACT EACH REALM STATES it serves (the far-view ship): a realm that states a
    /// digest is built on that artifact and on nothing else — a request while the head at that
    /// digest is not here WAITS (counted), and the recipe's own relief is never drawn for it.
    expected: BTreeMap<RealmId, [u64; 2]>,
    resident: BTreeSet<(RealmId, ChunkKey)>,
    /// Every chunk still building, with the priority it was last asked at (ruling B22 step 2: a
    /// re-request at the same priority leaves the workers' queue alone).
    pending: BTreeMap<(RealmId, ChunkKey), u32>,
    /// Resident plus pending chunks per realm, so a release is a lookup, not a scan (SL9).
    held: BTreeMap<RealmId, usize>,
    counters: ChunkCounters,
    /// ★ THE REQUESTS REFUSED FOR THE ARTIFACT THIS FRAME, NAMED (the coast flight's hole
    /// instrument): drained by the renderer into the stamp each frame, capped.
    awaiting_frame: Vec<Refusal>,
    /// ★ EVERY MISSING TILE THIS FRAME, counted over every awaiting chunk, uncapped (2026-09-22):
    /// which tiles the wanted set really waits for, held against the tiles the book holds.
    missing_hist: BTreeMap<(RealmId, u8, u32, u32), u32>,
    /// ★ THE CHUNKS HARVESTED EMPTY THIS FRAME, NAMED with the field level they were built on:
    /// drained into the stamp, capped.
    empty_frame: Vec<(RealmId, ChunkKey, u32)>,
    /// ★ EVERY RESIDENT CHUNK THAT HOLDS NO TRIANGLE, with the field level it was built on: what
    /// [`ChunkLane::hole_columns`] reads to find a column none of whose resident slices holds
    /// ground.
    empty_resident: BTreeMap<(RealmId, ChunkKey), u32>,
    /// ★ THE RESIDENT CHUNKS BUILT ON THE ROUGH GROUND (B86 item 7), and those of them building again on the fine ground.
    rough: BTreeSet<(RealmId, ChunkKey)>,
    upgrading: BTreeSet<(RealmId, ChunkKey)>,
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
            snow_lines: BTreeMap::new(),
            refused: BTreeSet::new(),
            artifacts: BTreeMap::new(),
            expected: BTreeMap::new(),
            resident: BTreeSet::new(),
            pending: BTreeMap::new(),
            held: BTreeMap::new(),
            counters: ChunkCounters::default(),
            awaiting_frame: Vec::new(),
            missing_hist: BTreeMap::new(),
            empty_frame: Vec::new(),
            rough: BTreeSet::new(),
            upgrading: BTreeSet::new(),
            empty_resident: BTreeMap::new(),
            parents: Arc::new(ParentCache::default()),
        }
    }

    /// The requests refused for the artifact since the last take, named (capped at
    /// [`NAMED_PER_FRAME`]).
    pub fn take_awaiting(&mut self) -> Vec<Refusal> {
        std::mem::take(&mut self.awaiting_frame)
    }

    /// The missing tiles of this frame's awaiting chunks, each with how many chunks wait on it.
    pub fn take_missing_hist(&mut self) -> BTreeMap<(RealmId, u8, u32, u32), u32> {
        std::mem::take(&mut self.missing_hist)
    }

    /// The chunks harvested empty since the last take, named with their field level (capped at
    /// [`NAMED_PER_FRAME`]).
    pub fn take_empty(&mut self) -> Vec<(RealmId, ChunkKey, u32)> {
        std::mem::take(&mut self.empty_frame)
    }

    /// ★ THE HOLE COLUMNS of a realm (the coast flight's hole instrument, 2026-09-20): the chunk
    /// columns that hold a resident chunk and NO resident chunk with a triangle — the ground of
    /// that column is drawn by nobody, and the eye sees a hole where it should be. The count,
    /// and the first [`NAMED_PER_FRAME`] of them by their empty key. A walk of the EMPTY resident
    /// chunks (an instrument's cost, a few thousand at most) with a lookup per neighbouring slice,
    /// never a scan of the whole resident set.
    #[must_use]
    pub fn hole_columns(&self, realm: RealmId) -> (u64, Vec<(ChunkKey, u32)>) {
        let mut seen: BTreeSet<(vd_seed::bend::Face, u8, i32, i32)> = BTreeSet::new();
        let mut count = 0u64;
        let mut named = Vec::new();
        // An empty chunk the lane no longer holds names no hole (a dropped realm's leftovers
        // MEASURED as 717 phantom holes before this guard). The guard reads inside the walk's own
        // filter, not as an `if` in the body: the lane writes `empty_resident` and `resident`
        // together on every path it has, so the body's `if` could never take both ways and HR5
        // could not measure it. Example: a key the home moon's realm dropped names no hole.
        for ((_, key), level) in self
            .empty_resident
            .iter()
            .filter(|((r, k), _)| (*r == realm) & self.resident.contains(&(realm, *k)))
        {
            if !seen.insert((key.face, key.rung, key.x, key.y)) {
                continue;
            }
            // A column's slices are a few apart at most: the span is the bound over the ground.
            let mut holds_ground = false;
            let mut dz = -HOLE_SIBLING_SLICES;
            while dz <= HOLE_SIBLING_SLICES {
                let sibling = ChunkKey {
                    z: key.z + dz,
                    ..*key
                };
                holds_ground |= (dz != 0)
                    & self.resident.contains(&(realm, sibling))
                    & !self.empty_resident.contains_key(&(realm, sibling));
                dz += 1;
            }
            if !holds_ground {
                count += 1;
                if named.len() < NAMED_PER_FRAME {
                    named.push((*key, *level));
                }
            }
        }
        (count, named)
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

    /// Forget a realm: its body, its artifact and every chunk of it.
    pub fn forget(&mut self, realm: RealmId) {
        self.bodies.remove(&realm);
        self.charters.remove(&realm);
        self.snow_lines.remove(&realm);
        self.refused.remove(&realm);
        self.artifacts.remove(&realm);
        self.expected.remove(&realm);
        self.drop_chunks(realm);
    }

    /// ★ STATE THE ARTIFACT A REALM SAYS IT SERVES (the far-view ship), off its look bag: `Some`
    /// makes every request for the realm wait until the head at that digest is here; `None`
    /// (a realm that states none) lets the recipe's own relief build. A change of digest drops
    /// the chunks built on the old one.
    pub fn expect_artifact(&mut self, realm: RealmId, digest: Option<[u64; 2]>) {
        let before = self.expected.get(&realm).copied();
        if before == digest {
            return;
        }
        match digest {
            Some(d) => {
                self.expected.insert(realm, d);
            }
            None => {
                self.expected.remove(&realm);
            }
        }
        if before.is_some() && self.held.get(&realm).copied().unwrap_or(0) > 0 {
            self.counters.artifact_rebuilds += 1;
            self.drop_chunks(realm);
        }
    }

    /// The artifact a realm states it serves, as the lane holds it.
    #[must_use]
    pub fn expected_artifact(&self, realm: RealmId) -> Option<[u64; 2]> {
        self.expected.get(&realm).copied()
    }

    /// Drop every chunk of a realm — resident, building, and the parents — keeping its body.
    fn drop_chunks(&mut self, realm: RealmId) {
        self.held.remove(&realm);
        self.parents.forget(realm);
        self.resident.retain(|(r, _)| *r != realm);
        self.empty_resident.retain(|(r, _), _| *r != realm);
        let pending: Vec<ChunkKey> = self
            .pending
            .keys()
            .filter(|(r, _)| *r == realm)
            .map(|(_, k)| *k)
            .collect();
        for key in pending {
            self.pending.remove(&(realm, key));
            self.workers.cancel(realm, key);
        }
    }

    /// ★ STATE A REALM'S ARTIFACT (slice 8c stage C4c), as the client's book holds it now. Held by
    /// pointer and refreshed when the book's epoch moved (a level or a tile landed), so the next
    /// request reads the newest field. The FIRST artifact for a realm that already has chunks
    /// drops them: they were built from the recipe's own relief and would stand beside chunks
    /// built on the field (counted as `artifact_rebuilds`).
    pub fn state_artifact(&mut self, realm: RealmId, cache: &ArtifactCache) {
        let held = self.artifacts.get(&realm).map(|a| (a.head, a.epoch));
        if held == Some((cache.head, cache.epoch)) {
            return;
        }
        let first = held.is_none_or(|(head, _)| head != cache.head);
        if first {
            let had = self.held.get(&realm).copied().unwrap_or(0) > 0;
            if had {
                self.counters.artifact_rebuilds += 1;
            }
            // ★ THE BODY TAKES THE SOLVED SEA (C5): every chunk built from here on reads the
            // artifact's sea where its column holds no water row (the far view, the halo). ★ AND THE
            // SOLVED BAND (ruling B55): the head's two heights size the ladder, so a summit over the
            // recipe's band is built whole; a head the ladder refuses keeps the recipe's band. A band
            // that moved drops every pending build too: each read the old ladder.
            let mut moved = false;
            if let Some(body) = self.bodies.get(&realm) {
                let wet = body.with_sea_m(cache.head.sea());
                // ★ AND THE FINE GROUND'S BAND (B2; ruling B83 item 3) where the 128 m level applies.
                let banded = wet
                    .with_fine_band_m(cache.head.low_m, cache.head.high_m, cache.head.sea_m)
                    .unwrap_or(wet);
                moved = banded.ladder() != body.ladder();
                self.bodies.insert(realm, Arc::new(banded));
            }
            if had | moved {
                self.drop_chunks(realm);
            }
        }
        // ★ THE SNOW LINE (ruling B75 item 4): read again with every change of the cache, so it stands from the moment
        // the artifact is whole; the chunks keep their bytes, the drawing paints the line.
        let line = self
            .bodies
            .get(&realm)
            .zip(self.charters.get(&realm))
            .and_then(|(body, charter)| crate::surface::snow_line_of(body, charter, cache));
        match line {
            Some(line) => self.snow_lines.insert(realm, line),
            None => self.snow_lines.remove(&realm),
        };
        self.artifacts.insert(realm, Arc::new(cache.clone()));
    }

    /// ★ THE SNOW LINE of a realm as the drawing reads it (ruling B75 item 4): `[base, per sin², per |sin|, 1]`, metres
    /// over the ladder radius at a latitude's sine ([`vd_terrain::surface::SnowLine`]); `None` where the realm draws no
    /// snow (its artifact not whole, or no snow on its body).
    #[must_use]
    pub fn snow_line(&self, realm: RealmId) -> Option<[f32; 4]> {
        self.snow_lines.get(&realm).map(|l| {
            [
                l.base_m.to_f64() as f32,
                l.per_sin2_m.to_f64() as f32,
                l.per_sin_m.to_f64() as f32,
                1.0,
            ]
        })
    }

    /// The artifact the lane holds for a realm.
    #[must_use]
    pub fn artifact(&self, realm: RealmId) -> Option<&Arc<ArtifactCache>> {
        self.artifacts.get(&realm)
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
        self.resident.contains(&(realm, key)) | self.pending.contains_key(&(realm, key))
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
        // ★ A RE-REQUEST RE-KEYS A WAITING JOB ONLY WHEN ITS PRIORITY CHANGED (ruling B22 step 2).
        // The engine asks for every wanted chunk not yet resident on every frame, and a re-key is a
        // lock of the workers' queue and a move inside it: MEASURED up to 2 395 waiting jobs at
        // 240 m/s, re-keyed on every frame whether or not a descent had moved their places.
        if let Some(asked) = self.pending.get_mut(&(realm, key)) {
            if *asked != priority {
                *asked = priority;
                self.workers.reprioritise(realm, key, priority);
            }
            return;
        }
        // ★ THE ARTIFACT'S FIELD MUST BE WHOLE FOR THIS CHUNK (slice 8c stage C4c): a chunk whose
        // level is assembling or whose tiles are not all here is not queued — asked again next
        // frame, while the coarser rung stands (ruling F9). ★ AND A REALM THAT STATES AN ARTIFACT
        // IS BUILT ON IT ALONE (the far-view ship): until the head at the stated digest is here,
        // nothing is built for it — never the recipe's own relief, which would pop away later.
        let artifact = self.artifacts.get(&realm).map(Arc::clone);
        if let Some(expected) = self.expected.get(&realm)
            && artifact.as_ref().map(|a| a.head.digest) != Some(*expected)
        {
            self.counters.awaiting_artifact += 1;
            if self.awaiting_frame.len() < NAMED_PER_FRAME {
                self.awaiting_frame.push((realm, key, Vec::new()));
            }
            return;
        }
        // ★ AND THE FIELDS ITS PARENTS READ (2026-10-01, the picture gate on the solved ground): a
        // build reads its parents' meshes for the geomorph's targets, and a parent whose field is not
        // whole builds nothing — so a chunk built then morphs every vertex onto the coarser field,
        // or onto nothing where that row is missing too, and keeps it: nothing rebuilds it when the
        // tile lands. MEASURED at the solved ground stand: 997 459 of 36.1 million vertices fell back
        // in the live landing, 69 513 of 32.5 million with every part held (`examples/fallback_probe`).
        // A parent always stands inside the coarser ring, so its tiles lie within the reach.
        let reads = std::iter::once(key).chain(parent_keys(body, key));
        let waiting: Vec<ChunkKey> = reads
            .filter(|k| field_of(body, artifact.as_deref(), *k).is_none())
            .collect();
        if !waiting.is_empty() {
            self.counters.awaiting_artifact += 1;
            // `field_of` refuses only where the realm holds an artifact AND the body a lattice, so
            // the pair is here; read as an iterator of one, as the refusal below reads it. Each
            // tile is named once, whichever of the chunk and its parents waits on it.
            let missing: BTreeSet<(u8, u32, u32)> = artifact
                .as_deref()
                .zip(body.macro_lattice())
                .into_iter()
                .flat_map(|(a, lattice)| {
                    waiting
                        .iter()
                        .flat_map(|k| a.missing_tiles(&lattice, *k))
                        .collect::<Vec<_>>()
                })
                .collect();
            for (f, tx, ty) in &missing {
                *self.missing_hist.entry((realm, *f, *tx, *ty)).or_insert(0) += 1;
            }
            if self.awaiting_frame.len() < NAMED_PER_FRAME {
                // ★ THE REFUSAL NAMES THE TILES the stencils still wait for — the chunk's own and its
                // parents'. `field_of` refuses ONLY when the realm holds an artifact AND the body
                // holds a macro lattice, so both are here. Example: the home moon's chunk at rung 0
                // names the artifact tiles under it.
                self.awaiting_frame
                    .push((realm, key, missing.into_iter().collect()));
            }
            return;
        }
        // ★ A CHUNK WHOSE FINE TILES HAVE NOT LANDED IS BUILT ON THE ROUGH VIEW (ruling B86 item 7): its field, its
        // parents' and its morph's are the one ground under the fine ground; it is built again when they land.
        let artifact = match (artifact, body.macro_lattice()) {
            (Some(a), Some(lattice)) if a.is_rough(&lattice, key) => Some(Arc::new(a.rough_view())),
            (artifact, _) => artifact,
        };
        self.pending.insert((realm, key), priority);
        *self.held.entry(realm).or_insert(0) += 1;
        self.counters.submitted += 1;
        self.workers.submit(ChunkJob {
            realm,
            body: Arc::clone(body),
            key,
            parents: Arc::clone(&self.parents),
            priority,
            artifact,
        });
    }

    /// ★ WHETHER A ROUGH CHUNK MAY BE BUILT AGAIN ON THE FINE GROUND (ruling B86 item 7): resident on the rough ground,
    /// not building again already, and every fine tile it and its parents read is here now.
    #[must_use]
    pub fn upgradable(&self, realm: RealmId, key: ChunkKey) -> bool {
        if !self.rough.contains(&(realm, key)) | self.upgrading.contains(&(realm, key)) {
            return false;
        }
        let (Some(body), Some(artifact)) = (self.bodies.get(&realm), self.artifacts.get(&realm)) else {
            return false;
        };
        body.macro_lattice().is_some_and(|lattice| {
            std::iter::once(key)
                .chain(parent_keys(body, key))
                .all(|k| !artifact.is_rough(&lattice, k))
        })
    }

    /// Build a rough chunk again on the fine ground, at `priority`: the rough one stays resident and drawn until the
    /// fine one lands in its place. Nothing for a chunk [`ChunkLane::upgradable`] refuses.
    pub fn upgrade(&mut self, realm: RealmId, key: ChunkKey, priority: u32) {
        if !self.upgradable(realm, key) {
            return;
        }
        let (Some(body), Some(artifact)) = (self.bodies.get(&realm), self.artifacts.get(&realm)) else {
            return;
        };
        self.upgrading.insert((realm, key));
        self.pending.insert((realm, key), priority);
        self.counters.rough_upgrades += 1;
        self.workers.submit(ChunkJob {
            realm,
            body: Arc::clone(body),
            key,
            parents: Arc::clone(&self.parents),
            priority,
            artifact: Some(Arc::clone(artifact)),
        });
    }

    /// How many resident chunks of a realm stand on the rough ground.
    #[must_use]
    pub fn rough_resident(&self, realm: RealmId) -> usize {
        self.rough.iter().filter(|(r, _)| *r == realm).count()
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
        out.retain(|ready| {
            self.pending
                .remove(&(ready.realm, ready.geometry.key))
                .is_some()
        });
        // ★ A STALE BUILD IS REFUSED: built on another artifact than the realm's now (the recipe's
        // relief before the head, or an older head). Its pending entry is gone with the retain
        // above, so the next request submits it again, on the field.
        let stale = |lane: &ChunkLane, ready: &ChunkReady| {
            ready.geometry.built_on != lane.artifacts.get(&ready.realm).map(|a| a.head.digest)
        };
        let mut kept = Vec::with_capacity(out.len());
        for ready in out {
            // An upgrade's chunk is resident already: it is counted held once, before and after.
            let upgrade = self.upgrading.remove(&(ready.realm, ready.geometry.key));
            if stale(self, &ready) {
                self.counters.stale_builds += 1;
                if !upgrade {
                    let left = self.held.entry(ready.realm).or_insert(1);
                    *left -= 1;
                }
                continue;
            }
            kept.push(ready);
        }
        let out = kept;
        for ready in &out {
            self.resident.insert((ready.realm, ready.geometry.key));
            if ready.geometry.rough {
                self.counters.rough_builds += 1;
                self.rough.insert((ready.realm, ready.geometry.key));
            } else {
                self.rough.remove(&(ready.realm, ready.geometry.key));
            }
            if ready.geometry.triangles.is_empty() {
                self.counters.empty_chunks += 1;
                self.empty_resident.insert(
                    (ready.realm, ready.geometry.key),
                    ready.geometry.field_level,
                );
                if self.empty_frame.len() < NAMED_PER_FRAME {
                    self.empty_frame.push((
                        ready.realm,
                        ready.geometry.key,
                        ready.geometry.field_level,
                    ));
                }
            }
        }
        self.counters.harvested += out.len() as u64;
        out
    }

    /// Drop a chunk: resident or still building (withdrawn from the workers). A lookup per realm,
    /// never a scan of every chunk held (SL9).
    pub fn release(&mut self, realm: RealmId, key: ChunkKey) {
        let was_resident = self.resident.remove(&(realm, key));
        self.empty_resident.remove(&(realm, key));
        self.rough.remove(&(realm, key));
        self.upgrading.remove(&(realm, key));
        let was_pending = self.pending.remove(&(realm, key)).is_some();
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
        self.pending.keys().copied().collect()
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

/// THE EYE'S GROUND TRUTH (8p, ruling V14 D8-7): the DRAWN ground under a body-frame point
/// ([`drawn_surface_m`]), the point's height over it, and the biome there. The stamp on every judged
/// picture states these, and the picture gate recomputes them from the state file — two readings of
/// one recipe that must agree (M8-4). The surface is read at rung 0, the true shape; a drawn rung
/// stands within its own bound of it. `None` at the body's centre (no direction) or for a non-finite
/// point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeSurface {
    /// The drawn ground's radius along the point's direction, in metres.
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

/// ★ THE GROUND THE CHUNKS DRAW, under a direction, at rung 0 (2026-09-24, the believability arc's
/// reference stand): the eye stood 400 m over the solved ground while the ladder, the stamp and the
/// ruler all measured it 7 054 m over the recipe's OWN relief — the relief the artifact replaced —
/// so the ladder drew the ground under the eye at rung 3 instead of rung 0 and stated a horizon
/// four times too far. The ladder's descent and probe and the eye's ground truth read this:
/// [`drawn_field_m`] at rung 0, the ground. No water is drawn (ruling B22), so the eye over a sea
/// reads the sea floor. Example: a hull 20 m over the belt's sea reads about 6.9 km of height, and
/// the ladder draws the dry floor under it at the rung that height asks for.
///
/// A realm that states an artifact draws NOTHING until a field answers (the lane waits for the head
/// and refuses a build whose field is not here). For that window this answers the recipe's own relief
/// as a STAND-IN, so the ladder still wants a band and never releases the realm's chunks; the ruler
/// asks [`drawn_field_m`] itself and plants no ball on ground nobody draws.
#[must_use]
pub fn drawn_surface_m(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    dir: [f64; 3],
) -> f64 {
    drawn_field_m(body, artifact, dir, 0)
        .unwrap_or_else(|| vd_terrain::height::height_m(body, dir, 0))
}

/// ★ THE SURFACE DRAWN AT A RUNG, along a direction (2026-09-24): with no artifact (or a body with no
/// macro lattice) the recipe's own relief at the rung, which is what such a realm's chunks draw; with
/// one, the field that rung reads, else the finest whole pyramid level the cache holds, each through
/// [`vd_terrain::height::height_field_m`] — the ground the chunks draw, also under a sea (ruling B22:
/// no water is drawn). `None` where the realm states an artifact and no field it holds answers:
/// nothing is drawn there yet.
#[must_use]
pub fn drawn_field_m(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    dir: [f64; 3],
    rung: u8,
) -> Option<f64> {
    let recipe = || Some(vd_terrain::height::height_m(body, dir, rung));
    let Some(artifact) = artifact else {
        return recipe();
    };
    let Some(lattice) = body.macro_lattice() else {
        return recipe();
    };
    let drawn = |field: &dyn ZField| vd_terrain::height::height_field_m(body, field, dir, rung);
    artifact
        .field_at_rung(&lattice, rung)
        .and_then(|field| drawn(&*field))
        .or_else(|| {
            (1..=artifact.head.levels)
                .filter_map(|k| artifact.level(k))
                .find_map(|level| drawn(&**level))
        })
}

#[must_use]
pub fn eye_surface(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    point_m: [f64; 3],
) -> Option<EyeSurface> {
    let len = (point_m[0] * point_m[0] + point_m[1] * point_m[1] + point_m[2] * point_m[2]).sqrt();
    if !positive(len) {
        return None;
    }
    let dir = [point_m[0] / len, point_m[1] / len, point_m[2] / len];
    let surface = drawn_surface_m(body, artifact, dir);
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

/// Where the centre ray from `eye_m` along `forward` meets the DRAWN surface AT THE DRAWN RUNG
/// ([`drawn_field_m`]: the artifact's field at that rung, else its finest level held; the recipe's own
/// relief only for a realm with no artifact) — so the ball rests on the drawn ground, not on a finer
/// shape it is not standing on, never under the solved ground the artifact put in the recipe's place,
/// and nowhere a realm draws nothing yet — marched one cell at a time out to `reach_m` and refined by
/// bisection. `None` when the ray leaves without
/// touching ground within reach (an eye looking at the sky), for a rung the ladder lacks, or for a
/// degenerate eye or direction. The ball's radius is the HIT's distance along the ray times
/// [`RULER_TAN_HALF_ANGLE`], never smaller than half a cell.
#[must_use]
pub fn ruler_on_surface(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
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
        drawn_field_m(body, artifact, dir, rung).is_some_and(|h| len <= h)
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

    /// The drawn ground's range of a home-moon artifact, as its head states it (ruling B55).
    fn drawn_of(artifact: &vd_terrain::artifact::Artifact) -> (i32, i32) {
        artifact.drawn_range_m(
            &vd_terrain::home::home_moon()
                .macro_lattice()
                .expect("a lattice"),
        )
    }
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

    /// ★ THE LANE HOLDS A REALM'S SNOW LINE (ruling B75 item 4) from the moment its artifact is whole, drops it when a
    /// head that is not whole replaces it, and forgets it with the realm; a body with no snow holds none.
    ///
    /// **Example.** The pilot logs in over the home planet: the head lands and no snow is painted; a second later the
    /// pyramid is whole and the line stands; the planet leaves the drawn set and the line goes with it.
    #[test]
    fn the_lane_holds_a_realms_snow_line_once_its_artifact_is_whole() {
        use crate::surface::tests::{six_blocks, tiny_cache, wet_charter};
        let realm = RealmId::Planet(HOME_PLANET_SEED);
        let mut lane = lane();
        lane.state_surface(realm, &surface(77), Some(&wet_charter()), &look());
        lane.state_artifact(realm, &tiny_cache(realm, 0, None));
        assert_eq!(lane.snow_line(realm), None, "the head alone");
        lane.state_artifact(realm, &tiny_cache(realm, 0, Some(six_blocks())));
        assert_eq!(lane.snow_line(realm).map(|l| l[3]), Some(1.0), "the whole artifact");
        lane.state_artifact(realm, &tiny_cache(realm, 1, None));
        assert_eq!(lane.snow_line(realm), None, "a new head, not whole");
        lane.state_artifact(realm, &tiny_cache(realm, 1, Some(six_blocks())));
        assert_eq!(lane.snow_line(realm).map(|l| l[3]), Some(1.0));
        lane.forget(realm);
        assert_eq!(lane.snow_line(realm), None, "forgotten with the realm");
        // The home planet's own charter states no water: no snow.
        let mut dry = self::lane();
        dry.state_surface(realm, &surface(77), Some(&charter()), &look());
        dry.state_artifact(realm, &tiny_cache(realm, 0, Some(six_blocks())));
        assert_eq!(dry.snow_line(realm), None);
    }

    /// ★ THE SEA FLOOR IS DRAWN (2026-09-28, ruling B22): no water is drawn until the water fill,
    /// so a chunk under 20 km of sea keeps its ground triangles. THE CONTROL: the floor cut that
    /// this rule replaces dropped every triangle of this same chunk.
    #[test]
    fn a_chunk_under_the_sea_draws_its_floor() {
        let wet = home_planet().with_sea_m(Some(20_000));
        let key = surface_key(&wet, 4, 300, 700);
        let g = geometry_with(&wet, None, planet(), key, &ParentCache::default()).expect("built");
        assert!(
            !g.triangles.is_empty(),
            "the floor under 20 km of sea is drawn"
        );
    }

    /// ★ THE LANE ON THE ARTIFACT (slice 8c stage C4c, C5): the moon's artifact stated to the lane
    /// gives its body the solved sea; a coarse chunk reads a pyramid level and builds; a fine
    /// chunk whose tiles are not here waits (counted) while the same request with the tiles held
    /// builds; a first head after chunks were built drops them (counted); a second statement of
    /// the same cache is a no-op; the parents and the morph fallback read the field.
    /// ★ A STALE BUILD IS REFUSED (2026-09-20): a job the workers were already running on the
    /// recipe's relief when the realm's head arrived lands after the rebuild dropped the realm's
    /// chunks, under the fresh request's pending entry. Workers that keep a finished job through
    /// a cancel stand in for the running thread. The lane refuses it, counts it, and the next
    /// request builds the chunk on the field; a chunk built on the field is kept.
    #[test]
    fn a_chunk_built_on_the_recipe_that_lands_after_the_head_is_refused_and_rebuilt() {
        use crate::artifact_book::ArtifactReceiver;
        use vd_terrain::home::{
            HOME_MOON_BULK_DENSITY_KGM3, HOME_MOON_GRAVITY_MM_S2, HOME_MOON_RADIUS_BITS,
            HOME_MOON_SEED, HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words,
        };
        use vd_terrain::solve::Schedule;
        use vd_wire::channels::BulkMsg;
        /// Inline workers whose cancel does nothing: a job already on a thread.
        struct LateWorkers(InlineWorkers);
        impl ChunkWorkers for LateWorkers {
            fn submit(&mut self, job: ChunkJob) {
                self.0.submit(job);
            }
            fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize) {
                self.0.drain(out, max);
            }
            fn cancel(&mut self, _realm: RealmId, _key: ChunkKey) {}
            fn reprioritise(&mut self, realm: RealmId, key: ChunkKey, priority: u32) {
                self.0.reprioritise(realm, key, priority);
            }
            fn built(&self) -> BuildCount {
                self.0.built()
            }
        }
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact =
            vd_terrain::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let realm = RealmId::Planet(HOME_MOON_SEED);
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn_of(&artifact).0,
            high_m: drawn_of(&artifact).1,
        });
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let cache = rx.book().get(realm).cloned().expect("a cache");
        let moon_charter = vd_core::look::BodyCharter {
            gravity_mm_s2: HOME_MOON_GRAVITY_MM_S2,
            bulk_density_kgm3: HOME_MOON_BULK_DENSITY_KGM3,
            ..charter()
        };
        let surface = SurfaceStmt {
            frame: FrameRef::PlanetCentered {
                planet_seed: HOME_MOON_SEED,
            },
            generator: 77,
        };
        let look = Boundary::Shell {
            r: f64::from_bits(HOME_MOON_RADIUS_BITS),
        };
        let mut lane = ChunkLane::new(Box::new(LateWorkers(InlineWorkers::default())), 77);
        lane.state_surface(realm, &surface, Some(&moon_charter), &look);
        let top = lane.body(realm).expect("a body").ladder().rungs - 1;
        let key = ChunkKey {
            face: Face::PosX,
            rung: top - 1,
            x: 0,
            y: 0,
            z: vd_terrain::digest::surface_chunk_z(&moon, Face::PosX, top - 1, 0, 0),
        };
        // Built on the recipe before the head, sitting finished in the workers: then the head
        // lands, the rebuild drops the realm's chunks, and the same key is asked again.
        lane.request(realm, key, 0);
        assert_eq!(lane.counters().submitted, 1);
        // The same key asked again while it builds is never submitted twice: the lane states the
        // new priority to the workers, and what they built stands at one chunk.
        lane.request(realm, key, 7);
        assert_eq!(lane.counters().submitted, 1);
        assert_eq!(lane.built().chunks, 1);
        lane.state_artifact(realm, &cache);
        assert_eq!(lane.counters().artifact_rebuilds, 1);
        lane.request(realm, key, 0);
        assert_eq!(lane.counters().submitted, 2);
        // The first to land is the recipe's build: refused, not resident, counted. Its landing
        // took the fresh request's pending entry with it, so the fresh build that lands next is
        // dropped too (the lane cannot tell which job an entry belongs to), the key is neither
        // resident nor pending, and the next request builds it once more — one build wasted per
        // stale key, once per head, never a chunk standing on the wrong ground.
        let first = lane.poll(1);
        assert_eq!(first.len(), 0, "the stale build is refused");
        assert_eq!(lane.counters().stale_builds, 1);
        assert!(!lane.is_resident(realm, key));
        let second = lane.poll(1);
        assert_eq!(
            second.len(),
            0,
            "the fresh build lands under no pending entry"
        );
        assert!(!lane.holds(realm, key));
        lane.request(realm, key, 0);
        assert_eq!(lane.counters().submitted, 3);
        let third = lane.poll(1);
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].geometry.built_on, Some(artifact.digest()));
        assert!(lane.is_resident(realm, key));
        assert_eq!(lane.counters().stale_builds, 1);
        // The held count is one chunk: a release empties the realm.
        lane.release(realm, key);
        assert_eq!(lane.resident(realm), vec![]);
    }

    /// ★ A CHUNK WAITS FOR THE TILES ITS PARENTS READ (2026-10-01, the picture gate on the solved
    /// ground): on the moon, a rung-0 chunk whose own stencil reads tile (0, 0) of +X while one of
    /// its parents' reads tile (1, 0) too. With tile (0, 0) alone it is refused, NAMED with its
    /// parent's tile; once that tile lands it builds, its morph targets on its parents' meshes. RED
    /// before: the chunk built with no parent mesh, and every vertex fell back to the field.
    #[test]
    fn a_chunk_waits_for_the_tiles_its_parents_read() {
        use crate::artifact_book::ArtifactReceiver;
        use vd_terrain::home::{
            HOME_MOON_BULK_DENSITY_KGM3, HOME_MOON_GRAVITY_MM_S2, HOME_MOON_RADIUS_BITS,
            HOME_MOON_SEED, HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words,
        };
        use vd_terrain::solve::Schedule;
        use vd_wire::channels::BulkMsg;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact =
            vd_terrain::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let realm = RealmId::Planet(HOME_MOON_SEED);
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn_of(&artifact).0,
            high_m: drawn_of(&artifact).1,
        });
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let tile = |tx: u32, ty: u32| BulkMsg::ArtifactTile {
            realm,
            face: Face::PosX.index(),
            tx,
            ty,
            rows: artifact.tile(Face::PosX, tx, ty).to_bytes(),
        };
        rx.accept(tile(0, 0));
        let moon_charter = vd_core::look::BodyCharter {
            gravity_mm_s2: HOME_MOON_GRAVITY_MM_S2,
            bulk_density_kgm3: HOME_MOON_BULK_DENSITY_KGM3,
            ..charter()
        };
        let surface = SurfaceStmt {
            frame: FrameRef::PlanetCentered {
                planet_seed: HOME_MOON_SEED,
            },
            generator: 77,
        };
        let look = Boundary::Shell {
            r: f64::from_bits(HOME_MOON_RADIUS_BITS),
        };
        let mut lane = lane();
        lane.state_surface(realm, &surface, Some(&moon_charter), &look);
        let cache = rx.book().get(realm).cloned().expect("a cache");
        lane.state_artifact(realm, &cache);
        let body = *lane.body(realm).expect("a body").as_ref();
        let own = (Face::PosX.index(), 0, 0);
        let next = (Face::PosX.index(), 1, 0);
        // The column: rung 0, a third of the face up, the first along +u whose own stencil reads
        // tile (0, 0) alone while a parent's reads tile (1, 0) as well; its slices on the field.
        let y = (lattice.edge as i32 / 3) * lattice.cells_per_node as i32 / CHUNK_EDGE as i32;
        let at = |x: i32, z: i32| ChunkKey {
            face: Face::PosX,
            rung: 0,
            x,
            y,
            z,
        };
        let x = (0..body.ladder().cells_per_edge(0) as i32 / CHUNK_EDGE as i32)
            .find(|x| {
                let mine = vd_terrain::artifact::tiles_of_chunk(&lattice, at(*x, 0));
                (mine.len() == 1)
                    & mine.contains(&own)
                    & parent_keys(&body, at(*x, 0))
                        .iter()
                        .any(|p| vd_terrain::artifact::tiles_of_chunk(&lattice, *p).contains(&next))
            })
            .expect("a column at the tile's edge");
        let (span, _) = crate::ladder_view::column_span(
            &body,
            Some(&cache),
            crate::ladder_view::Column {
                face: Face::PosX,
                rung: 0,
                x,
                y,
            },
        );
        let slices: Vec<ChunkKey> = (span.lo..=span.hi).map(|z| at(x, z)).collect();
        for k in &slices {
            lane.request(realm, *k, 0);
        }
        assert_eq!(lane.counters().awaiting_artifact, slices.len() as u64);
        assert!(lane.poll(64).is_empty());
        let named = lane.take_awaiting();
        assert_eq!(named.len(), slices.len().min(NAMED_PER_FRAME));
        assert!(
            named.iter().all(|n| n.2 == vec![next]),
            "the parent's tile, never the chunk's own: {named:?}"
        );
        // The parent's tile lands: the same requests build, on their parents' meshes.
        rx.accept(tile(1, 0));
        let cache = rx.book().get(realm).cloned().expect("a cache");
        lane.state_artifact(realm, &cache);
        for k in &slices {
            lane.request(realm, *k, 0);
        }
        assert_eq!(lane.counters().awaiting_artifact, slices.len() as u64);
        let mut built = Vec::new();
        while built.len() < slices.len() {
            let more = lane.poll(64);
            assert!(!more.is_empty(), "every slice builds");
            built.extend(more);
        }
        let (fallbacks, vertices) = built.iter().fold((0u64, 0u64), |(f, v), b| {
            (
                f + u64::from(b.geometry.morph_fallbacks),
                v + b.geometry.vertices.len() as u64,
            )
        });
        assert!(vertices > 0, "the column holds ground");
        assert!(
            fallbacks * 100 <= vertices,
            "{fallbacks} of {vertices} fell back"
        );
    }

    #[test]
    fn the_lane_builds_on_the_artifacts_field_and_waits_for_its_tiles() {
        use crate::artifact_book::ArtifactReceiver;
        use vd_terrain::home::{
            HOME_MOON_BULK_DENSITY_KGM3, HOME_MOON_GRAVITY_MM_S2, HOME_MOON_RADIUS_BITS,
            HOME_MOON_SEED, HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words,
        };
        use vd_terrain::solve::Schedule;
        use vd_wire::channels::BulkMsg;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact =
            vd_terrain::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let realm = RealmId::Planet(HOME_MOON_SEED);
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn_of(&artifact).0,
            high_m: drawn_of(&artifact).1,
        });
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let moon_charter = vd_core::look::BodyCharter {
            gravity_mm_s2: HOME_MOON_GRAVITY_MM_S2,
            bulk_density_kgm3: HOME_MOON_BULK_DENSITY_KGM3,
            ..charter()
        };
        let surface = SurfaceStmt {
            frame: FrameRef::PlanetCentered {
                planet_seed: HOME_MOON_SEED,
            },
            generator: 77,
        };
        let look = Boundary::Shell {
            r: f64::from_bits(HOME_MOON_RADIUS_BITS),
        };
        let mut lane = lane();
        lane.state_surface(realm, &surface, Some(&moon_charter), &look);
        assert_eq!(lane.body(realm).expect("a body").sea_m(), None);
        // A chunk built before the head: the recipe's own relief.
        let top = lane.body(realm).expect("a body").ladder().rungs - 1;
        let key = |rung: u8, x: i32, y: i32| ChunkKey {
            face: Face::PosX,
            rung,
            x,
            y,
            z: vd_terrain::digest::surface_chunk_z(&moon, Face::PosX, rung, x, y),
        };
        lane.request(realm, key(top, 0, 0), 0);
        assert_eq!(lane.poll(8).len(), 1);
        assert_eq!(lane.artifact(realm), None);
        // The head arrives: the chunks are dropped and rebuilt; the body takes the sea.
        let cache = rx.book().get(realm).cloned().expect("a cache");
        lane.state_artifact(realm, &cache);
        assert_eq!(lane.counters().artifact_rebuilds, 1);
        assert_eq!(lane.resident(realm), vec![]);
        assert!(lane.artifact(realm).is_some());
        assert_eq!(lane.body(realm).expect("a body").sea_m(), artifact.sea());
        // ★ THE SLICES UNDER THE BAND THE HEAD SIZED (ruling B55): the solved moon's craters reach
        // under the recipe's band, so the head moved its floor; from here on a slice is named on the
        // lane's own body, as the lane names it.
        let banded = **lane.body(realm).expect("a body");
        assert_ne!(
            banded.ladder().floor_m,
            moon.ladder().floor_m,
            "the craters moved the floor"
        );
        let key = |rung: u8, x: i32, y: i32| ChunkKey {
            face: Face::PosX,
            rung,
            x,
            y,
            z: vd_terrain::digest::surface_chunk_z(&banded, Face::PosX, rung, x, y),
        };
        lane.state_artifact(realm, &cache);
        assert_eq!(
            lane.counters().artifact_rebuilds,
            1,
            "the same cache again is nothing"
        );
        // A coarse chunk reads the pyramid and builds; its parent (the top rung) reads the field
        // for the morph targets.
        lane.request(realm, key(top - 1, 0, 0), 0);
        let built = lane.poll(8);
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].geometry.key.rung, top - 1);
        assert_eq!(lane.counters().awaiting_artifact, 0);
        // A fine chunk waits for its tiles — and the refusal is NAMED with the tiles it waits for.
        lane.request(realm, key(0, 4_000, 4_000), 0);
        assert_eq!(lane.counters().awaiting_artifact, 1);
        assert!(lane.poll(8).is_empty());
        let named = lane.take_awaiting();
        assert_eq!(named.len(), 1);
        assert_eq!((named[0].0, named[0].1), (realm, key(0, 4_000, 4_000)));
        assert_eq!(
            named[0].2,
            cache.missing_tiles(&lattice, key(0, 4_000, 4_000)),
            "the tiles the stencil waits for"
        );
        assert!(!named[0].2.is_empty());
        // The missing tiles are counted once each, by how many waiting chunks read them — the
        // counts the renderer's gap readout names.
        let hist = lane.take_missing_hist();
        assert_eq!(hist.len(), named[0].2.len());
        assert!(
            named[0]
                .2
                .iter()
                .all(|(f, tx, ty)| hist.get(&(realm, *f, *tx, *ty)) == Some(&1))
        );
        assert!(lane.take_missing_hist().is_empty(), "taken once");
        assert!(
            cache.missing_tiles(&lattice, key(top - 1, 0, 0)).is_empty(),
            "a rung that reads a level waits for no tile"
        );
        assert!(lane.take_awaiting().is_empty(), "taken once");
        // The cap: thirteen refusals name twelve.
        for i in 0..13 {
            lane.request(realm, key(0, 4_000 + i, 4_000), 0);
        }
        assert_eq!(lane.take_awaiting().len(), NAMED_PER_FRAME);
        // The tiles land: the same request builds.
        for tx in 0..artifact.tiles_per_edge() {
            for ty in 0..artifact.tiles_per_edge() {
                rx.accept(BulkMsg::ArtifactTile {
                    realm,
                    face: Face::PosX.index(),
                    tx,
                    ty,
                    rows: artifact.tile(Face::PosX, tx, ty).to_bytes(),
                });
            }
        }
        let cache = rx.book().get(realm).cloned().expect("a cache");
        lane.state_artifact(realm, &cache);
        assert_eq!(
            lane.counters().artifact_rebuilds,
            1,
            "a later cache drops nothing"
        );
        lane.request(realm, key(0, 4_000, 4_000), 0);
        assert_eq!(lane.counters().awaiting_artifact, 14);
        assert!(
            lane.take_awaiting().is_empty(),
            "nothing refused once the tiles are here"
        );
        let built = lane.poll(8);
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].geometry.key.rung, 0);
        // ★ THE RECIPE'S OWN SLICE BUILDS EMPTY ON THE ARTIFACT (the coast flight, 2026-09-20):
        // `key` asked the slice the recipe's relief stands in, the chunk was built on the field,
        // and it holds NO triangle — a hole in the drawn ground, counted and named now.
        assert!(built[0].geometry.triangles.is_empty());
        assert_eq!(built[0].geometry.field_level, 0, "built on the rows");
        assert_eq!(built[0].geometry.built_on, Some(artifact.digest()));
        assert_eq!(lane.counters().empty_chunks, 1);
        assert_eq!(lane.take_empty(), vec![(realm, key(0, 4_000, 4_000), 0)]);
        assert!(lane.take_empty().is_empty(), "taken once");
        // The FIELD span's slices hold the ground: at least one of them builds with triangles, and
        // the empty count grows by exactly the slices that hold none.
        let with_sea = **lane.body(realm).expect("a body");
        let span = vd_terrain::digest::surface_column_field(
            &with_sea,
            &artifact,
            Face::PosX,
            0,
            4_000,
            4_000,
        )
        .expect("a span");
        let span_keys: Vec<ChunkKey> = (span.lo..=span.hi)
            .map(|z| ChunkKey {
                face: Face::PosX,
                rung: 0,
                x: 4_000,
                y: 4_000,
                z,
            })
            .filter(|k| *k != key(0, 4_000, 4_000))
            .collect();
        for k in &span_keys {
            lane.request(realm, *k, 0);
        }
        let built = lane.poll(8);
        assert_eq!(built.len(), span_keys.len());
        let with_ground = built
            .iter()
            .filter(|b| !b.geometry.triangles.is_empty())
            .count();
        assert!(with_ground > 0, "the field span holds the ground");
        let empties = built.len() - with_ground;
        assert_eq!(lane.counters().empty_chunks, 1 + empties as u64);
        assert_eq!(lane.take_empty().len(), empties);
        // ★ THE HOLE COLUMNS: the column holds an empty chunk and one with ground within the rule's
        // reach ([`HOLE_SIBLING_SLICES`]) — no hole; release every chunk with ground and the column
        // is a hole, named by its empty key; release the empty ones too and nothing is left to be a
        // hole. The recipe's own slice, asked before the field said where the ground stands, is
        // released first: on generator 26's moon it stands farther from the solved ground than the
        // rule's reach, and alone it is a hole of its own. The column's empty chunks are the two air
        // slices right over the field's span: one column with two empty chunks is one hole, named
        // once.
        lane.release(realm, key(0, 4_000, 4_000));
        let air = [1, 2].map(|up| ChunkKey {
            z: span.hi + up,
            ..key(0, 4_000, 4_000)
        });
        for k in air {
            lane.request(realm, k, 0);
        }
        let over = lane.poll(8);
        assert_eq!(over.len(), 2);
        assert!(
            over.iter().all(|b| b.geometry.triangles.is_empty()),
            "the air over the span"
        );
        let _ = lane.take_empty();
        assert_eq!(lane.hole_columns(realm), (0, vec![]));
        for b in built.iter().filter(|b| !b.geometry.triangles.is_empty()) {
            lane.release(realm, b.geometry.key);
        }
        let (holes, named) = lane.hole_columns(realm);
        assert_eq!(holes, 1);
        assert_eq!(
            named.len(),
            1,
            "one column, named once for its two empty chunks"
        );
        assert_eq!(
            (
                named[0].0.face,
                named[0].0.rung,
                named[0].0.x,
                named[0].0.y,
                named[0].1
            ),
            (Face::PosX, 0, 4_000, 4_000, 0)
        );
        // Release the empty ones too — the air and any slice of the span that holds no ground.
        let empty: Vec<ChunkKey> = built
            .iter()
            .filter(|b| b.geometry.triangles.is_empty())
            .map(|b| b.geometry.key)
            .chain(air)
            .collect();
        for k in empty {
            lane.release(realm, k);
        }
        assert_eq!(lane.hole_columns(realm), (0, vec![]));
        // ★ THE CAPS ON THE STAMP: thirteen columns whose asked slice holds no ground name twelve
        // empty chunks and twelve holes, while the counters carry all thirteen.
        let thirteen: Vec<ChunkKey> = (0..13).map(|i| key(0, 4_000 + i, 4_000)).collect();
        for k in &thirteen {
            lane.request(realm, *k, 0);
        }
        let built = lane.poll(16);
        assert_eq!(built.len(), thirteen.len());
        let empties = built
            .iter()
            .filter(|b| b.geometry.triangles.is_empty())
            .count();
        assert_eq!(
            empties,
            thirteen.len(),
            "the recipe's slice holds no ground"
        );
        assert_eq!(lane.take_empty().len(), NAMED_PER_FRAME);
        let (holes, named) = lane.hole_columns(realm);
        assert_eq!(holes, thirteen.len() as u64);
        assert_eq!(named.len(), NAMED_PER_FRAME);
        for k in &thirteen {
            lane.release(realm, *k);
        }
        assert_eq!(lane.hole_columns(realm), (0, vec![]));
        // A realm's rebuild (a NEW head) drops its empty chunks with the rest: no phantom hole.
        lane.request(realm, key(0, 4_000, 4_000), 0);
        assert_eq!(lane.poll(8).len(), 1);
        assert_eq!(lane.hole_columns(realm).0, 1);
        let mut other = cache.head;
        other.digest = [other.digest[0] ^ 1, other.digest[1]];
        let fresh = {
            let mut rx2 = ArtifactReceiver::default();
            rx2.accept(BulkMsg::ArtifactHead {
                realm,
                world_tag: 9,
                version: other.version,
                edge: other.edge,
                digest: other.digest,
                tiles_per_edge: other.tiles_per_edge,
                levels: other.levels,
                sea_m: other.sea_m,
                coast_parts: 0,
                low_m: other.low_m,
                high_m: other.high_m,
            });
            rx2.book().get(realm).cloned().expect("a cache")
        };
        lane.state_artifact(realm, &fresh);
        assert_eq!(lane.counters().artifact_rebuilds, 2);
        assert_eq!(lane.hole_columns(realm), (0, vec![]));
        // The field pick and the fallback height, directly.
        let body = **lane.body(realm).expect("a body");
        let picked = |f: Option<Option<SharedField>>| f.map(|f| f.map(|f| f.level()));
        assert_eq!(
            picked(field_of(&body, Some(&cache), key(0, 4_000, 4_000))),
            Some(Some(0))
        );
        assert_eq!(
            picked(field_of(&body, None, key(0, 4_000, 4_000))),
            Some(None)
        );
        let dir = [1.0, 0.0, 0.0];
        assert!(coarser_field_m(&body, Some(&cache), dir, top).is_some());
        assert!(coarser_field_m(&body, None, dir, top).is_some());
        // A body too small for a macro lattice reads the recipe whatever it holds.
        let pebble = body.without_macro_lattice();
        assert!(pebble.macro_lattice().is_none());
        assert_eq!(
            picked(field_of(&pebble, Some(&cache), key(0, 0, 0))),
            Some(None)
        );
        assert!(coarser_field_m(&pebble, Some(&cache), dir, 0).is_some());
        // The level the two top rungs read (level 1 on the moon) still assembling, the level
        // above it here: the child builds with no parent mesh and its morph fallback reads
        // nothing through the level, so it keeps its own radius.
        let mut half = ArtifactReceiver::default();
        half.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn_of(&artifact).0,
            high_m: drawn_of(&artifact).1,
        });
        half.accept(BulkMsg::ArtifactPyramid {
            realm,
            level: 2,
            part: 0,
            parts: 1,
            z_m: artifact.pyramid[1].clone(),
            water_m: artifact.pyramid_water[1].clone(),
        });
        let half_cache = half.book().get(realm).cloned().expect("a cache");
        assert!(ParentMesh::build(&body, Some(&half_cache), key(top - 1, 0, 0)).is_none());
        assert_eq!(
            coarser_field_m(&body, Some(&half_cache), dir, top - 1),
            None
        );
        assert!(
            geometry_with(
                &body,
                Some(&half_cache),
                realm,
                key(top - 1, 0, 0),
                &ParentCache::default()
            )
            .is_none(),
            "a waiting field builds nothing"
        );
        let mut lane2 = ChunkLane::new(Box::new(InlineWorkers::default()), 77);
        // A head stated before the body and before any chunk: nothing dropped, nothing rebuilt.
        lane2.state_artifact(realm, &half_cache);
        assert_eq!(lane2.counters().artifact_rebuilds, 0);
        lane2.state_surface(realm, &surface, Some(&moon_charter), &look);
        lane2.state_artifact(realm, &cache);
        assert_eq!(lane2.counters().artifact_rebuilds, 0);
        assert_eq!(lane2.body(realm).expect("a body").sea_m(), artifact.sea());
        // ★ THE EXPECTATION (the far-view ship): a realm that states an artifact at a digest the
        // lane does not hold builds NOTHING — not the recipe's relief; once the head at that
        // digest is here it builds; a changed digest drops what was built; `None` lets the
        // recipe build again.
        let mut lane3 = ChunkLane::new(Box::new(InlineWorkers::default()), 77);
        lane3.state_surface(realm, &surface, Some(&moon_charter), &look);
        lane3.expect_artifact(realm, Some(artifact.digest()));
        assert_eq!(lane3.expected_artifact(realm), Some(artifact.digest()));
        // Before its head lane3 holds the recipe's band: its slices are named on the recipe's body.
        lane3.request(realm, surface_key(&moon, top, 0, 0), 0);
        assert_eq!(lane3.counters().awaiting_artifact, 1);
        assert!(lane3.poll(8).is_empty());
        // The cap holds on the head's own refusal too: thirteen more refusals name twelve, and
        // the counter carries them all.
        for i in 0..13 {
            lane3.request(realm, surface_key(&moon, 0, 4_000 + i, 4_000), 0);
        }
        assert_eq!(lane3.counters().awaiting_artifact, 14);
        assert_eq!(lane3.take_awaiting().len(), NAMED_PER_FRAME);
        lane3.state_artifact(realm, &cache);
        lane3.request(realm, key(top, 0, 0), 0);
        assert_eq!(lane3.poll(8).len(), 1);
        lane3.expect_artifact(realm, Some(artifact.digest()));
        assert_eq!(
            lane3.counters().artifact_rebuilds,
            0,
            "the same digest again is nothing"
        );
        lane3.expect_artifact(realm, Some([9, 9]));
        assert_eq!(
            lane3.counters().artifact_rebuilds,
            1,
            "a new digest drops the chunks"
        );
        assert_eq!(lane3.resident(realm), vec![]);
        lane3.request(realm, key(top, 0, 0), 0);
        assert_eq!(lane3.counters().awaiting_artifact, 15);
        lane3.expect_artifact(realm, None);
        assert_eq!(lane3.expected_artifact(realm), None);
        lane3.request(realm, key(top, 0, 0), 0);
        assert_eq!(lane3.poll(8).len(), 1);
        lane3.forget(realm);
        assert_eq!(lane3.expected_artifact(realm), None);
        // Forgetting the realm drops the artifact with the body.
        lane.forget(realm);
        assert_eq!(lane.artifact(realm), None);
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

    /// A synthetic box of rung `rung`: rock (gap −20) under the layer `ground`, air (gap 20) over
    /// it, every column alike.
    fn layered_box(rung: u8, ground: i32) -> vd_terrain::lattice::SampleBox {
        use vd_terrain::lattice::{BOX_CELLS, BOX_EDGE, SampleBox, Site};
        let mut cells = vec![
            vd_terrain::chunk::Cell {
                stratum: vd_terrain::strata::Stratum::Air,
                gap: 20,
            };
            BOX_CELLS
        ];
        for c in BOX_LO..=BOX_HI {
            for b in BOX_LO..=BOX_HI {
                for a in BOX_LO..=BOX_HI {
                    if c < ground {
                        cells[SampleBox::index(a, b, c)] = vd_terrain::chunk::Cell {
                            stratum: vd_terrain::strata::Stratum::Granite,
                            gap: -20,
                        };
                    }
                }
            }
        }
        SampleBox {
            key: ChunkKey {
                face: Face::PosX,
                rung,
                x: 0,
                y: 0,
                z: 0,
            },
            cells,
            sites: vec![
                Site {
                    face: 0,
                    i: 0,
                    j: 0
                };
                BOX_EDGE * BOX_EDGE
            ],
            dirs: vec![Default::default(); BOX_EDGE * BOX_EDGE],
        }
    }

    /// ★ THE SEALED HOLLOWS ARE DRAWN AS ROCK (ruling B22 step 2): a pocket of air under rock in
    /// its own column becomes rock of the stratum over it; the open air over the ground stays; a
    /// hollow at the box's top, whose roof would be in the chunk above, stays; a box with no
    /// hollow, and a rung that carves nothing, give no copy.
    #[test]
    fn a_sealed_hollow_is_drawn_as_rock_and_the_open_air_is_kept() {
        use vd_terrain::lattice::SampleBox;
        let body = home_planet();
        // A ground at layer 40 with a pocket at layers 10–12 under columns (5..=7, 5..=7).
        let mut samples = layered_box(0, 40);
        for c in 10..=12 {
            for b in 5..=7 {
                for a in 5..=7 {
                    samples.cells[SampleBox::index(a, b, c)].gap = 30;
                }
            }
        }
        assert!(has_sealed_hollow(&samples));
        let sealed = sealed_hollows(&body, &samples).expect("rung 0 carves, and it holds a hollow");
        assert!(!has_sealed_hollow(&sealed));
        let pocket = sealed.cells[SampleBox::index(6, 6, 11)];
        assert_eq!(pocket.gap, -1);
        assert_eq!(pocket.stratum, vd_terrain::strata::Stratum::Granite);
        // The open air over the ground and the rock itself are as they were.
        assert_eq!(sealed.cells[SampleBox::index(6, 6, 50)].gap, 20);
        assert_eq!(sealed.cells[SampleBox::index(6, 6, 30)].gap, -20);
        // Every cell outside the pocket is untouched.
        let changed = samples
            .cells
            .iter()
            .zip(sealed.cells.iter())
            .filter(|(before, after)| before != after)
            .count();
        assert_eq!(changed, 27);
        // A box with no hollow gives no copy; a hollow at the box's top with its roof beyond the
        // box is not sealed here.
        let ground_only = layered_box(0, 40);
        assert!(sealed_hollows(&body, &ground_only).is_none());
        let mut roofless = layered_box(0, BOX_HI + 1);
        roofless.cells[SampleBox::index(3, 3, BOX_HI)].gap = 5;
        assert!(!has_sealed_hollow(&roofless));
        // A rung too coarse to carve gives no copy, whatever its cells hold.
        let coarse_rung = (0..body.ladder().rungs)
            .find(|r| {
                !vd_terrain::carve::tubes_carve_at(&body, *r)
                    & !vd_terrain::carve::caverns_carve_at(&body, *r)
            })
            .expect("a rung that carves nothing");
        let mut coarse = layered_box(coarse_rung, 40);
        coarse.cells[SampleBox::index(6, 6, 11)].gap = 30;
        assert!(has_sealed_hollow(&coarse));
        assert!(sealed_hollows(&body, &coarse).is_none());
    }

    /// ★ A RE-REQUEST AT THE SAME PRIORITY LEAVES THE QUEUE ALONE (ruling B22 step 2): the
    /// engine asks for every wanted chunk not yet resident on every frame; the lane re-keys a
    /// waiting job only when its priority changed.
    #[test]
    fn a_re_request_at_the_same_priority_does_not_touch_the_queue() {
        use std::sync::atomic::{AtomicU64, Ordering};
        /// Workers that keep every job waiting and count the re-keys.
        struct Counting(InlineWorkers, Arc<AtomicU64>);
        impl ChunkWorkers for Counting {
            fn submit(&mut self, _job: ChunkJob) {}
            fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize) {
                self.0.drain(out, max);
            }
            fn cancel(&mut self, _realm: RealmId, _key: ChunkKey) {}
            fn reprioritise(&mut self, _realm: RealmId, _key: ChunkKey, _priority: u32) {
                self.1.fetch_add(1, Ordering::Relaxed);
            }
            fn built(&self) -> BuildCount {
                self.0.built()
            }
        }
        let rekeys = Arc::new(AtomicU64::new(0));
        let mut lane = ChunkLane::new(
            Box::new(Counting(InlineWorkers::default(), Arc::clone(&rekeys))),
            77,
        );
        lane.state_surface(planet(), &surface(77), Some(&charter()), &look());
        let k0 = surface_key(&home_planet(), 0, 300, 700);
        lane.request(planet(), k0, 5);
        lane.request(planet(), k0, 5);
        lane.request(planet(), k0, 5);
        assert_eq!(
            rekeys.load(Ordering::Relaxed),
            0,
            "the same priority: no re-key"
        );
        lane.request(planet(), k0, 4);
        assert_eq!(
            rekeys.load(Ordering::Relaxed),
            1,
            "a new priority: one re-key"
        );
        lane.request(planet(), k0, 4);
        assert_eq!(rekeys.load(Ordering::Relaxed), 1);
        assert_eq!(lane.pending_count(), 1);
        assert!(lane.poll(8).is_empty(), "nothing was built");
        assert_eq!(lane.built().chunks, 0);
        lane.release(planet(), k0);
        assert_eq!(lane.pending_count(), 0);
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
            artifact: None,
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
        let samples = sample_box(&body, None, key).expect("in the band");
        // The box the drawn mesh reads: its sealed hollows as rock (`sealed_hollows`).
        let samples = sealed_hollows(&body, &samples).unwrap_or(samples);
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
            rough: false,
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
            rough: false,
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
        let g = geometry_with(&body, None, planet(), edge, &cache).expect("the edge chunk");
        assert!(g.morph_seam > 0, "no seam vertex on an edge chunk");
        let (fallbacks, vertices) = (g.morph_fallbacks, g.vertices.len() as u32);
        assert!(
            fallbacks * 100 < vertices,
            "{fallbacks} fallbacks of {vertices}"
        );
        let samples = sample_box(&body, None, edge).expect("in the band");
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
        let mid = geometry_with(
            &body,
            None,
            planet(),
            surface_key(&body, 0, 300, 700),
            &cache,
        )
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
            let samples = sample_box(&body, None, key).expect("in the band");
            // The box the drawn mesh reads: its sealed hollows as rock (`sealed_hollows`).
            let samples = sealed_hollows(&body, &samples).unwrap_or(samples);
            let mesh = extract(&samples);
            let parents: Vec<Arc<ParentMesh>> = parent_keys(&body, key)
                .into_iter()
                .filter_map(|p| cache.get(planet(), &body, None, p))
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
                worst = worst.max(gap);
                measured += usize::from(met);
            }
        }
        // ★ THE FEW RADIALS THAT DO NOT, RE-MEASURED 2026-09-21 on slice 8d step 1 (EACH BED ITS
        // OWN HARDNESS): 4 128 of the 4 134 radials stand inside the sink; SIX stand outside it,
        // the worst at 55.85 m. Half the bed tops stopped pulling, so the ground under this box
        // moved and more radials slip past the hill into a cave.
        //
        // The number moves at every stage that moves the ground, and it has moved seven times now:
        // SIX of 4 142 radials (worst 210.26 m) before the spectrum; NINETEEN of 5 284 (worst
        // 170.28 m) at 8a stage 1; THREE of 3 242 (worst 136.21 m) at stage 2; THIRTEEN of 6 180
        // (worst 128.91 m) at stage 3; FIFTEEN of 5 630 (worst 197.08 m) at stage 4; ONE of 4 476
        // (worst 31.29 m) at 8b stage 3, where the relief law halved the mountains and the caves
        // under this box with them; SIX of 4 134 (worst 55.85 m) here. The RADIAL COUNT is
        // the field's own roughness read back: a
        // smoother column carries more vertices over the parent's cells that answer a radial at
        // all, and the factor lowers the fine half of nine columns in ten.
        //
        // THE RESIDUE WAS ONE CLASS, MEASURED while it lasted: every outside hit stood BELOW its field
        // and inside the body's own CAVE BAND (8 m to 370 m under the surface) — the radial slipped
        // past the hill into a cave and reported the cave's depth, never a misplaced parent mesh.
        //
        // ⚠ OWED, THE OWNER'S CALL: whether a morph target should follow its parent into a cave at
        // all. ★ ANSWERED BY THE SEALED HOLLOWS (2026-09-28, ruling B22 step 2): the drawn meshes —
        // the chunk's and its parents' — read their boxes with the caves as rock, so no radial can
        // meet a cave. MEASURED: 0 of 2 744 radials outside the sink, the worst 4.14 m (before: 6 of
        // 4 134, the worst 55.85 m, every one in a cave); the radials are fewer because the caves'
        // own vertices are gone. With no residue left, the two cave discriminators that explained it
        // (below the field, inside the cave band) retire: `over` is nought, so each would read an
        // empty list.
        assert_eq!(
            over, 0,
            "{over} of {measured} radials outside the sink, the worst {worst:.2} m"
        );
        assert!(
            worst < 5.0,
            "the worst radial stands {worst:.2} m from its field"
        );
        assert!(measured > 1000, "{measured}");
    }

    #[test]
    fn a_parent_mesh_answers_the_radials_over_its_cells_nearest_the_asked_radius() {
        let body = home_planet();
        let key = surface_key(&body, 1, 150, 350);
        let pm = ParentMesh::build(&body, None, key).expect("the parent");
        assert_eq!(pm.key(), key);
        assert!(pm.triangle_count() > 0);
        // Every vertex of the parent's own mesh is met by its own radial, at its own radius,
        // within the cell it lies in (a vertex is on its triangles).
        let samples = sample_box(&body, None, key).expect("in the band");
        // The box the drawn mesh reads: its sealed hollows as rock (`sealed_hollows`).
        let samples = sealed_hollows(&body, &samples).unwrap_or(samples);
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
        assert!(ParentMesh::build(&body, None, outside).is_none());
    }

    #[test]
    fn the_parent_cache_shares_a_build_and_forgets_the_oldest() {
        let body = home_planet();
        let cache = ParentCache::default();
        assert!(cache.is_empty());
        let key = surface_key(&body, 1, 150, 350);
        let a = cache.get(planet(), &body, None, key).expect("built");
        let b = cache.get(planet(), &body, None, key).expect("cached");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 1);
        // The same key inserted again keeps one entry (and one place in the order).
        cache.insert(planet(), key, Arc::clone(&a));
        assert_eq!(cache.len(), 1);
        // A key outside the body builds nothing and keeps nothing.
        assert!(
            cache
                .get(planet(), &body, None, ChunkKey { z: -1, ..key })
                .is_none()
        );
        assert_eq!(cache.len(), 1);
        // A hit refreshes its place: after the cap is passed by fresh keys, the hit one stays
        // and the oldest untouched one leaves.
        let second = ChunkKey { x: 151, ..key };
        let b2 = cache.get(planet(), &body, None, second).expect("built");
        let _ = cache
            .get(planet(), &body, None, key)
            .expect("hit, refreshed");
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
                .get(planet(), &body, None, key)
                .is_some_and(|c| Arc::ptr_eq(&c, &a))
        );
        assert!(
            cache
                .get(planet(), &body, None, second)
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
                .get(planet(), &body, None, key)
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
        let gl = geometry_with(&body, None, planet(), left, &cache).expect("left");
        let gr = geometry_with(&body, None, planet(), right, &cache).expect("right");
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
            .filter_map(|p| cache.get(planet(), &body, None, p))
            .collect();
        let samples = sample_box(&body, None, left).expect("in the band");
        // The box the drawn mesh reads: its sealed hollows as rock (`sealed_hollows`).
        let samples = sealed_hollows(&body, &samples).unwrap_or(samples);
        let mesh = extract(&samples);
        let o = DVec3::from_array(gl.origin_m);
        let mut checked = 0;
        let mut deep = 0u32;
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
            // through the same function, so the two can never drift apart. Counted branch-free: the
            // sealed box holds none (below), so the old per-vertex check of a cave vertex's SHADE
            // (the parent's shade nearest its target, else the coarser field's) has no vertex here —
            // a synthetic test of that rule is owed (DEFERRED, THE SPEED STEP (5)).
            let len = DVec3::new(p[0], p[1], p[2]).length();
            let field = vd_terrain::height::height_m(&body, [dir.x, dir.y, dir.z], left.rung + 1);
            let reach = sink_m(&body, left.rung + 1);
            deep += u32::from(!on & ((field - len).abs() > reach));
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
        // (2026-09-21, slice 8d step 1, EACH BED ITS OWN HARDNESS: the box now holds ONE cave
        // vertex, and it stands 30.49 m under its field — still inside the cave band. Half the bed
        // tops stopped pulling, so most of this box's ground rose off the depth the caves are
        // carved at.)
        //
        // (2026-09-15, the earlier reading: 4 489 of 4 493 on a parent triangle, four fallbacks.
        // Before the crust rule the same box held 8 189 targets and two fell into a CAVE MOUTH in
        // the parent.)
        //
        // ★ (2026-09-28, ruling B22 step 2, THE SEALED HOLLOWS ARE DRAWN AS ROCK: the drawn mesh
        // reads this box with its caves as rock, so the box holds NO cave vertex; the one of step 1,
        // 30.49 m under its field, is gone with its cave, and no target can follow a parent into a
        // cave any more.)
        let seam = gl.morph_seam;
        assert_eq!(
            checked + fallbacks + seam + deep,
            targets,
            "{checked} on the parent of {targets}, {fallbacks} fallbacks, {seam} on a seam, {deep} in a cave"
        );
        assert_eq!(seam, 0, "this box holds no seam vertex");
        assert_eq!(deep, 0, "the drawn box holds no cave vertex");
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
    fn the_skirts_reach_the_sunk_coarser_ground_under_every_boundary_edge_facing_outward() {
        let body = home_planet();
        let key = surface_key(&body, 0, 300, 700);
        let samples = sample_box(&body, None, key).expect("in the band");
        // The box the drawn mesh reads: its sealed hollows as rock (`sealed_hollows`).
        let samples = sealed_hollows(&body, &samples).unwrap_or(samples);
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
        // Every skirt vertex hangs the skirt's drop under a top vertex along its radial — deeper than
        // the coarser rung sinks under this one, so a missing neighbour leaves no crack — keeps its
        // normal, and its morph target is the top's target dropped the same.
        let o = DVec3::from_array(g.origin_m);
        let drop = skirt_drop_m(&body, key.rung);
        assert!(
            drop > sink_m(&body, key.rung + 1),
            "{drop} m against the sink"
        );
        let ceiling = 2.0 * f64::from(vd_seed::ladder::cell_m(key.rung));
        assert!(drop >= ceiling);
        let (mut down, mut up) = (0u32, 0u32);
        let mut i = mesh.vertices.len();
        while i < g.vertices.len() {
            let q = o + DVec3::from(g.vertices[i].map(f64::from));
            // The top: the vertex of the base mesh at the same radial — the drop OVER the skirt
            // vertex under a sky-facing edge, the ceiling's two cells UNDER it over a ceiling's.
            let hang = |p: DVec3| -> f64 { (p - q).dot(p.normalize()) };
            let top = g.vertices[..mesh.vertices.len()]
                .iter()
                .enumerate()
                .find(|(_, v)| {
                    let p = o + DVec3::from(v.map(f64::from));
                    let along = (p - q).normalize().dot(p.normalize()).abs() > 0.999_999;
                    along & (((hang(p) - drop).abs() < 0.01) | ((hang(p) + ceiling).abs() < 0.01))
                })
                .map(|(j, _)| j)
                .expect("a top vertex over the skirt vertex");
            let p_top = o + DVec3::from(g.vertices[top].map(f64::from));
            let is_down = hang(p_top) > 0.0;
            down += u32::from(is_down);
            up += u32::from(!is_down);
            // Every skirt of this sealed box hangs DOWN (below), so its depth is the drop.
            let depth = drop;
            assert_eq!(g.normals[i], g.normals[top]);
            let (si, st) = (
                DVec3::from(g.sink_of(i).map(f64::from)),
                DVec3::from(g.sink_of(top).map(f64::from)),
            );
            assert!((si - st).length() < 1e-3);
            let mt = o + DVec3::from(g.morph_target(top).map(f64::from));
            let ms = o + DVec3::from(g.morph_target(i).map(f64::from));
            assert!(((mt - ms).length() - depth).abs() < 0.01);
            i += 1;
        }
        // This box's ground faces the sky and its hollows are sealed (`sealed_hollows`), so it holds
        // no cave roof: every skirt hangs down. (Before the sealing its caves' roofs rose; the
        // ceiling's own rule is `a_ceilings_skirt_hangs_up_and_a_grounds_hangs_down`.)
        assert!(down > 0, "{down} down");
        assert_eq!(up, 0, "{down} down, {up} up");
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
        // THE STAND-IN'S SINK (slice R1c): the next coarser rung's sink, nought at the top rung.
        let top = body.ladder().rungs - 1;
        assert_eq!(stand_in_sink_m(&body, 0), sink_m(&body, 1));
        assert_eq!(stand_in_sink_m(&body, top - 1), sink_m(&body, top));
        assert_eq!(stand_in_sink_m(&body, top), 0.0);
        // On a chunk: rung 0 carries zeros; rung 1 carries its radial times the sink.
        let g0 = geometry_of(&body, surface_key(&body, 0, 300, 700)).expect("rung 0");
        assert_eq!(g0.sink_m, 0.0);
        assert_eq!(g0.stand_in_sink_m, sink_m(&body, 1) as f32);
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
            stand_in_sink_m: 0.0,
            vertices: vec![[0.0; 3]; 70_000],
            normals: Vec::new(),
            packed_normals: Vec::new(),
            morph_normals: Vec::new(),
            surface: Vec::new(),
            heights: Vec::new(),
            radials: Vec::new(),
            triangles: vec![[0, 1, 2]],
            bounds: ([0.0; 3], [0.0; 3]),
            skirt_start: 0,
            field_level: NO_FIELD,
            built_on: None,
            rough: false,
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
                    let r = g.ramp_start(i);
                    assert!(lo[k] <= r[k]);
                    assert!(r[k] <= hi[k]);
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
            stand_in_sink_m: 0.0,
            vertices: Vec::new(),
            normals: Vec::new(),
            packed_normals: Vec::new(),
            morph_normals: Vec::new(),
            surface: Vec::new(),
            heights: Vec::new(),
            radials: Vec::new(),
            triangles: Vec::new(),
            bounds: ([0.0; 3], [0.0; 3]),
            skirt_start: 0,
            field_level: NO_FIELD,
            built_on: None,
            rough: false,
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
        let read = eye_surface(&body, None, eye).expect("a direction");
        assert_eq!(eye_surface(&body, None, eye), Some(read));
        assert!((read.surface_m - surface).abs() < 1e-6, "{read:?}");
        assert!((read.altitude_m - 3.4).abs() < 1e-6, "{read:?}");
        assert_eq!(
            read.biome,
            vd_terrain::height::biome_at(&body, dir, surface)
        );
        assert_eq!(eye_surface(&body, None, [0.0, 0.0, 0.0]), None);
        assert_eq!(eye_surface(&body, None, [f64::NAN, 0.0, 0.0]), None);
    }

    /// ★ THE HOME MOON'S ARTIFACT IN THREE AMOUNTS (2026-09-24, the drawn ground): the head alone,
    /// the head and its pyramid, and all of it with the tiles of one face — the three states a
    /// realm's cache passes through while its parts land.
    fn moon_caches() -> (BodyDefinition, [ArtifactCache; 3]) {
        use crate::artifact_book::ArtifactReceiver;
        use vd_terrain::home::{
            HOME_MOON_SEED, HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words,
        };
        use vd_terrain::solve::Schedule;
        use vd_wire::channels::BulkMsg;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact =
            vd_terrain::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let realm = RealmId::Planet(HOME_MOON_SEED);
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
            low_m: drawn_of(&artifact).0,
            high_m: drawn_of(&artifact).1,
        });
        let head = rx.book().get(realm).cloned().expect("a head");
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let pyramid = rx.book().get(realm).cloned().expect("a pyramid");
        for tx in 0..artifact.tiles_per_edge() {
            for ty in 0..artifact.tiles_per_edge() {
                rx.accept(BulkMsg::ArtifactTile {
                    realm,
                    face: Face::PosX.index(),
                    tx,
                    ty,
                    rows: artifact.tile(Face::PosX, tx, ty).to_bytes(),
                });
            }
        }
        let whole = rx.book().get(realm).cloned().expect("the tiles");
        (moon, [head, pyramid, whole])
    }

    /// A direction on the face whose tiles land in [`moon_caches`].
    const MOON_DIR: [f64; 3] = [0.957_826, 0.201_144, -0.124_518];

    #[test]
    fn the_coarser_fields_normal_faces_out_and_is_absent_where_no_row_answers() {
        let (moon, [head, _, whole]) = moon_caches();
        let dir = DVec3::from_array(vd_seed::bend::normalize(MOON_DIR));
        let n = field_normal(&moon, Some(&whole), dir, 1).expect("the tiles answer");
        assert!((n.length() - 1.0).abs() < 1e-9, "{n:?}");
        assert!(n.dot(dir) > 0.0, "the normal faces out: {n:?}");
        // The recipe's own relief (no artifact) answers too; a head alone answers nothing.
        assert!(field_normal(&moon, None, dir, 1).is_some());
        assert_eq!(field_normal(&moon, Some(&head), dir, 1), None);
    }

    #[test]
    fn a_skirt_reaches_past_the_coarser_sink_and_the_top_rung_hangs_the_least() {
        let body = home_planet();
        let top = body.ladder().rungs - 1;
        for rung in 0..top {
            let cell = f64::from(vd_seed::ladder::cell_m(rung));
            assert_eq!(
                skirt_drop_m(&body, rung),
                (2.0 * cell).max(sink_m(&body, rung + 1) + cell)
            );
        }
        assert_eq!(
            skirt_drop_m(&body, top),
            2.0 * f64::from(vd_seed::ladder::cell_m(top))
        );
    }

    /// ★ A CEILING'S SKIRT HANGS UP INTO THE ROCK (2026-09-24): one triangle whose face turns to the
    /// body's centre — a cave's roof, a tunnel's ceiling — is all boundary; each of its three edges
    /// hangs a skirt OUTWARD from the centre, never down into the open space under it; the same
    /// triangle wound to face the sky hangs its skirts down.
    #[test]
    fn a_ceilings_skirt_hangs_up_and_a_grounds_hangs_down() {
        let skirted = |tri: [u32; 3]| -> ChunkGeometry {
            let mut g = ChunkGeometry {
                key: surface_key(&home_planet(), 0, 300, 700),
                origin_m: [1_000_000.0, 0.0, 0.0],
                morph_m: vec![0.0; 3],
                morph_fallbacks: 0,
                morph_seam: 0,
                sink_m: 0.0,
                stand_in_sink_m: 0.0,
                // A small triangle across the radial +X, a metre on a side.
                vertices: vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                normals: vec![[1.0, 0.0, 0.0]; 3],
                packed_normals: vec![[0, 0]; 3],
                morph_normals: vec![[0, 0]; 3],
                surface: vec![[0, 0, 0, 0]; 3],
                heights: vec![0.0; 3],
                radials: vec![[1.0, 0.0, 0.0]; 3],
                triangles: vec![tri],
                bounds: ([0.0; 3], [0.0; 3]),
                skirt_start: 0,
                field_level: NO_FIELD,
                built_on: None,
                rough: false,
            };
            add_skirts(&mut g, 5.0, 2.0);
            g
        };
        // Wound (0, 1, 2) the face turns to +X, the sky: every skirt vertex hangs the drop, 5 m DOWN.
        let ground = skirted([0, 1, 2]);
        assert_eq!(ground.vertices.len(), 3 + 2 * 3);
        assert!(
            ground.vertices[3..]
                .iter()
                .all(|v| (v[0] + 5.0).abs() < 1e-3)
        );
        // Wound (0, 2, 1) it turns to the centre, a ceiling: every skirt vertex rises only the
        // ceiling's 2 m, UP into the rock.
        let ceiling = skirted([0, 2, 1]);
        assert!(
            ceiling.vertices[3..]
                .iter()
                .all(|v| (v[0] - 2.0).abs() < 1e-3)
        );
    }

    #[test]
    fn the_drawn_surface_reads_the_finest_field_the_cache_holds() {
        let (moon, [head, pyramid, whole]) = moon_caches();
        let dir = vd_seed::bend::normalize(MOON_DIR);
        let recipe = vd_terrain::height::height_m(&moon, dir, 0);
        let drawn = |field: &dyn ZField, d: [f64; 3]| {
            vd_terrain::height::height_field_m(&moon, field, d, 0).expect("a field answers")
        };
        let on_tiles = drawn(&**whole.tiles(), dir);
        let on_level = drawn(&**pyramid.level(1).expect("level 1 landed"), dir);
        let top = pyramid.head.levels;
        let on_top = drawn(
            &**pyramid.level(top).expect("the coarsest level landed"),
            dir,
        );
        // Every reading below discriminates: the tiles, the finest level, the coarsest level and the
        // recipe stand apart at this direction, so an order read wrongly cannot pass.
        assert_ne!(on_tiles, on_level);
        assert_ne!(on_level, on_top);
        assert_ne!(on_level, recipe);
        assert_ne!(on_tiles, recipe);
        // No artifact: the recipe, which such a realm draws. A head alone: no field answers, so
        // nothing is drawn there — and the eye's reading stands in with the recipe.
        assert_eq!(drawn_field_m(&moon, None, dir, 0), Some(recipe));
        assert_eq!(drawn_field_m(&moon, Some(&head), dir, 0), None);
        assert_eq!(drawn_surface_m(&moon, Some(&head), dir), recipe);
        // The pyramid alone: its FINEST level. The tiles: ahead of every level.
        assert_eq!(drawn_surface_m(&moon, Some(&pyramid), dir), on_level);
        assert_eq!(drawn_surface_m(&moon, Some(&whole), dir), on_tiles);
        // The opposite face, whose tiles never landed: the finest level there, not the recipe.
        let other = vd_seed::bend::normalize([-MOON_DIR[0], MOON_DIR[1], MOON_DIR[2]]);
        let other_level = drawn(&**whole.level(1).expect("level 1 landed"), other);
        assert_eq!(drawn_surface_m(&moon, Some(&whole), other), other_level);
        assert_ne!(other_level, vd_terrain::height::height_m(&moon, other, 0));
        // A body with no macro lattice draws the recipe, artifact or not.
        let rock = moon.without_macro_lattice();
        assert_eq!(
            drawn_field_m(&rock, Some(&whole), dir, 0),
            Some(vd_terrain::height::height_m(&rock, dir, 0))
        );
        // The eye's ground truth stands on the drawn ground.
        let eye = dir.map(|c| c * (on_tiles + 3.4));
        let read = eye_surface(&moon, Some(&whole), eye).expect("a direction");
        assert!((read.surface_m - on_tiles).abs() < 1e-6, "{read:?}");
        assert!((read.altitude_m - 3.4).abs() < 1e-6, "{read:?}");
    }

    #[test]
    fn the_ruler_rests_on_the_drawn_ground_and_plants_nothing_where_nothing_is_drawn() {
        use vd_core::glam::DVec3;
        let (moon, [head, pyramid, whole]) = moon_caches();
        let up = DVec3::from_array(vd_seed::bend::normalize(MOON_DIR));
        let on_tiles = drawn_surface_m(&moon, Some(&whole), up.to_array());
        let recipe = drawn_surface_m(&moon, None, up.to_array());
        // MEASURED here: the recipe's relief stands 3.26 km over the solved ground. So the eye stands
        // 300 m over the HIGHER of the two, the nose 60° down, and the march runs at rung 2, whose
        // 4 m cells reach 16 km: the ray meets each ground inside that reach.
        let rung = 2u8;
        let eye = (up * (on_tiles.max(recipe) + 300.0)).to_array();
        let level = up.cross(DVec3::Z).normalize();
        let tilt = 60.0_f64.to_radians();
        let nose = (level * tilt.cos() - up * tilt.sin())
            .normalize()
            .to_array();
        let reach = RULER_MAX_CELLS * f64::from(vd_seed::ladder::cell_m(rung));
        // The ball's foot: two radii under its centre, against a ground read at its own direction
        // and at the march's own rung.
        let foot = |r: Ruler, ground: &dyn Fn([f64; 3]) -> f64| {
            let c = DVec3::from_array(r.centre_m);
            c.length() - 2.0 * r.radius_m - ground(c.normalize().to_array())
        };
        let drawn_at = |cache: &ArtifactCache, d: [f64; 3]| {
            drawn_field_m(&moon, Some(cache), d, rung).expect("drawn there")
        };
        let on = ruler_on_surface(&moon, Some(&whole), eye, nose, rung, reach)
            .expect("the ray meets the tiles' ground");
        let on_foot = foot(on, &|d| drawn_at(&whole, d));
        assert!(
            on_foot.abs() < 1.0,
            "the ball's foot stands {on_foot} m off the tiles' ground"
        );
        let bare = ruler_on_surface(&moon, None, eye, nose, rung, reach)
            .expect("the ray meets the recipe's ground");
        let bare_foot = foot(bare, &|d| vd_terrain::height::height_m(&moon, d, rung));
        assert!(
            bare_foot.abs() < 1.0,
            "the ball's foot stands {bare_foot} m off the recipe"
        );
        assert_ne!(on, bare);
        // A head alone: the realm draws nothing yet, so no ball stands anywhere.
        assert_eq!(
            ruler_on_surface(&moon, Some(&head), eye, nose, rung, reach),
            None
        );
        // The pyramid without the tiles: the ball rests on the finest level, as the eye reads it.
        let lvl = ruler_on_surface(&moon, Some(&pyramid), eye, nose, rung, reach)
            .expect("the ray meets the level's ground");
        let lvl_foot = foot(lvl, &|d| drawn_at(&pyramid, d));
        assert!(
            lvl_foot.abs() < 1.0,
            "the ball's foot stands {lvl_foot} m off the level"
        );
        assert_ne!(lvl, on);
    }

    /// ★ THE DEFECT OF 2026-09-24 AT THE LADDER: the descent and the band probe read the eye's height
    /// over the DRAWN ground — the artifact's where the realm holds one — never over the recipe's own
    /// relief the artifact replaced.
    #[test]
    fn the_ladder_reads_the_eyes_height_over_the_drawn_ground() {
        use crate::ladder_view::{LadderView, floored_altitude_m, horizon_m};
        let (moon, [_, _, whole]) = moon_caches();
        let up = vd_seed::bend::normalize(MOON_DIR);
        let ground = drawn_surface_m(&moon, Some(&whole), up);
        let recipe = drawn_surface_m(&moon, None, up);
        let eye = up.map(|c| c * (ground + 3.4));
        let with = LadderView::default().wanted(&moon, eye, Some(&whole));
        let without = LadderView::default().wanted(&moon, eye, None);
        let want = floored_altitude_m(3.4);
        assert!((with.altitude_m - want).abs() < 1e-6, "{}", with.altitude_m);
        let over_recipe = floored_altitude_m(ground + 3.4 - recipe);
        assert!(
            (without.altitude_m - over_recipe).abs() < 1e-6,
            "{}",
            without.altitude_m
        );
        assert_ne!(with.altitude_m, without.altitude_m);
        let probe = LadderView::default().probe(&moon, Some(&whole), eye, with.keys[0]);
        assert!(
            (probe.horizon_m - horizon_m(ground, want)).abs() < 1e-6,
            "{probe:?}"
        );
    }

    /// ★ AND THE RUNG IT DRAWS: on the stand of the moon's first face where the solve raised the
    /// ground most over the recipe's relief, an eye standing on the solved ground asks rung 0, and the
    /// recipe's reading of the same eye asks a coarser ring.
    #[test]
    fn an_eye_on_the_solved_ground_asks_rung_zero() {
        use crate::ladder_view::{EYE_HEIGHT_M, LadderView, switch_m};
        let (moon, [_, _, whole]) = moon_caches();
        let (gap, stand) = (-8..=8)
            .flat_map(|a| (-8..=8).map(move |b| (a, b)))
            .map(|(a, b)| vd_seed::bend::normalize([1.0, f64::from(a) * 0.1, f64::from(b) * 0.1]))
            .map(|d| {
                (
                    drawn_surface_m(&moon, Some(&whole), d) - drawn_surface_m(&moon, None, d),
                    d,
                )
            })
            .max_by(|x, y| x.0.total_cmp(&y.0))
            .expect("a stand");
        assert!(
            gap > 2.0 * switch_m(0),
            "the largest raise on PosX is {gap} m"
        );
        let on = drawn_surface_m(&moon, Some(&whole), stand);
        let eye = stand.map(|c| c * (on + EYE_HEIGHT_M));
        assert_eq!(
            LadderView::default()
                .wanted(&moon, eye, Some(&whole))
                .rung_min,
            0
        );
        assert!(LadderView::default().wanted(&moon, eye, None).rung_min > 0);
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
        let ruler = ruler_on_surface(&body, None, eye.to_array(), nose.to_array(), 0, reach_m)
            .expect("the ray meets the ground");
        assert_eq!(
            ruler_on_surface(&body, None, eye.to_array(), nose.to_array(), 0, reach_m),
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
            ruler_on_surface(&body, None, eye.to_array(), up.to_array(), 0, 400.0),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, None, eye.to_array(), up.to_array(), 0, 466_000.0),
            None
        );
        // Straight down from aloft at a coarse rung: the hit is under the eye, one cell's step.
        let aloft = up * (surface + 60_000.0);
        let down = ruler_on_surface(
            &body,
            None,
            aloft.to_array(),
            (-up).to_array(),
            9,
            400_000.0,
        )
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
            ruler_on_surface(&body, None, buried.to_array(), nose.to_array(), 0, 400.0),
            None
        );
        // The angular size is two degrees, pinned to the trigonometry it names.
        assert!((RULER_TAN_HALF_ANGLE - 2.0_f64.to_radians().tan()).abs() < 1e-15);
        // Refusals: a rung the ladder lacks, a degenerate eye, a zero direction.
        assert_eq!(
            ruler_on_surface(
                &body,
                None,
                eye.to_array(),
                nose.to_array(),
                body.ladder().rungs,
                400.0
            ),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, None, [0.0, 0.0, 0.0], nose.to_array(), 0, 400.0),
            None
        );
        assert_eq!(
            ruler_on_surface(&body, None, eye.to_array(), [0.0, 0.0, 0.0], 0, 400.0),
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
        let mesh = Arc::new(ParentMesh::build(&body, None, key).expect("in the ladder"));
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
        assert!(cache.get(planet(), &body, None, other).is_some());
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
        let mesh = Arc::new(ParentMesh::build(&body, None, key(3)).expect("in the ladder"));
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
        assert!(cache.get(planet(), &body, None, key(3)).is_some());
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
        let mesh = ParentMesh::build(&body, None, key).expect("in the ladder");
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
