# Slice 7 — THE CLIENT LINKS THE RECIPE: the first drawn hill, and the render seam

Written 2026-09-08 for the owner's discussion, before any code. ASD-STE100, in-game examples,
drawings where they help. Every recommendation in §9 waits for a YES or a NO.

What this builds on: slice 5 (the recipe, `vd-terrain`) and slice 6 (the extractor: cells → a mesh, one
mapping to metres, byte-identical on five legs). What exists on the client today: a renderer-free
library (`vd-client`, Tier-A) that decodes the window into a scene of realm boxes and hands the Bevy
renderer plain vertex buffers; the renderer draws a sphere or a box per realm, the star sky, and takes
screenshots for the harness.

---

## 1. What this slice decides, and what it leaves to slice 8

**Decides:** the client LINKS the one generator (SL10 clause 2: the same crate, never a port) and
states its world identity at login; a realm's SURFACE statement reaches the client and turns into a
body definition; a CHUNK LANE in the library (request, poll, release) runs the generator and the
extractor on the library's own workers, never on the render thread; the geometry a chunk hands the
engine (`ChunkGeometry`) and where it is placed (as a child of the realm's row, floating origin); the
first PICTURES of the home planet, from orbit and from the ground, taken by the harness.

**Leaves to slice 8:** WHICH rung a chunk is drawn at and when it changes (the tier rule), the
crossfade between rungs, the order chunks arrive in, how far ahead the client builds (the residency
band). This slice draws ONE rung at a time and asserts that it never draws two.

**Example.** A pilot's client receives the home planet's row in the window, with its surface
statement: "I am seed-shaped; my seed is this; my recipe is version 1." The client builds the body
from the seed, asks its workers for the chunks around the pilot at one rung, and the engine draws
them under the hull. The planet's shard drew nothing; it stated one line.

---

## 2. The words, explained once

| Industry term | What it means here |
|---|---|
| **Vertex buffer / index buffer** | The two arrays a graphics card draws from: the points (vertices, with their attributes) and the triangles as three indices each. Slice 6 already produces both as integers; this slice hands them to the engine as floats. |
| **Attribute** | A per-vertex value beside the position: a normal, a colour, a texture coordinate. The engine's mesh has one buffer per attribute. |
| **Normal** | The direction a surface faces at a vertex. Lighting needs it. **Flat shading** uses one normal per triangle (every facet visible); **smooth shading** averages the normals of the triangles around a vertex (a rounded look). Neither moves a vertex, so both are style (ruling S6-5). |
| **Material** | How a surface reacts to light: colour, roughness, metalness. **PBR** (physically based rendering) is the standard model; Bevy's `StandardMaterial` is one. |
| **Directional light** | A light infinitely far away, like a sun: one direction, one colour, one strength. |
| **Floating origin** | Drawing everything relative to a point near the camera, because a 32-bit float (`f32`) has about seven digits: at 3 351 000 m from the planet's centre a metre is the last digit and a millimetre is lost. The library subtracts in 64-bit first and hands the engine small numbers. |
| **Frustum culling** | Not drawing what the camera cannot see (outside its view pyramid). The engine does it per mesh; a chunk is one mesh. |
| **Draw call** | One instruction to the graphics card to draw one mesh. Thousands per frame are fine; hundreds of thousands are not. One chunk is one draw call. |
| **Worker pool / worker thread** | Threads that run heavy work (generating a chunk) so the thread that draws frames never waits. The library owns the pool; the engine only asks and harvests. |
| **Seam (here)** | The boundary between the library and the engine: a set of functions and plain data types that any engine could call. |
| **ABI / C mirror** | A C-language header for the seam so a program not written in Rust can call it. Not built in this slice (§6). |
| **LOD / tier / rung** | One thing: the detail level of a chunk. Our word is RUNG (0 = one-metre cells, 11 = two-kilometre cells on the home planet). |
| **Skirt** | A short curtain hanging from a chunk's edge that hides a crack against a coarser neighbour. Slice 8's. |

---

## 3. The path from a shard's statement to a drawn hill

```text
  the planet's shard                    the gateway                 the client LIBRARY (vd-client, Tier-A)          the ENGINE (Bevy)
  ─────────────────                     ───────────                 ────────────────────────────────────            ─────────────────
  states its look:            ──window──▶ composes the      ──▶     RealmBox { shape, placement, facing,             asks: chunk_request(realm, key, rung)
   shell radius r                         picture per                 surface: Some(SurfaceStmt) }                    harvests: chunk_poll() → ChunkGeometry
   TAG_SURFACE:                           observer                  BodyDefinition::from_seed(seed, r)                 builds a Mesh, places it as a child of
    frame = PlanetCentered{seed}                                    ──workers──▶ sample_box → extract → positions       the realm's row, draws it
    generator = the version tag                                     ChunkGeometry { key, rung, origin_m, vertices,      never calls the generator itself
                                                                                    normals, triangles }
```

Nothing new crosses a realm boundary: the surface statement was planted in slice 4 and stated since
slice 5; the look shell was always there. SL6 is untouched.

---

## 4. The world hello, and what the client states

At login the client now SENDS its world identity: the DECLARED half (the recipe's version folded with
the universe seed) and the MEASURED half (the eight golden chunks of the home planet, cells and mesh,
computed by this client's binary on this chip). The gateway compares and refuses a mismatch by name,
ending the session (slice 5). The client can compute the measured half without the forest because the
home planet's two literals live in the recipe crate (`vd_terrain::home`).

```text
   client                                          gateway
   ──────                                          ───────
   Hello (version, unit)              ──▶          Welcome (the label; the universe seed is NOT on the wire yet — §12.6)
   HelloWorld { declared, measured }  ──▶          compares with its own WorldIdentity
                                      ◀──          nothing (equal), or WorldRefused { ours, theirs, half } + end
```

**Example.** A player updates the game on a machine whose graphics driver also changed. Nothing
changes: the identity is computed by the CPU from the recipe. A player who runs a build with one
extra octave states a different declared half and is told "update your build" before one hill is
drawn wrong.

---

## 5. The chunk lane: what the library gives the engine, and what it keeps

**The library keeps** the generator, the extractor, the body per realm, the worker pool, and every
number that decides what is drawn. **The engine gets** plain data and three verbs.

```text
  chunk_request(realm, key, rung)                  the engine asks; the library queues the work
  chunk_poll() -> Vec<ChunkReady>                  the engine harvests finished chunks, a bounded number per call
  chunk_release(realm, key, rung)                  the engine drops a chunk; the library frees it

  ChunkGeometry {
      key, rung,
      origin_m: [f64; 3],        the chunk's own origin in the realm's frame — the floating-origin anchor
      vertices: Vec<[f32; 3]>,   metres, relative to origin_m (small numbers, exact enough)
      normals:  Vec<[f32; 3]>,   client-derived from the triangles (style, ruling S6-5)
      triangles: Vec<[u32; 3]>,  the extractor's, unchanged
  }
```

**Why `origin_m` and relative vertices.** The extractor's `vertex_position_m` gives body-frame metres
in 64-bit (3 351 000 m from the centre). Handing those to an engine as 32-bit floats loses the
millimetres. So the library subtracts the chunk's origin in 64-bit and hands the engine the
differences; the engine places the chunk at `row placement ⊕ origin`, reduced against the eye before
the last narrowing — the pattern the realm boxes already use.

**Where the work runs.** A `ChunkWorkers` seam: one trait with one method (take a job, return a
receiver). The Tier-A tests drive an INLINE implementation that runs the job on the calling thread,
so every branch is covered without a thread; the shipped binary installs a threaded implementation
on `std::thread::spawn` and `crossbeam-channel`, which is already a workspace dependency (no new
library). The render thread never generates.

---

## 6. What is NOT in this slice, and why

- **The C mirror.** The owner removed the other engine from every plan (ruling V4): Bevy continues and
  the client stays replaceable because the SERVER is authoritative, not because a C header exists.
  The seam is plain data and three verbs, so a header can be generated the day a second engine is
  real. `cbindgen` is a library decision for that day.
- **The tier rule, the crossfade, the residency band, the skirt** — slice 8. This slice draws one
  rung and asserts it.
- **Diffs** (mined cells, placed blocks) — slices 9 and 10. This slice draws the seed's shape only.
- **The atmosphere and the Hillaire sky** — the lighting lane, later. This slice lights the ground
  with one directional light from the brightest luminous row in the window (the star), so a hill has a
  lit side and a shadow side. Flat or smooth normals are a debug switch, both derived on the client.
- **Trees, grass, decoration** — slice 14.

---

## 7. Placement: a chunk is a child of its realm's row

```text
   the realm's row (from the window):   placement P (metres, f64), facing F (the planet's spin)
   the chunk:                           origin O (body frame, f64), vertices v (relative, f32)
   the eye:                             E (f64, the origin frame)

   engine transform of the chunk  =  narrow( P + F·O − E )   computed in f64, narrowed ONCE
   a vertex on screen             =  transform · v
```

Two chunks of one column reduce against the same eye and never disagree on their shared edge. The
planet's spin is on the row, so every chunk turns with it at no cost. The observer standing on the
planet is INSIDE the planet's frame: the origin frame is the planet's, `P` is zero, and the chunks sit
at their own origins.

**The camera's UP — ruled (V11, 2026-09-08).** Up is the SERVER's: the containing realm's gravity
function applied to the body's state — toward a planet's centre, the normal of the wall a station
walker's magnetic boots hold, away from a cylinder's axis — authored into the body's pose. LOOKING is
an INPUT the client applies at once inside the body's limits and the server confirms; other players see
the turn; hits are the server's. The client computes no up. In this slice the camera sits fixed in the
hull's frame and the ground pictures come from a hull the planet's shard berths upright on the surface
it computes from the same recipe.

---

## 8. The measurements the slice owes, FIRST

| # | What | How | Gate |
|---|---|---|---|
| M7-1 | The link's cost: shipped client binary size and cold build time, before and after | `cargo build --release -p vd-client-bin` twice | under 8 MB and under 60 s added (the proposal's numbers) |
| M7-2 | Chunks per second on ONE core and on EIGHT (the threaded workers), rung 0 | a bench in the client crate over the home planet's chunks | recorded; slice 8 sizes the band from it |
| M7-3 | Bytes per chunk handed to the engine, and the copy's cost | the same bench | recorded |
| M7-4 | THE FIRST PICTURES: the home planet from orbit (rung 8–11) and from the ground (rung 0), by the harness's screenshot, at the standing look and at a cave mouth | `vdctl` screenshots in the window flight | the owner judges: the crease question (surface nets vs dual contouring), the cave lattice step (V35), the sea, the landforms |
| M7-5 | The world hello round trip: a client with a wrong declared half is refused and ends | a process-tier test | red on a served mismatch |

Nothing below is built before M7-1 shows the link is affordable.

---

## 9. What you decide

| # | Question | Recommendation |
|---|---|---|
| S7-1 | The link | **`vd-terrain` and `vd-seed` become dependencies of `vd-client`; `vd-physics` stays a dev-only dependency; a crate-isolation row refuses a client dependency on any motion crate** |
| S7-2 | The world hello | **The client states both halves at login, computed on the home planet's literals; a client that cannot compute them does not log in** |
| S7-3 | The surface statement | **`RealmBox` carries the realm's `SurfaceStmt`; the client builds the body from the seed in the frame and the look shell's radius; a statement whose generator tag is not the client's is refused and counted, never drawn** |
| S7-4 | The chunk lane | **Three verbs and one data type, as §5; the library never refuses a legal rung; a bounded harvest per call** |
| S7-5 | One rung per session in this slice | **A dev flag names the rung; the library asserts that no two rungs of one realm are resident at once; the tier rule waits for slice 8** |
| S7-6 | The workers | **An injected `ChunkWorkers` trait: inline in Tier-A tests, `std::thread::spawn` + `crossbeam-channel` in the binary; no new library** |
| S7-7 | Lighting in this slice | **One directional light from the brightest luminous row in the window; normals derived on the client, flat or smooth by a debug switch; the atmosphere later** |
| S7-8 | The camera's up on a planet | **RULED (V11): up is the server's, from the realm's gravity function applied to the body — not a constant; looking is an input the client applies at once inside the body's limits and the server confirms; hits are the server's.** Nothing of it is built here; the camera sits fixed in the hull's frame |
| S7-9 | The C mirror | **Not now** (ruling V4); the seam stays plain data so a header can be generated later |
| S7-10 | The gates | **The drawn vertices equal the extractor's positions (client decoration moves nothing the collider shares — the TRIM rule, stated now, enforced by a comparison); the one-rung assertion; the isolation rows; the refusal counters; `vd-client` stays Tier-A at 100 %** |
| S7-11 | The pictures | **Part of the slice's delivery: from orbit and from the ground, through the harness, judged by the owner before slice 8 and before the first saved world** |

---

## 10. Laws

- **SL10 clause 2** (one generator, a port forbidden): the client links the crate; nothing in the
  client re-implements a cell or a vertex.
- **SL10 clause 7** (the client derives no pose, no velocity, no entity state): the client derives
  SHAPE at an address, and places it on a delivered row. The camera's up (S7-8, ruling V11) is the
  delivered pose of the body the camera rides; the client computes none.
- **SL1 clause 4** (never fold an absolute from the root): the chunk is placed as a child of ITS
  realm's row, never composed from a chain the client walks.
- **SL3** (a realm draws itself): the realm's look for terrain IS its statement; the client samples
  it at a rung. Slice 8's tier rule is the observer's sampling, not the realm's choice.
- **HR3**: the client never branches on a realm kind; it branches on the presence of a surface
  statement, which is a capability the realm stated.
- **HR5**: `vd-client` stays Tier-A; the threaded workers live in the Tier-B binary behind the seam.
- **SL8**: this slice draws one rung so no rung boundary reaches a picture before its detector does.

---

## 11. How it is built

1. M7-1 (the link's cost), then the link and the world hello.
2. The surface statement into `RealmBox`; the body per realm; the chunk lane and the worker seam.
3. `ChunkGeometry` in the renderer: a mesh per chunk as a child of the row; the light; the up.
4. The window flight with screenshots (M7-4): from orbit, from the ground, at a cave mouth.
5. The refuter, every finding answered; the gates; the report; the owner's word.

## 12. Results (2026-09-08, after the build)

### 12.1 The measurements

| # | Result |
|---|---|
| M7-1 | The link costs the headless release client 51 280 bytes (4 170 816 → 4 222 096) and 12.7 s of cold build. The windowed release client is 80 703 712 bytes and builds in 141 s; the terrain module is a small part of that (Bevy is the rest). Both under the proposal's bounds (8 MB, 60 s). |
| M7-2 | Client geometry at rung 0 over the 243 chunks around the ground spot (9 × 9 columns, three chunks each): ONE thread 239 chunks/s (4.18 ms per chunk: sample box + extraction + positions + normals); 14 threads 2 004 chunks/s (8.4×). `cargo run --release -p vd-bins --example terrain_cost`. |
| M7-3 | Per rung-0 chunk handed to the engine: 8 577 vertices, 16 615 triangles, 405 228 bytes (f32 positions, f32 normals, u32 triangles). A 13 × 13 × 3 patch is 507 chunks ≈ 205 MB of mesh data; slice 8's residency band is sized from this and from the rungs above (fewer crossings per chunk at a coarse rung is NOT true: a chunk is 62³ cells at every rung, so the bytes per chunk are the same order at every rung and the count of chunks is what the band controls). |
| M7-4 | THREE pictures, taken by the harness on the shipped path (`crates/bins/tests/terrain_pictures.rs`, `docs/investigation/2026-09-07/pictures/{ground,hill,aloft}.png`): from the ground (rung 0, 1.8 m over the surface, 13 × 13 columns), from a hill (rung 3, 300 m up), from aloft (rung 9, 60 km up, 25 × 25 columns of 32 km). All three on the planet's DAY side with the star 25° over the horizon, computed from the planet's own orbit. A cave-mouth picture is NOT taken (§12.4). |
| M7-5 | NOT done as a process test. The client library's own tests cover the hello (sent once after the welcome, retried on a failed send, terminal on a refusal); the process-tier flight with a served mismatch is owed (§12.4). |

### 12.2 What the pictures found (five defects, each MEASURED before it was fixed)

The first ground picture was black over a light-blue ball. Every one of these was found by reading the
harness's own instruments (the captured state, then a diagnostic line per second in the renderer:
each row's drawn centre and facing, the chunks' nearest and farthest distance, the camera's planes),
never by guessing:

1. **The login landed at the planet's centre.** The stand-in spawn pose was lowered through a
   lattice position whose cell half was discarded; a 3 351 km offset became a few metres. The
   stand-in now keeps the metres the operator stated (`SpawnPose.offset_m`).
2. **The column under the eye lay 230 km away.** `chunks_around` fed the face TANGENTS a direction
   gives into the cell index, which wants the UNBENT face parameter. The inverse bend now sits between
   them, as `vd_core::grid::shell` already did. The unit test
   `the_column_under_a_point_holds_the_point_at_every_rung` fails by 230 554 m without the fix — the
   same number the flight measured.
3. **The light-blue ball was the player's own marker.** The pilot camera lifted the eye 1.6 m along
   the FRAME's `+Y`; on a planet the avatar's up is the radial, so the marker sat in front of the eye
   and filled the lower half of the frame. The eye now lifts along the avatar's OWN up (at rest the
   same vector; the pixel gates reconstruct the camera through the same function).
4. **The ground was on the night side.** The chosen direction had the star 70° BELOW the horizon; only
   the fill light reached the ground. The gate now computes the star's direction from the planet's
   orbit (the elements its shard authors, at the clock's genesis) and stands the accounts where the
   star is 25° over the horizon.
5. **The ground was thirty times white.** The sun carried a physical 100 000 lux under the camera's
   default exposure (EV100 9.7, which every star sprite and marker was measured under), and the stub
   world's key light added 10 000 lux from a direction no star stands in. The sun's illuminance is now
   DERIVED from the camera's exposure (`π / exposure`: a white face square to the sun renders white)
   and the key light retires when the sun is born.

Two more things the pictures made necessary: the avatar is born with a FACING (the stand-in's new
`@qx,qy,qz,qw`, lowered through `StoredHome::in_realm_facing`), so a ground picture has the radial as
its up and its nose toward the star, with no look-at loop that would re-derive yaw and pitch against
the frame's `+Y`; and a body farther than two radii gets no column under the eye (`FAR_EYE_RADII`),
because the two other planets of the home system, 1.5 × 10¹¹ m away, each cost a column of chunks
placed where nobody looks.

### 12.3 What the pictures show, and what the owner is asked to judge

- The ground is a lit plain with a soft gradient; the relief at the standing spot is gentle (the seed's
  long waves are 20–400 km; over 400 m the slope is small). The patch's far edge is a hard horizon 400 m
  away: that is `D-TERRAIN-3` (one rung), and slice 8's ladder replaces it with the rungs beyond.
- From the hill and from aloft the long-wave relief shows as a rise on the horizon. The picture from
  aloft covers 800 km at rung 9 (512 m cells); the surface reads as smooth, with no visible cell.
- The sky is black. The star's own disc is not drawn (the system's row is a look shell 6 × 10¹⁰ m
  away, hidden behind its own outline rules), and the star field held 0 stars at the moment of the
  screenshot — UNMEASURED why: the most likely cause is the sky's paced part delivery (1 309 parts over
  beats) not yet confirmed 10 s after login. Neither is slice 7's; both are noted for the refuter and
  for slice 8's pictures.
- THE CREASE QUESTION (surface nets vs dual contouring) and THE CAVE LATTICE STEP cannot be judged
  from these three pictures: the standing spot is inside the `+Y` face (at face coordinates
  (0.012, −0.855)), so no face seam is in frame, and no cave mouth is in frame. Slice 8's pictures at
  a seam and at a cave mouth are owed before the first saved world (§12.4).

### 12.4 Owed, and where it is registered

| Owed | Where | When |
|---|---|---|
| The one rung per realm (the flag, the assertion) | `D-TERRAIN-3` 🟥 | slice 8 deletes it |
| The stand's UP stated by the operator (the spawn facing stand-in); the real up is the realm's gravity function, server-side (ruling V11) | `D-TERRAIN-4` 🟥 | the character/suit slice; the stand-in is deleted with `VD_SPAWN_POSES` (P7) |
| A picture at a face seam and at a cave mouth | slice 8's pictures | before the first saved world |
| M7-5 as a process test (a served mismatch refuses a client) | slice 8 or the persistence slice, whichever first changes the identity | — |
| The star field on a planet's sky at login (0 stars at the screenshot) | to measure in slice 8's flight (sky parts held vs time) | slice 8 |

### 12.5 The refuter

The refuter (Opus 5, read-only) reported 19 findings: 6 defects, 9 weaknesses, 4 notes
(`verdicts/slice_07_refutation.md`, every one answered). Fixed: the client's compiled-in universe
seed (now read as the gateway reads it), the work light that stood BELOW the observer, the fill light
never re-aimed, five uncovered branches, the surface band from one height (now five: the centre and
the corners), refusals re-counted every frame, the scene clone with nothing drawn, the sleep in the
picture gate (now `terrain_chunks_pending le 0`, a new wait field), an unbounded radius (now
`MAX_RADIUS = 64`), no cancellation of a released chunk (now `ChunkWorkers::cancel`), two scans per
release, a misplaced doc comment, threads started without a rung flag. Kept with a reason: the hidden
outline beyond the patch (it is `D-TERRAIN-3`), the wanted set under a spinning planet (the ground
really moves), the malformed-tag counter (the lane's, slice 10), the eye clearance, the paint
classifier's reach. The second lint, coverage and flight ran after these fixes.

### 12.6 An ask: the universe seed on the wire

The client must state the world identity at login, and the declared half folds the universe seed.
Today the client reads the seed from its own environment, as the gateway does. A shipped client has
no operator's environment: it must LEARN the seed from the gateway before it states the identity —
one `u64` on `Welcome` (the connection plane's lane, gateway → client, not a realm boundary). SL6
still says ask before a wire arm changes, so this is asked, not done: **may `Welcome` carry the
universe seed?** Until then, a cluster on another seed needs `VD_UNIVERSE_SEED` in the client's
environment too.
