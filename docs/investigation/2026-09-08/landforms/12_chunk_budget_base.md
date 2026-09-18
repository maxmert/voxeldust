# 12 — The chunk budget as it stands, before 8a (2026-09-16)

Ruling T1 of `docs/design/owner_decisions_2026-09-16_terrain.md`, **the eight-millisecond budget
(L25)**: one rung-1 chunk on the walk's path builds in 111–112 ms, fourteen times the budget, before
any roughness. 8a makes such chunks common. This document states the BASE, so 8a's delta is judged
against a measurement and not against a memory.

**THE HEADLINE.** The 111 ms is not the chunk's own extraction. It is **THE EIGHT PARENT MESHES**
the chunk's morph targets read — 80 % of the build, a phase the 8 ms budget never named. And more
than half of THAT is the recipe again: a cold chunk runs the sample box **NINE times**, once for
itself and once for each parent. The two phases the budget does name are 5.7 ms of the 102 ms a
worker pays.

Every number is marked MEASURED or ESTIMATED. Nothing here is argued.

---

## 1. The method

**The program.** `crates/bins/examples/chunk_budget.rs` (added by this measurement; an example — no
product code changed). It builds real chunks of THE world's home planet through the shipped path —
`vd_terrain::lattice::sample_box`, `vd_terrain::extract::extract`,
`vd_client::chunks::geometry_from`, `vd_client::chunks::ParentMesh` and `ParentCache` — and times
the phases apart.

```text
cargo run --release -p vd-bins --example chunk_budget
```

**The machine.** Apple M4 Pro, 14 cores (10 performance, 4 efficiency), macOS 26.5.1. On this
machine the client's own rule `worker_share(14)` gives the terrain **3 CPU builders** (ruling F6:
the workers get a share of the cores, never all). The bench is single-threaded, so every number
below is ONE worker's own time.

**The load.** Each run waited for TWO consecutive one-minute load samples under 1.5, sixty seconds
apart — two and not one, because the owner flew a window while this was queued, and a lull between
its legs reads low for a single sample. The distribution run started at a one-minute load of
**1.18** and ended at **1.52**; the parent-mesh run started at **1.04**. MEASURED. (The waiter's
45-minute budget was extended once to 90 minutes for that flight, on the coordinator's instruction.)

**The rounds.** Five rounds on the named chunk and on the parents, the mean printed. ONE round per
chunk in the distribution — a distribution wants many chunks, not many rounds; single-round noise
shows in the phase split of any one row, never in the quantiles.

**The stands.** Derived exactly as the two flights derive them: the day side on the equator, the
star 15° over the horizon (`crates/bins/tests/terrain_moving_eye.rs` for the walker,
`crates/bins/tests/terrain_pictures.rs` for the six picture stands — ground, hill 300 m, aloft
60 km, orbit 2 000 km, the cube-edge seam, and the far stand at a quarter of the frame). The descent
reads only the EYE POINT, never a facing, so the nose and its tilt are not restated. The walk's
stand and the ground stand share one eye by construction; both are run, and they agree to within
2 %, which is this bench's own repeatability.

**The wanted set.** `LadderView::default()` — the UNBOUNDED ask, which is what the walk leg flew
(`bound_samples: 0` on that leg in all three pair flights). Up to 100 chunks per rung per stand, at
rungs 0 to 3, evenly sampled across each ring.

**The parent cache** is set to the renderer's own budget — 256 MB with a floor of one chunk's eight
parents per worker (`vd_client_render::terrain::PARENT_CACHE_BYTES`) — and it is warm across a
stand, exactly as the lane's workers share one.

**What "total" means.** The worker's own cost of one chunk: the box, the parents it must raise, the
extraction and the geometry. It is reported two ways — **COLD** (the parents are missed and built)
and **WARM** (the lane already holds them).

---

## 2. The named chunk: `ChunkKey { face: NegX, rung: 1, x: 39847, y: 28299, z: 2146 }`

### 2.1 Where its time goes — MEASURED, 5 rounds

| Phase | Time | Share of the cold build |
|---|---|---|
| The columns (one height, direction and biome per column) | 0.93 ms | 1 % |
| The core cells and the carve | 1.97 ms | 2 % |
| The halo and the box's set-up | 1.13 ms | 1 % |
| **THE EIGHT PARENT MESHES** | **81.35 ms** | **80 %** |
| The extraction (the surface nets) | 1.67 ms | 2 % |
| The geometry (positions, morph targets, normals, skirts) | 14.92 ms | 15 % |
| **TOTAL, the parents COLD** | **101.97 ms** | **12.7 × the budget** |
| **TOTAL, the parents WARM** | **20.62 ms** | **2.6 × the budget** |

15 187 vertices, 29 420 triangles, 805 152 bytes uploaded (786 KB). MEASURED.

**This IS the flight's chunk.** The flights read 111–112 ms; this bench reads 101.97 ms on the same
key on a quiet machine, and 111.69 ms on the slowest chunk of the seam stand's rung 1 (§3). The two
agree within 9 %. MEASURED.

**The budget's own two phases are 5.73 ms.** `07_measurements_costs.md` §7.1 accounts a chunk as
"sample box + extraction" and reads 3.30 ms for a rung-0 surface chunk. The same two phases of THIS
chunk read 4.03 + 1.67 = **5.73 ms** — the right order, and comfortably under 8 ms. **The 8 ms
budget was set on a quantity that is 6 % of what a worker pays.** MEASURED.

### 2.2 Why it is dense — MEASURED

It is **not a cliff**: the surface crosses exactly ONE chunk along the radial (chunks 2146 to 2146),
and the whole column holds **16.9 m** of relief (6 345 721.0 m to 6 345 737.9 m sampled).

It is **not a seam**: the face holds 80 335 chunks per edge at rung 1, and this one stands at
(39 847, 28 299) — nowhere near an edge.

It is **cave-riddled.** Of its 3 844 radial columns:

| What the column holds | Count | Share |
|---|---|---|
| ONE surface (rock under air, one sign change) | 1 919 | 50 % |
| **A CAVE (three sign changes or more)** | **1 925** | **50 %** |
| Solid rock all the way | 0 | 0 % |
| Air all the way | 0 | 0 % |

The deepest column changes sign **six times**. 161 927 of its 238 328 cells are rock (68 %). Both
carves are on at rung 1 (`tubes_carve_at` true, `caverns_carve_at` true). MEASURED.

Six of its columns, bottom first (`R` = rock, `A` = air, the number is cells):

```text
  ( 0, 0) R40 A1 R13 A8     <- a cave roof ONE cell thick, then rock again, then the sky
  (15,15) A5 R46 A11
  (31,31) R48 A14           <- a plain surface column
  (47,47) R46 A16
  (61,61) R11 A11 R24 A16   <- an 11-cell cavern under 24 cells of rock
  (31, 0) R9 A7 R36 A10
```

Half the columns carry a second and a third surface, so the extractor emits vertices for a cave
roof, a cave floor AND the ground — 29 420 triangles. **The CARVE makes this chunk dense, not the
landform.** That matters for 8a: 8a adds roughness to the surface, and this chunk is already dense
without it.

---

## 3. The distribution — 1 600 chunks, MEASURED

100 chunks per rung per stand, rungs 0 to 3, evenly sampled across each ring. The **aloft**,
**orbit** and **far** stands want NOTHING at rungs 0 to 3 (they want rungs 6–12, 11–14 and 16–17),
so four stands carry the table.

### 3.1 Per stand and rung, the parents COLD

| Stand | Rung | Chunks | Over 8 ms | Median | p90 | p99 | Max | box | parents | extract | geometry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walk | 0 | 100 | **97** | 35.20 ms | 65.75 | 71.14 | 71.73 | 3.33 | 28.97 | 1.19 | 4.09 |
| walk | 1 | 100 | **96** | 45.19 ms | 81.49 | 103.42 | 106.25 | 3.70 | 32.35 | 1.41 | 8.56 |
| walk | 2 | 100 | **97** | 37.09 ms | 58.15 | 65.77 | 67.07 | 4.20 | 21.47 | 1.72 | 9.72 |
| walk | 3 | 100 | **84** | 17.17 ms | 26.38 | 30.15 | 31.18 | 4.05 | 9.56 | 1.11 | 2.04 |
| ground | 0 | 100 | 97 | 35.56 ms | 65.33 | 71.01 | 72.67 | 3.37 | 29.11 | 1.19 | 4.12 |
| ground | 1 | 100 | 96 | 45.08 ms | 82.69 | 102.90 | 107.24 | 3.77 | 32.72 | 1.43 | 8.70 |
| ground | 2 | 100 | 97 | 37.86 ms | 58.89 | 66.62 | 67.81 | 4.25 | 21.77 | 1.74 | 9.86 |
| ground | 3 | 100 | 84 | 17.19 ms | 26.32 | 30.13 | 31.00 | 4.06 | 9.59 | 1.11 | 2.04 |
| hill | 0 | 100 | 92 | 35.71 ms | 52.34 | 72.27 | 77.67 | 3.47 | 25.41 | 1.21 | 4.47 |
| hill | 1 | 100 | 98 | 48.06 ms | 83.95 | 103.00 | 103.12 | 3.75 | 36.21 | 1.41 | 8.51 |
| hill | 2 | 100 | 94 | 36.81 ms | 58.06 | 67.51 | 69.61 | 4.14 | 21.32 | 1.70 | 9.39 |
| hill | 3 | 100 | 87 | 18.42 ms | 28.93 | 33.14 | 33.42 | 4.07 | 11.38 | 1.11 | 1.99 |
| **seam** | 0 | 100 | 99 | 38.48 ms | 65.79 | 75.85 | 77.45 | 3.56 | 31.64 | 1.22 | 4.35 |
| **seam** | **1** | 100 | **100** | **68.13 ms** | 96.87 | 109.65 | **111.69** | 3.91 | **50.68** | 1.44 | 9.62 |
| **seam** | 2 | 100 | 97 | 40.60 ms | 63.54 | 70.80 | 72.39 | 4.34 | 26.54 | 1.68 | 9.03 |
| **seam** | 3 | 100 | 97 | 21.89 ms | 31.13 | 35.44 | 36.48 | 5.28 | 13.21 | 1.11 | 2.11 |

The seam stand is the worst at every rung, and **every one of its hundred rung-1 chunks is over
budget**, with a median of 68 ms. Its box phase is also the most expensive, because a partial chunk
at a face's far edge reads the partner face's columns as well.

### 3.2 The whole set

| | Median | p90 | p99 | Max | Over 8 ms |
|---|---|---|---|---|---|
| The parents COLD | 33.14 ms | 66.26 ms | 100.05 ms | 111.69 ms | **1 512 of 1 600 (94 %)** |
| The parents WARM | 9.68 ms | 20.60 ms | 28.71 ms | 33.02 ms | **1 030 of 1 600 (64 %)** |

Time by phase over all 1 600 chunks: **the box 11 %, THE PARENTS 69 %, the extraction 4 %, the
geometry 17 %.** MEASURED.

**The cache is full.** At each ground-level stand the four rings alone filled the renderer's 256 MB
parent cache — 947 to 1 060 meshes, 255 MB held — and it was evicting before the stand ended. The
hit rate is 75 % at the walk and hill stands (4 815 hits, 1 585 builds) and 68 % at the seam (4 257
hits, 1 983 builds). MEASURED.

### 3.3 The ten slowest chunks measured

| Stand | Total | Rung | Chunk | columns | core | halo | **parents** | extract | geometry | Triangles |
|---|---|---|---|---|---|---|---|---|---|---|
| seam | **111.7 ms** | 1 | NegZ 17891/80331/2124 | 1.04 | 2.45 | 1.23 | **84.1** | 1.97 | 20.91 | 37 958 |
| seam | 109.7 ms | 1 | NegZ 17885/80329/2124 | 1.10 | 2.31 | 1.23 | **82.0** | 1.97 | 21.01 | 38 435 |
| ground | 107.2 ms | 1 | NegX 39830/28273/2148 | 0.99 | 2.38 | 1.13 | **83.8** | 1.84 | 17.11 | 35 861 |
| walk | 106.3 ms | 1 | NegX 39830/28273/2148 | 1.00 | 2.32 | 1.19 | **82.7** | 1.86 | 17.19 | 35 861 |
| seam | 104.8 ms | 1 | NegZ 17868/80331/2124 | 1.05 | 1.70 | 1.19 | **84.9** | 1.67 | 14.29 | 28 818 |
| seam | 104.1 ms | 1 | PosX 17875/15/2124 | 1.06 | 2.66 | 1.20 | **80.0** | 1.73 | 17.45 | 33 444 |
| walk | 103.4 ms | 1 | NegX 39827/28279/2148 | 0.96 | 3.49 | 1.22 | **82.8** | 1.72 | 13.27 | 29 063 |
| hill | 103.1 ms | 1 | NegX 39834/28273/2148 | 0.99 | 2.26 | 1.22 | **79.2** | 1.77 | 17.70 | 34 438 |
| hill | 103.0 ms | 1 | NegX 39829/28274/2148 | 0.96 | 2.32 | 1.07 | **83.6** | 1.64 | 13.36 | 27 978 |
| ground | 102.9 ms | 1 | NegX 39827/28279/2148 | 0.96 | 3.55 | 1.16 | **82.4** | 1.66 | 13.22 | 29 063 |

Every one of the ten is **rung 1**, and in every one the parents are 77–82 % of the time. The
chunk's own extraction never passes 1.97 ms. MEASURED.

**Why rung 1 and not rung 0.** A chunk's parents live one rung COARSER, and a coarse chunk covers
eight times the volume with the same cell count, so fewer of its cells can be skipped and more of it
crosses the cave band. The measured mean parent cost per chunk runs 29.0 ms at rung 0, **32.4 ms at
rung 1**, 21.5 ms at rung 2 and 9.6 ms at rung 3 on the walk stand.

---

## 4. The parent path

### 4.1 What a hit saves — MEASURED, 5 rounds, on the named chunk

| | Time |
|---|---|
| The geometry step with the 8 parents WARM | 16.55 ms |
| The geometry step with the 8 parents COLD | 97.93 ms |
| The parents' own builds (8 meshes, 338 937 triangles) | 80.43 ms |
| **A hit SAVES** | **81.38 ms — 83 % of the cold build** |

One parent mesh is about **10 ms and 42 000 triangles** — bigger than the chunk it serves (29 420).
Eight of them are 338 937 triangles: **a chunk raises eleven times its own triangle count in
parents.** MEASURED.

**The flights say the same.** In `pair2_on.log` the lane's hit rate rises with a leg's length, and
the mean build falls with it:

| Leg | Chunks built | Mean build | parent hits | parent builds | Hit rate |
|---|---|---|---|---|---|
| **the walk** (2 976 frames) | 142 | **26.3 ms** | 700 | 420 | **63 %** |
| a long leg | 10 487 | 15.1 ms | 93 367 | 9 716 | 91 % |
| a long leg | 8 587 | 12.4 ms | 63 476 | 4 907 | 93 % |

MEASURED. The walk is the WORST case for the cache: a slow eye asks for few chunks and each one is a
fresh neighbourhood — which is exactly the leg on which the 111 ms chunk was found.

### 4.2 What ONE parent mesh costs, phase by phase — MEASURED, 5 rounds

The eight parents of the named chunk (rung 2). `ParentMesh::build` is `sample_box` +
`extract_all_edges` + positions + the triangle bucket table + smooth normals.

| Parent (face x/y/z) | box | all-edges extraction | positions + buckets + normals | total | vertices | triangles |
|---|---|---|---|---|---|---|
| NegX 19923/14149/1073 | 4.16 | 1.71 | 1.75 | 7.62 ms | 15 555 | 29 958 |
| NegX 19923/14149/1072 | 6.49 | 2.46 | 3.57 | 12.52 ms | 30 209 | 58 409 |
| NegX 19923/14150/1073 | 3.60 | 1.65 | 1.66 | 6.91 ms | 13 688 | 26 377 |
| NegX 19923/14150/1072 | 8.70 | 2.52 | 3.45 | 14.67 ms | 29 999 | 57 744 |
| NegX 19924/14149/1073 | 3.54 | 1.64 | 1.73 | 6.91 ms | 14 532 | 27 935 |
| NegX 19924/14149/1072 | 5.61 | 2.43 | 3.54 | 11.58 ms | 29 524 | 56 885 |
| NegX 19924/14150/1073 | 3.60 | 1.55 | 1.28 | 6.43 ms | 11 804 | 22 602 |
| NegX 19924/14150/1072 | 8.38 | 2.50 | 3.63 | 14.51 ms | 30 540 | 59 027 |
| **All eight** | **44.08 ms (54 %)** | **16.44 ms (20 %)** | **20.62 ms (25 %)** | **81.14 ms** | | **338 937** |

**Two findings here.**

1. **A parent's biggest phase is its own SAMPLE BOX — 54 %.** The recipe, not the mesh. Every one of
   those eight boxes costs 3.5–8.7 ms, MORE than the fine chunk's own 4.03 ms box, because a coarser
   chunk skips fewer cells and crosses more of the cave band.
2. **`extract_all_edges` is NOT the expensive half.** Measured against `extract` on the SAME box it
   costs almost the same (1.71 vs 1.71 ms; 2.46 vs 2.40 ms), for about 1–2 % more triangles. The
   all-edges rule is not a cost problem. MEASURED.

### 4.3 The cold build, fully expanded

Putting §2.1 and §4.2 together, the named chunk's 101.97 ms cold build is:

| The real phase | Time | Share |
|---|---|---|
| **The recipe — the chunk's own sample box** | 4.03 ms | 4 % |
| **The recipe — the EIGHT PARENTS' sample boxes** | **44.08 ms** | **43 %** |
| The parents' all-edges extraction | 16.44 ms | 16 % |
| The parents' positions, buckets and normals | 20.62 ms | 20 % |
| The chunk's own extraction | 1.67 ms | 2 % |
| The chunk's geometry (morph targets, normals, skirts) | 14.92 ms | 15 % |

**THE RECIPE IS 48.11 ms — 47 % of the build.** A cold chunk runs the sample box **NINE times**:
once for itself and once for each of its eight parents. Nobody has been counting the eight.

Over the whole 1 600-chunk set the same arithmetic gives the recipe **about 48 %** of all the time
(11 % the chunks' own boxes, plus 54 % of the parents' 69 %). That share is ESTIMATED — the 54 %
split is MEASURED on the named chunk's eight parents only, and the other rungs are not measured.

---

## 5. The card's reachable share

**What the card does.** Ruling F9 item 2 makes the card a SECOND BUILDER that computes the SAMPLE
BOX byte for byte and hands it to the same geometry step (`vd_terrain::gpu`; `geometry_from` takes
"a box somebody already sampled, whoever computed it"). The card does the cells and the plan. It
does not do the extraction, and it does not do the geometry.

### 5.1 The share as the card is wired TODAY — it takes the chunk's own box only

| | Of the whole 1 600-chunk set | Of the named chunk, cold | Of the named chunk, warm |
|---|---|---|---|
| The box — what the card takes | **11 %** | 4.03 ms of 101.97 ms (**4.0 %**) | 4.03 ms of 20.62 ms (**20 %**) |
| CPU either way | **89 %** | 97.94 ms (96 %) | 16.59 ms (80 %) |

**The card's own cost, from the flights** (MEASURED, three pair flights, `card_device_timed: true`):
the card's device time for one box reads **0.107 to 0.142 ms** (`card_per_box_ms`), and the pipeline
the bounded ask reads states **69 to 88 chunks a second** — 11 to 14 ms of round trip per chunk. The
card computes in a tenth of a millisecond a box the CPU takes 4 ms over, and gives it back on a trip
measured in milliseconds.

### 5.2 The share the card COULD reach — if it also took the parents' boxes

A parent's sample box is the SAME CALL on the same recipe (`ParentMesh::build` opens with
`sample_box`). Nothing about it is special.

| | Of the set | Of the named chunk, cold |
|---|---|---|
| The card today (the chunk's box) | 11 % MEASURED | 4.03 ms, 4 % MEASURED |
| **The card pointed at the parents' boxes too** | **~48 % ESTIMATED** | **48.11 ms, 47 % MEASURED** |

**This is the single biggest lever this measurement found, and it needs no new machinery** — only
the parent build routed through the same second builder.

### 5.3 Would a "chunk over budget" trigger have caught the 111 ms chunk?

**Yes as a detector. No as a cure, the way the card is wired today.**

- **It WOULD have fired.** On the walk leg of all three pair flights the card was judged on every
  frame and stood down on every frame — `card_judged: 2976, card_stood_down: 2976` (pair2),
  2 979/2 979 (pair1), 2 972/2 972 (pair3). The stand-down rule keys on QUEUE DEPTH, and a walker at
  1.4 m/s never has a deep queue, so today's rule is silent exactly where the 111 ms chunk lives. A
  trigger keyed to a chunk over budget fires there at once: **96 of 100** rung-1 chunks at that
  stand are over 8 ms, and **100 of 100** at the seam stand. MEASURED.
- **It would NOT have cured it today.** Handing that chunk's own box to the card removes **4.03 ms
  of 101.97 ms**. The chunk still costs 97.94 ms and is still **12.2 × over budget**.
- **It could cure half of it tomorrow.** With the parents' boxes on the card as well, the same
  trigger moves **48.11 ms of 101.97 ms** off the CPU — the chunk falls to about 54 ms, still over
  budget but no longer fourteen times it. ESTIMATED from the measured split; the round-trip cost of
  eight extra boxes is NOT measured and must be before this is promised.

**What the card can never help with:** the extraction and the geometry — the morph target's ray cast
per vertex into a coarse mesh above all. Those are 33 % of the build and they are the phases that
grow with roughness.

---

## 6. Which phase dominates, and which grows with roughness

**The arc's claim** (`00_proposed_landforms.md` §1.4, `07` §7.1): the term that grows is the
extraction, and the densest measured chunk goes from 5.22 ms to 7.09–8.07 ms at twice the vertices.

**What is measured now.** The chunk's own extraction is **4 %** of the set and **2 %** of the named
chunk. Doubling it adds about 1.7 ms to a 33 ms median. On its own the arc's term is no longer the
question.

**The arc is right about the KIND of work and wrong about where it is charged.** Four phases read
what 8a changes:

1. **The recipe, nine times over (47 %).** 8a's roughness — the slope spectrum, the per-column
   roughness factor, the ridged middle band — is read in the COLUMN and CELL passes, which is the
   sample box. §7.1 priced the landform work at **+0.45 ms per sample box**. A cold chunk runs NINE
   boxes, so the same estimate is **+4.05 ms per CHUNK**, not +0.45. ESTIMATED, by carrying §7.1's
   own per-box figure across the nine boxes this bench counted.
2. **The chunk's own extraction (4 %)** — grows about linearly with vertices.
3. **The parents' all-edges extraction (16 %)** — the same work on a larger mesh; grows about
   linearly with vertices. MEASURED as barely dearer than the surface-only extraction (§4.2), so
   the all-edges rule itself is not the cost.
4. **The geometry's morph targets (15 %)** — every vertex casts a radial ray into the parents'
   triangle buckets, so it reads the FINE vertex count AND the COARSE triangle count. Roughness
   raises it on both sides: **about quadratically**. ESTIMATED from the shape of the code
   (`geometry_from`'s per-vertex `radial_hit` over `parent_meshes`); the exponent is NOT measured,
   and 8a must measure it.

**The honest statement.** The extraction is not the dominant phase — **the recipe is, at 47 %, and
it is charged nine times per chunk.** The phase with the worst growth law is the morph target, which
nobody has priced. Both were invisible while the budget named only "sample box + extraction" of ONE
box.

---

## 7. THE RECOMMENDATION ON L25 — five lines

1. **KEEP the 8 ms number and RESTATE WHAT IT COVERS.** Today it names one box and one extraction —
   5.73 ms of the named chunk, comfortably inside 8 ms — while a worker pays 101.97 ms cold and
   20.62 ms warm. A budget that governs 6 % of the cost governs nothing. MEASURED.
2. **MAKE THE BUDGET THE WHOLE BUILD, and publish this base with it:** median 33.14 ms cold /
   9.68 ms warm, p99 100.05 ms, max 111.69 ms; 94 % over 8 ms cold and 64 % over warm, on 1 600
   chunks across four stands. 8a's delta is measured against these numbers, on this bench. MEASURED.
3. **SHRINK THE PARENT PATH FIRST, never the extraction.** The parents are 69 % of the set and 80 %
   of the named chunk; a hit saves 83 %; and the renderer's 256 MB cache is already FULL at ONE
   stand's four rings (947–1 060 meshes, 68–75 % hit). Raising that cache, and giving the morph a
   cheaper target than a ray cast into a coarse mesh, is worth far more than any extraction dial.
4. **ADD the second stand-down trigger AND POINT THE CARD AT THE PARENTS' BOXES.** The trigger alone
   is only a detector — it fires where the queue rule is silent (the card stood down on all 2 976
   walk frames while 96 of 100 rung-1 chunks were over budget) but buys 4.03 ms of 101.97 ms. With
   the parents' boxes on the card it buys **48.11 ms of 101.97 ms** — the recipe is 47 % of the
   build and the card already computes that exact call. Measure the eight extra round trips first.
5. **DO NOT MOVE THE NUMBER BEFORE 8a MEASURES ITS OWN DENSEST CHUNK.** Raising 8 ms today hides a
   47 % phase nobody was counting; shrinking it squeezes a 4 % phase. 8a's first measurement must be
   this same bench on 8a's ground, reading the morph target's growth law off it — that is the only
   number that can honestly move the budget.

---

## 8. What this measurement adds

| File | What |
|---|---|
| `crates/bins/examples/chunk_budget.rs` | NEW. The measurement example: the named chunk's phases and anatomy, the distribution over the seven stands, the parent hit, and one parent mesh's own phases. An example only — no product code changed. |
| `docs/investigation/2026-09-08/landforms/12_chunk_budget_base.md` | NEW. This document. |

**Reproduce it:**

```text
cargo run --release -p vd-bins --example chunk_budget
# knobs: VD_BUDGET_ROUNDS (default 5), VD_BUDGET_PER_RUNG (default 100),
#        VD_BUDGET_STANDS (a comma-separated list of stand names)
```

**What is NOT measured here, and is owed:** the morph target's growth exponent; the parents' box
share at rungs other than 2; the round-trip cost of putting eight extra boxes on the card; and the
same distribution with the BOUNDED ask instead of the unbounded one.
