> ### ⚠ INVESTIGATION BASE — NOT A DECISION RECORD
>
> **This document is input to an investigation, not the output of one.** It is analysis produced to
> explore a problem space, and it is deliberately more confident in tone than its status warrants —
> that was useful for finding defects and is misleading for planning.
>
> **Nothing here is committed.** Every ruling, recommendation, number and "settled" verdict is a
> *proposal to be re-validated by the pre-implementation investigation for its phase*, including the
> owner rulings recorded at the top of `block_system_design.md`, which record the owner's direction at
> the time rather than a frozen commitment.
>
> **The binding specs are elsewhere:** `docs/design/PLAN.md`, `docs/design/roadmap.json`,
> `docs/design/integration.json`, `docs/design/DEFERRED.md` and the hardened subsystem designs in
> `docs/design/`. Where this document and a binding spec disagree, **the binding spec wins** until an
> investigation says otherwise.
>
> **What this document IS good for:** the prior art it collected, the arithmetic it did, the failure
> modes it found, the one-way doors it named, and the questions it framed. Reuse those. Re-derive the
> conclusions.
>
> See `docs/investigation/README.md`.

# The decision board — what is settled, what is open, and what happens in what order

**Date:** 2026-08-04. **Replaces §8.1 (the decision register), §8.2 (the one-way doors) and §8.3 (the
slice plan) of `docs/investigation/block_system_design.md`.** Those three subsections were written before
a day of rulings that materially changed the board; where this document and they disagree, this document
is the later one. **§8's other subsections still stand** — §8.4 (the benchmarks and gates), §8.5 (the new
deferred-registry entries) and §8.6 (the cross-section consistency report) are unchanged **except where
§3, §6 and §7 below correct them by name**.

**Two status notes, both load-bearing, before anything else.**

1. **The ten source documents moved today** to `docs/investigation/`, and that folder's own `README.md`
   declares them **not binding** — "analysis, prior art, arithmetic, failure modes, one-way doors and
   framed questions… to be re-validated before implementation." That is the more recent statement and it
   wins. So read this board as *the register you answer rows in*, not as a plan of record: an answer here
   becomes real only when it is promoted into `docs/design/`. The nineteen owner rulings are a record of
   intent, not a frozen commitment. Nothing below asks you to implement; it asks you to decide, and it
   tells you when each decision stops being cheap.
2. **This file was written to `scripts/`** because that is where the run was told to put it; the README
   announces it landing in `docs/investigation/` beside the other ten. Move it there — it is one `git mv`
   and no content depends on the path.

---

## How to use this document

Read **§1 once** and stop carrying any of it in your head — those questions are answered and the answer
is recorded with whoever answered it. Then work **§2 top-down**, because it is ordered by what answering
late actually costs, and the rows at the top are the ones where being wrong means regenerating a galaxy
underneath player builds rather than editing a file. **§5 is the shortest path** — if you only act on two
sections, act on §2's first nine rows and §5.

---

## §1 What is settled

Everything in this table is closed. The right-hand column names what closed it, so a citation survives.
Where a section of the design still argues one of these, that section is stale — the ruling wins, and §7
lists the specific places that still need striking.

### The owner's nineteen rulings

| # | What is settled | Retires |
|---|---|---|
| **R1** | **The saved block change is EIGHT bytes**, not five. Fifteen bits reserved and *rejected on decode if non-zero* (fourteen after R6 — see G4 in §7). The three extra bytes cost ≈300 MB across 100 M edits, which is noise against a format migration. Also settled by the same row: identity is ONE combined number, never independent substance/shape fields; decoration's state bits come from the existing sparse side table | Register row 2; §2.7.1's five-byte layout; §8.6 item 1's three-way disagreement; Addendum 1 §B.4's framing; markers 3-H, 4-A, 6-7 |
| **R2** | **Light has a level AND a colour**, both ordinary signals where the block is functional. A torch carries standard values on its substance row and does not vary | Register row 26 |
| **R3** | **Colour lives where you see it; brightness lives where it is simulated.** The server's gameplay field stays single-channel brightness + sky exposure, because "can this grow" and "is it dark here" have no hue. Emissive blocks inside the drawn radius are real renderer lights with full colour at no propagation cost. This is what makes R2 free instead of 3× the flood work and 3× the stored state. The field is tier-0 and is never coarsened | §2's "no coloured light"; marker 2-A; door 24 |
| **R5** | **Appearance-driving signals: both lanes, narrow by default.** A functional block may have colour, emissive level and a small state index driven by signals; plain terrain never is, so the streaming cost is capped per functional block rather than open-ended | §5/§7's implicit static-appearance assumption |
| **R6** | **Provenance is THREE-state**: bits 23..22, `0 Terrain / 1 Feature / 2 Placed`, value 3 rejected on decode. Base terrain never collapses — everywhere, permanently, no radius, no exception. Biome features (tree, boulder, arch, ice shelf) collapse like built structures when cut. **Zero marginal bytes, zero wire cost, zero pyramid cost.** Ordered by anchor privilege descending, so "provenance never gains privilege" is ONE `>=` in ONE typed constructor — which structurally closes the laundering exploit: no edit path can construct `Terrain`, so built matter can never become unbreakable ground | §2.5.2's binary `placed` bit AND its unconditional-anchor rule; R6-1, R6-2 |
| **R7** | **Regrowth is both** a deliberate player activity and slow natural recovery. Near settlements harvest outpaces recovery and land stays cleared; far out old growth persists | — |
| **R8** | **Planted things store a growth STAGE (one byte, sparse); regrow-toward-baseline stores nothing** and prunes its own delta as it succeeds. Not a planting timestamp — growth is memoryless, so a binomial draw over the chunk's elapsed sleep gives the exact distribution from the already-reserved last-ticked stamp | Register row 15's direction |
| **R9** | **Reserve the terrain-instability flag now; ship it unset.** One spare bit in the per-material flag word + a per-realm cave-ins toggle, with a coherence assertion that every v1 row is stable. Collapsible terrain stays a one-material change later instead of a table migration over every saved planet | R6-6 |
| **R10** | **The support flood is axial-free, lateral-limited.** No attenuation toward the anchor along gravity; normal attenuation sideways. MAX_SUPPORT_RADIUS = 32 keeps the lateral cantilever job and **loses the flood-bound job** (the flood is bounded by the existing carry-over cell budget). Unlimited towers, limited overhangs. Without this every tree, tower, mast and antenna in the game was capped at 32 blocks tall | §2.5.2's isotropic radius; R6-5 |
| **R11** | **The strategic resource tier is never a visible block at all** — it is what a rock ASSAYS TO when broken, not something you can see. The client is never told, so no pixel is lost, zero bytes on the wire, zero stored, and "terrain never streams" survives with **no exception at all**. Common ore stays seed-derived, visible and free | Register row 9 as an architecture question; §3.8's option (C) hybrid |
| **R12** | **Grass and bushes are pure decoration and never collide** — derived, cosmetic, zero bytes stored, zero sent, deliberately different per client. Grass still parts, whispers, slows you (server-side, from the actual soil) and can conceal you. A *harvestable* bush cannot be decoration | UD-1 and its harvest-or-stand borderline |
| **R13** | **Bushes are real cells that do not collide.** Stored, authoritative, harvestable, identical for every player, drawn as a 3D mesh, spanning more cells as you place more — with an EMPTY collider. You walk through and are slowed. Minecraft's tall grass is the precedent | `collidable_decoration.md`'s borderline |
| **R14** | **Leaves are pass-through too.** Canopy cells are real cells with empty colliders — you fall through a canopy. Trunks still collide. **Deferred by the owner to just before implementation:** what collider each shape gets, whether a pass-through cell hides faces behind it, whether it counts as solid for ambient occlusion and for building on, fitted-vs-cube for rocks. See O24 — two of those four are due earlier than "just before" | Minecraft's standable leaves; the solid-canopy spec |
| **R15** | **A name is an address, not a key.** From outside a realm the takeover attack already fails three ways; from inside it was wide open (three assertions of a write ACL, zero specifications). A write capability is an **issued compiled edge, admitted once at bind time by the realm that owns the channel, never derived from a name.** Checked at bind, not per message — per-signal costs 3× to 37,720× the whole delivery constant; bind-time costs zero on the hot path. Measured total authorisation cost under **0.2% of a tick** | §7.5.2's "in-realm traffic is implicitly trusted" |
| **R16** | **Encryption and permission are different tools.** Permission stops people DOING; encryption stops people SEEING. The takeover is a doing attack, so encryption would not stop it by an inch. Encryption stays a player-built cipher block on the open radio plane | The "cypher the signals" framing |
| **R17** | **Relays are player-built functional blocks, with TIME as the cost** — infrastructure players build, own, defend, tax and destroy, reusing the existing deterministic light-lag. **But the message shape must admit a hop count, an arrival deadline and a dedup identity from the FIRST slice** or it is a migration | §7.3.2's `Relay { net }` engine plane; D-23 option (B) |
| **R18** | **Blueprints carry ALL wiring and are built at a DOCK.** The dock prices the design, takes payment, spins up a realm belonging to you, reconstructs the ship with wiring intact, optionally paying the author a royalty — **and lists every channel the design reaches outside the construct** beside the bill. Informed consent, not stripping. Three consequences that need owners: ship construction gains a place/price/profession; the payment is the anti-abuse brake on realm spin-up; author royalties need a durable identity that survives copying | The strip-external-bindings fix (proposed and rejected) |
| **R19** | **APPLIED to §7 in place.** (1) **Channel keys are SCOPED** — `Realm` (default) / `Construct` / `Net` / `Protocol` — so one player's `thrust` is no longer every player's `thrust`. `dock.*`, `sys.*`, `hull.*`, `chat.*` become compiled constants **mintable by nobody**, closing the forged-autopilot phishing hole. Zero hot-path and zero wire cost. (2) **The autopilot handshake is re-ordered verify → resolve → display**: signature checked before the dialog renders; the offer resolved into the actuator set by walking the ship's own subscriber list; the dialog composed by the SHIP's realm; the effect set re-resolved at mint | §7.4's bare-name key; §7.12.2 steps 2 and 4; D-2, D-3 |
| **R4** | **OPEN — the one ruling that is not settled.** Multi-block composites (place tree blocks, get a real tree; more blocks, a bigger one; rock blocks, a non-blocky rock; **all of it collides**) is raised and undesigned. See O26 | — |

### Register rows closed, dissolved or answered

| Row | What is settled | By |
|---|---|---|
| **7** | **"No detail levels" bans nothing — the ban is retracted.** One ladder, one rung per power of two; at tier L a cell is 2^L m and a chunk is still 62³ cells; container, palette, mesher, material tables and shape catalogue **identical at every tier**; a tier is a number on a chunk address, never a second code path, so HR3 holds by construction. For fractal terrain the correct detail level is **exact and cheaper** — tier L drops the L highest-frequency octaves, which is simultaneously the right anti-aliasing filter, the right detail level and a saving. **~379 chunk columns per tier, ~505 in tier 0; a 100 km view is ~3,158 columns ≈ 6,800 chunks ≈ 270 MB packed, against 8.17 million columns without tiers.** Resident chunk count is logarithmic in view distance, not quadratic. Seams closed by skirts (~20 lines, no case table, no dependency on the neighbour's tier). No pop, ever — a dither crossfade between two rungs of the *same* terrain. The realm proxy becomes the coarsest rung. **Exclusion list, binding:** collision is always tier 0; anything that can shoot or be shot; anything a signal or functional block depends on; anything whose appearance carries actionable gameplay information. *Two players at different distances may see different terrain DETAIL; never different terrain FACTS.* The test is **derived-from versus substituted-for** | Owner ruling, 2026-08-03; Addendum 2 |
| **4** | **Planet radius is per-body DATA, not a decision.** `R = 124·m/π` metres for any integer m — a 39.47 m step on which every real body lands within 20 m and Earth within 0.4 m (m = 161,412 → 6,371,000.4 m). The "unique value" of 161,671 m was an artefact of an unnecessary power-of-two constraint on chunks-per-face, and is **withdrawn**; its rider ("take 81 km if row 7 holds") is **void**. Residual only: what the *starter* world gets (O42) | Addendum 1 §C |
| **5** | **Gravity is derived from an authored per-body MASS.** Surface gravity, escape velocity, sphere of influence, orbital period and atmosphere scale height are all derived and never authored. `g = (4/3)·π·G·ρ·R`, so at equal density gravity is proportional to radius: a 162 km body at Earth density pulls at 1/39 g, at a typical 3,000 kg/m³ at 1/72. Both behaviours are available per planet — a curated starter world can be written dense. Honest cost, now local: SOI scales as (m/M)^0.4, so ×72 mass ⇒ ×5.5 SOI. Residual only: the *band* the generator may draw from (O43) | Addendum 1 §D |
| **23** | **The 2–50 km altitude band is solved** — it is the rungs between tier 0 and the body's face-covering rung, and the realm proxy is the rung above that. The two grounds on which a chunk tier to orbit was rejected are both supplied: edits fold in through the pyramid, and the seam is closed by skirts, **which work precisely because the geometry is blocky** (two rungs of axis-aligned quads meet in a vertical step, not an interpolated curve, so a downward apron closes the crack exactly). **The analytic displaced-sphere shell is NOT BUILT** — one representation fewer beats the saving, and reserve nothing | Addendum 2; D-RND-8 discharged |
| **40** | **Virtual texturing: do not build.** (A) hardware sparse is *not available* — the vendored wgpu 27.0.1 exposes no sparse/tiled/partially-resident feature at all. (B) solves a problem we do not have — UV is exact integer modulo of a 4-cell world period, so the 512 MiB material array is **already the complete resident set for an entire Earth-sized planet at any view distance** and the ladder adds not one byte. (C) saves ~4 taps over ~1.45 M pixels ≈ 0.03–0.06 ms, below noise, and thrashes in exactly the sustained-edit regime. **Trigger, so it is a measurement not a mood: fragment tap count above ~16, or the composite above ~1 ms.** The cost of not recording this is someone spending a month on (B) | Row 40 |
| **9** | **Dissolved into R11's reframe** plus five cheap rows (O8, O9, O44). Nothing inside the client can hide anything — the secret is the function's OUTPUT, and a client that can draw the ore can enumerate it, so obfuscation, WASM, bytecode VMs, server-supplied shaders, per-session code streaming, white-box crypto (broken by automated attacks since CHES 2016) and consumer enclaves (SGX removed from Intel client CPUs in 2021, and an enclave cannot render) are all closed with no engineering to spend. The concealed half is **a drop table, not an overlay** — it never enters the block grid, which is what makes the cost zero and structurally closes the pyramid leak, the CAS break, the crack-timing oracle and the rim-warp tell at once. **The exposure-reveal lane is rejected**: reveal traffic is MOVEMENT-bound, not dig-bound — 54 cores at 10,000 walking players, 2,171 at 10,000 at 200 m/s. Also refuted: "ore concealment protects the price signal" — EVE runs the genre's benchmark player-driven economy on fully public static belts. Decide O8 on the *gameplay* argument | R11; `concealed_resources.md` |

### Measured, so no longer arguable

| What | The number | Consequence |
|---|---|---|
| **Terrain noise** | **12.19 ns per evaluation; 0.885 ms per complete 62³ surface chunk** (Apple M4 Pro, one core, rustc 1.94.1, opt-level 3, fat LTO, black_box on every accumulator, scalar f64, no transcendental, coarse-lattice 3D cave field sampled every fourth cell) | **The generation risk is CLOSED.** A 6,800-chunk 100 km view is ~6 s on one core, 0.75 s on eight, *before* the octave saving. The multi-second horizon this design was written around does not exist. D-WLD-1 🟩 |
| **Crossfade band** | f = 0.15 → **7,353** resident chunks (passes ≤ 7,500); f = 0.20 → **7,510** (fails). Steady state 6,821 | **The shipped default is 0.15 because 0.2 does not pass.** The gate bounds the tuning value, not the reverse |
| **The maximum flight speed's binding term** | Not noise. **Mesher throughput + residency-under-motion**, corroborated independently at **240 m/s (864 km/h)** from the terrain UPLOAD budget as the tightest of four ceilings | Door 42's derivation moved. See O31 |

### Verified engine and hardware facts — closed on fact, not taste

| Refused | The fact that decides it |
|---|---|
| Hardware ray-traced global illumination | The graphics API reports **zero acceleration structures** on the development backend; NVIDIA-only in practice; alpha masks unsupported, which deletes the foliage tier |
| Multisampling / alpha-to-coverage foliage | Foreclosed in engine code — the occlusion pass and the temporal resolve both hard-require it OFF — and by our own dither crossfade |
| Screen-space reflections | Verified in source: "currently only supported with deferred rendering"; our terrain is a forward material |
| Planar reflections | A second full scene render, **and geometrically wrong on a small planet**: sagitta 0.77 m at 500 m and 3.09 m at 1 km on a 161,671 m body, against 0.08 m at 1 km on Earth |
| Baked irradiance volumes through the documented workflow | The engine ships no baker, and our world is player-mutable at metre granularity. Reject the workflow, keep the container |
| Clustered decals | Disabled on macOS/iOS — the development machine |
| Automatic exposure as the shipped default | Stars vanish on warp arrival — a direct hard-rule violation |
| Columnar (Capacitor-style) block storage | **Row reordering is not expensive here, it is UNDEFINED** — in a spreadsheet a row is an unordered bag; here **the row IS the position**, so shuffling would not re-encode the world, it would build a different one. Both techniques that make the format fast are already present under other names (joint dictionary encoding; column separation), and joint dictionary beats per-field columns provably |
| A surface-only / shell world representation | You already have something **strictly stronger**: nothing about the natural landscape is ever saved or sent. **Zero is not a number a shell can beat.** "Within a certain depth" has no finite answer — at coarse rungs a hollow mountain would need a 128 m shell to still look like a mountain, and for digging you can always go one block deeper |
| SVDAG for the live world; 1-D run encoding | Both explicitly do-not-build. **Ship the greedy 3-D BOX encoding instead** (16 B/box, ~250 lines, applied to the delta, the pyramid and blueprints alike) — 244× smaller on a typical hollow building and the strongest single improvement in the storage analysis. The compactor's "smallest wins" means it can never lose |
| A raymarched sky as the shipping path; cloud raymarching as the first cloud; billboard clouds; a spectral ocean; a particle library now; per-cell rounded rock templates; puddles and rain-laid snow as real blocks | 3–6 ms is a third of the owned frame and the two sky modes differ visibly so a hard switch pops; no closed form at a point means boats float where the water is not; 21–131× the triangle count |

### Adjudications that stand (from §8.6, unchanged)

- **Two new crates, not four**, and the crumbling-edge function lives in **core** (it is a pure function of a corner identity, an occupancy byte and a material amplitude table). That leaves exactly ONE new crate: mesher + decoration evaluator, Tier-A, added to the coverage list in the commit that creates it. **The ladder adds no crate.** Naming hazard to fix on the way in: the cube-sphere mapping curve and the edge displacement are both called "warp".
- **§4's shape row supersedes §2's.** §2's volume field cannot express one third, one sixth, two thirds or five sixths — four of the twenty-three catalogue entries — and its millidegree slope field is a tripwire wired to nothing on a warped grid (the same wedge presents 43.5°–56.1° across one planet). **A cell is the unit of matter** stands: one block yields one unit and weighs one nominal cell's worth wherever mined, removing a 2.5× position-dependent mining exploit.
- **The geometry seam has NINE operations**, not six or eight, all taking the tier as a parameter, with the exhaustive gate running at more than one tier.
- **One tuning struct per subsystem, never an inline literal.** The ladder adds five fields to the render struct and no fifth struct. `crossfade_band_fraction` is ONE field, read by four named consumers and never re-declared.
- **The general rule, stated once: a subsystem may not invent its own distance ladder; it consumes the chunk's tier.** Three private carve-outs (the AO slab, decoration's two-tier scheme, the analytic shell) collapse into rungs of the one ladder.
- **The material feature ladder is REQUIRED, not "trivially permitted".** At tier 5 a chunk spans 1,984 m while the detail tile's period stays 4 m — it repeats **496 times** and hardware mip selection returns its average colour. Without the fade-out the ladder looks *worse* than no ladder. Tap count falls ~8 → ~2.
- **Decoration's rule key** re-derives against the combined identity number through the block-type table, not packed record bits. **Leaf render model** is a flag bit on §4's shape row plus a foliage shape family, not a new enum.

### The collision rule, stated once

**A thing collides if and only if it occupies a CELL, or it is an ENTITY.** It occupies a cell iff (a) a
body can rest on it, or a player can remove it and keep it removed; AND (b) it is representable in the
shape catalogue at cell scale. Below the catalogue's smallest extent (Panel 0.125 m, Nub 0.5 m) there is
no cell and therefore no collider, ever — the base cell is never subdividable. Everything else is
**derived**: it may bend, sound, mark, slow, conceal and be sampled at a point by the server; it may never
own a collider, a raycast hit, a fence, or a byte of per-instance storage, at any distance, on any
machine. Both arms are load-bearing — five things ride the entity arm (a crane cable, a hinged door whose
collider sweeps, a falling group that deliberately leaves the block field for ~2.5 s, a ship, a
projectile).

Two supporting rulings: **per-blade grass collision is rejected by 5,500×, and permanently** — the server
has no camera so it must use the plateau set (π·64²·100 = **1.29 million colliders**, 150 MB against a
≤27 MB budget, rebuilt per tick) — but the deeper reason is that the amount of grass drawn depends on how
far away you stand and how much memory your graphics card had spare last frame, and *the server has
neither a distance nor a graphics card*; that will not change with better hardware. And **the promotion
boundary is about IDENTITY, not INFLUENCE**: a sampled scalar field derived from the block field MAY
influence the simulation; an INSTANCE may never be acted on, targeted, collided with, raycast or
persisted. Walking slower in deep grass is the *soil blocks* slowing you, computed server-side, so a
modified client that draws no grass is still slowed.

---

## §2 The open register

**Fifty-nine numbered rows (O1–O59), deduplicated across all ten documents, ordered by cost of lateness —
worst first.** The ten source documents carry seven separate registers between them (41 rows here, 17 storage,
21 rendering, 16 decoration, 35 signal, 11 concealment, 11 latency, 14 provenance); the deduplicated
total is what follows. **Seven decisions that lived only in body text and never reached any register are
included** and marked ⚠NEW — of which two must shut before P4 starts.

**Rank is derived, not judged**: a row's position is (what answering late costs) × (how many slices it
blocks). **⛔ marks a hard deadline.** Every row's `Source` lets you find the full argument.

### Band A — before the first world is generated or saved

Answering any of these late means **regenerating a galaxy underneath existing player builds, or a format
migration over every planet, ship, station and blueprint that has ever been edited**. They are one
session's work and they share one deadline.

---

**O1 ⛔ before any P4 code · What shape is a planet's grid, and how far up the detail ladder do blocks go?**

The second half is the expensive one: how far up the ladder blocks go decides how many tier bits the
*persisted chunk key* carries, and the key width is the irreversible half.
*Options:* grid — (A) cube wrapped onto a sphere / (B) literal cube planet / (C) hex grid, which forks the
ship grid / (D) flat octree with an analytic surface, which makes a wall at latitude 40 a staircase.
Ladder top — **(G1)** blocks to the face-covering rung, proxy only beyond, **five** tier bits; **(G2)**
cross to a textured analytic sphere, four bits, but a *second* surface representation; **(G3)** cross to
today's flat proxy at rung 7, three bits, and a grey ball at 200 km while the body subtends hundreds of
pixels.
*Recommendation:* **A + G1, five tier bits at chunk-key bits 60..56.** Ceiling is rung 12 for the 162 km
starter body, 18 Earth-sized, 20 for the largest addressable; five bits give 0–31 with eleven spare.
Layout: 3 (sector) + 5 (tier) + 20 + 20 + 10 = **58 of 64, six spare**, all zero-checked on decode.
*Blocks:* everything in P4. **O2, O3, O5, O6 and every slice from S0.1 onward presuppose it.**
*Cost of deferring:* no P4 code can start, and the bits are free today against a key-format migration over
every saved world tomorrow.
*Source:* register row 1 (absorbs 5-G′); doors 1 and 14.

---

**O2 ⛔ before any P4 code · ⚠NEW · Which planet radii are legal?**

Does a body's radius index `m` have to be divisible by 2^(T−2) so every rung tiles every cube face
exactly? **This is a full one-way door raised inline at §3 and it has no register row and no door row.**
*Options:* **(A) snap the ladder** — every rung tiles exactly, the seam table and eight corner columns are
byte-identical at every tier, the address function gains no branch; the cost is radius quantisation (at
T=8 the step is 2,526 m and Earth lands 158 m low, 0.0025%; at T=10 the step is 10,104 m and Earth is
5,209 m low, 0.082%). **(B) do not snap**, and accept that the outermost row of coarse cells on each face
edge is PARTIAL — a partial cell at a face edge, in the exact file Addendum 1 names as the one that will
hurt.
*Recommendation:* **(A)**, and note O1's G1 tightens it further to "the radius step is a power of two" for
any full-depth body.
*Blocks:* the legal radius set, which is persisted data from the moment a body's first block is saved.
*Cost of deferring:* a body authored under one rule cannot be re-radiused later without regenerating it.
*Source:* `[USER DECISION 3-I]`, cited at register row 1 but never registered.

---

**O3 ⛔ before any P4 code · Which curve wraps the cube onto the sphere?**

*Options:* (A) angle-proportional / (B) area-optimal / (D) edge-length-optimal. The honest picture: the
worst blocks are ~0.71 m across one way at the midpoints of the twelve cube-edge arcs — **not** at the
corners, where they are 0.93 m and equal-edged; the distorted region is twelve lens-shaped bands ~31 km
wide along 200 km arcs.
*Recommendation:* **(A)** — simultaneously at the minimum of size spread, near the minimum of shape
distortion, and angle-proportional, which is what makes the radius arithmetic exact. (B)'s "one block
mined equals one block of material" argument evaporates once a mined block yields one unit regardless of
physical size, which is already ruled. Note the research pool separately found the JCGT 2018 optimal
5th-order odd polynomial `f(a)=0.1239a+0.1305a³+0.7456a⁵` beats `tan()` on area RMSE (0.0209 vs 0.047)
using only + − ×.
*Blocks:* generation, meshing, collision, every stored edit — and now **every tier**, since a tier-L cell
is 2^L of these and inherits the distortion exactly.
*Cost of deferring:* minutes once O1 is settled; after players build, it invalidates every stored edit.
*Source:* register row 3; door 2.

---

**O4 ⛔ before the first world is saved · What is in one edit-pyramid entry, and is the pyramid stored at all?**

Player edits do **not** coarsen for free, which is why this exists: without it a dug tunnel vanishes at
200 m and reappears at 100 m and a base pops in and out as you fly. **Six sub-questions; the first two are
the expensive ones and the second was never registered (⚠NEW, it lived as `[USER DECISION 1-B]`).**

- **(a) Stored or recomputed on wake?** Stored costs at most one small entry per rung above the finest —
  twelve on the starter world's thirteen-rung ladder. Recomputed costs zero bytes and no format, but the
  work is proportional to every change ever made and is paid at exactly the wrong moment, because a world
  must be warm *before* a player can reach it: a long-inhabited planet makes its own approach stutter,
  which the seamless law forbids. **Recommend stored.** *This must be answered before (b), not after.*
- **(b) The entry's contents.** **(A1) 8 bytes** — 18-bit cell index, 16-bit dominant substance, 8-bit
  octant mask, 8-bit mean fill, 14 reserved rejected-if-non-zero. **(A2)** drop the octant mask (7 B,
  saves ~12% ≈ 105 MB over 100 M edits) and every distant player build becomes solid-or-empty — the
  documented "castles read as voxel blobs" failure. **(A3)** add min/max solid height — rejected on
  *structure*: that is a per-COLUMN quantity in a per-CELL record, so it replicates up to 62×. **(A4)**
  tag-length-value — loses the fixed width the sorted drain needs. **The genuine judgement is taste: A1
  means a distant build reads FACETED, A2 means BLOCKY.**
- **(c) How the fill is spent on a shape.** **B1** maps (octant mask, fill bucket) through a baked
  4,608-byte table folded into the block-registry hash — zero extra bytes, **no tunable threshold anywhere
  on the path**, and a mismatch is a refused connection rather than a silently wrong distant silhouette.
  B2 thresholds to solid-or-empty and is A2's failure arriving through the read side.
- **(d) Storage granularity.** **C1** one storage block per (tier, chunk) in the same database file as the
  deltas — a 1,000-block tunnel becomes ~40 records instead of ~500 keys. C2 is one key per entry, i.e.
  hundreds of millions of keys on a built planet.
- **(e) The write invariant.** **D1** only tier 0 is writable, **expressed in the store's type signature**
  so `write(tier != 0)` does not compile. D2 is a convention plus a checklist, which is what every project
  that later found a coarse-tier write in production had.
- **(f) Divergence policy.** **E1** recompute + a divergence counter the harness asserts is zero (a failing
  build in development, an invisible self-heal in production). E2 refuses to open the store, which destroys
  access to a world over reconstructible cosmetic data.

*Recommendation:* **stored + A1 + B1 + C1 + D1 + E1.**
*The numbers:* storage is proportional to edits and to the *shape* of the edit set, which spans 50×: a
compact cluster 0.143 entries/edit, a wall 0.333, a one-cell line ~1.000. A 1,000-block tunnel is **504
entries ≈ 3.9 KiB**; 100 M mixed edits ≈ **880 MB**. Walk-up cost is **≤ 1 entry write and ≤ 1 storage-block
read per rung per touched coarse cell** (the eight siblings always lie in ONE block because a chunk edge of
62 is even), **zero reads of the edit log, zero reads of any tier-0 delta**; worst case **5.85 µs at
thirteen rungs, 9.75 µs at twenty-one**, and the common single edit inside solid rock terminates at the
first or second rung at ~0.5 µs.
*Two corrections that must ride with it:* the earlier "no disk read" claim **cannot produce the record** (a
parent is a function of its eight children); and the early-exit condition must compare against the
**effective** summary (stored-or-generated), not the stored one — as written it is wrong in the common
case, because entries equal to the generator are *deleted* by the design's own prune rule. That is one
line now and an audit of a hot path plus every performance figure taken against the wrong behaviour later.
*Blocks:* whether player edits survive coarsening at all; the delta store's layout; the chunk key.
*Cost of deferring:* the same class as O5 — a migration over every edited planet, ship, station and
blueprint.
*Source:* register row 37 / `[8-A]` (absorbs 2-H, 2-I, 3-J, 4-E); `[USER DECISION 1-B]`; storage D-1, D-11;
door 43.

---

**O5 ⛔ before P6 writes one edit; the chunk format depends on it at P4 · The saved record's exact field order**

R1 settled the **width** (eight bytes) and R6 settled the **provenance field** (bits 23..22, values
0/1/2, 3 rejected). What is still a freeze is the rest of the layout, and it must be published once so
nobody implements two of them.
*Recommendation:* take `block_provenance_collapse.md` §3.1 verbatim — cell 63..46 (18), block_type 45..30
(16), orient 29..24 (6, mirror included), provenance 23..22 (2), state 21..14 (8), **reserved 13..0 (14,
MUST be zero, decode REJECTS non-zero)**. Note the reserved run is **fourteen, not fifteen** — see G4 in
§7, where R6 resolves a one-bit discrepancy that had existed since the record was first written.
*Also decide in the same sitting, because both are permanent and both are cheap:* whether a **fluid or
cosmetic rung** spends any of those fourteen bits (⚠NEW — `[USER DECISION 2-J]`, never registered; the
ruling is that cosmetic per-block state is tier-0 only, so a cleared patch of snow is invisible from a
hilltop 3 km away, and the fix costs 2 bytes on an 8-byte record, a 25% growth on permanent data); and
whether `PaletteEntry` becomes a **u32 newtype** holding exactly bits 45..22 (one shift and one mask for
the record→palette projection, a single integer compare for equality, one padding byte removed from a
structure that IS persisted inside the masked-dense encoding).
*Cost of deferring:* a migration over every saved planet, ship, station and blueprint.
*Source:* R1, R6; §8.6 item 1; `[2-J]`; R6-14.

---

**O6 ⛔ before the first world is saved · The seven storage-format freezes, in one sitting**

None of these has a door row and all seven are permanent delta data.
1. **Greedy 3-D box encoding as a fourth delta arm** — 16 B/box, ~250 lines plus a canonicality property
   test, applied to the delta, the pyramid and blueprint contents alike; 244× smaller on a typical hollow
   building. The *mechanism* is free later; **the 16-byte record layout is permanent**.
2. **An orthogonal `codec` field beside `encoding`**, always `Codec::None` today. `core/src/tlv.rs`
   already pins `CODEC_FLAGS_V1 = 0` and rejects anything else, so the pattern exists. Never add extra
   encoding arms for codecs — codec × encoding is a product, not a sum. **This is what makes O39
   (compression) safely deferrable.**
3. **Prune-on-equality as the ONE universal delta rule** — an entry exists iff it differs from what the
   generator would produce, applied to the pyramid, to `column_sky` / `snow_depth` / `grass_cover`, **and
   to the chunk record with a provenance comparison**, which is what simultaneously lets R7/R8 regrowth
   reclaim its storage and blocks a break-and-replace laundering exploit. Without it the store is bounded
   between 461 MB and 7.7 GB by an unwritten policy. *This is why R7's natural recovery is a **storage
   requirement**, not flavour: clear-cutting costs ≈2 kB of permanent record per tree, and clearing a
   quarter of a thousandth of a planet's surface fills the entire budget — logging is a bigger threat to a
   planet's storage than construction is.*
4. **Split `state_bits` into its own `(u32 local, u16 state_bits)` table** — 1.84× on the largest writer
   the table will ever have (the weather), for one table.
5. **Blueprint stamps: lazy reference or eager cell write, and the snap lattice.** Lazy buys 244–463× and
   is worth most for ships and stations, which are 100% delta today; it is explicitly **not** a stepping
   stone (a world stamped eagerly gains nothing retroactively). The lattice — 1 m (100% of the pyramid
   stored) / 2 m (25%) / 4 m (6.2%) / **8 m (1.6%)** / 16 m (0.4%) — is a taste question about how
   building *feels*, and the storage analysis explicitly declines to pick it for you. One-way in the
   tightening direction only. Reuse the existing `sha2` workspace dependency for the stamp id; do not
   introduce a second hash.
6. **The pyramid's storage-block granularity as a tuning field** (`pyramid_storage_block_cells`) — a
   persisted-key decision either way.
7. **Confirm the `CellIndex` linearisation as a deliberate door.** `(c*62 + b)*62 + a`, `a` fastest, is
   **already frozen** into the WAL record, the pyramid entry and the mask's bit order — but it is stated
   only in a doc comment. The chosen branch is also the better one (~2× fewer runs than radial-fastest on
   terrain). The risk is that the door was spent *silently*.

*Cost of deferring:* each is a migration over every saved world, and item 3 is also an unbounded-storage
outage waiting on a live planet.
*Source:* storage D-2, D-4, D-5, D-8, D-9, D-10, D-11, D-16; provenance §5.4, §8 row 13.

---

**O7 ⛔ before generator code · Where does the terrain noise come from, and does the sampler take an octave count?**

A library adoption, therefore the owner's by standing rule. The library the roadmap names seeds its
permutation table through a loosely-pinned RNG crate, so a patch bump regenerates every planet in the game
with **no compile error and no test failure**.
*Options:* (A) vendor ~400 lines into the core crate inside the determinism guard — honest cost, corrected:
10–20 data-dependent branches, each needing a directed test under HR5; it is **not** branchless. (B) use
the declared pinned library with our own table. (C) adopt a different library.
*Recommendation:* **(A)**. **And not optional, whichever way it goes: the sampler must take an explicit
OCTAVE COUNT as a parameter from the first commit** — tier L is literally "the same call with L fewer
octaves", and a fixed-octave sampler makes the ladder a second code path, which HR3 forbids.
*Also settled here because it freezes with the same digest:* the coarse-lattice cave step (4 vs 8, worth
0.205 ms/chunk if the reference CPU comes back badly).
*Blocks:* the generator, and therefore P4 entirely.
*Cost of deferring:* changing the noise after players build means regenerating every world underneath
their existing edits. **Note this contradicts `roadmap.json` P4's "pinned `noise`" deliverable — see §7.**
*Source:* register row 8; door 16.

---

**O8 ⛔ before the first world is written · Which materials are concealed, and how is placement conditioned?**

Three coupled sub-questions with one deadline.
- **Which materials.** Recommend the small strategic/exotic tier the price signal depends on. Iron,
  copper, coal and all building stock stay PUBLIC and visible. **Decide it on the gameplay argument** — the
  economic argument is refuted (see §1).
- **The placement rule, which matters more than the key.** Concealed placement should own the COUNT,
  MATERIAL and RICHNESS and read the public world **only through the coarse integer depth band**. Realism
  is bought back through a **public indicator material** with a declared enrichment factor — a hint you
  learn to read, at zero cost. Today's exposure is a benign 2.1–7.8×; a *believable* geology rule leaves
  only 7.6 bits and a **1,192× prediction advantage**, and the next revision is exactly where believability
  rules get added — which is why this must be taken before the ore table freezes rather than audited after.
- **`concealed_key_id: [u8; 8]` on `RealmDescriptor`**, refused loudly on open, master key in the
  deployment secret store and never in a realm database file. Eight bytes in a record that already exists
  is free now; without it a rolling deploy mid-rotation forks a planet's ore **invisibly to every gate**.

*Cost of deferring:* moving a material from public to concealed after players mine moves ore out from
under their tunnels — a galaxy-wide regeneration under existing builds. Invisible when wrong, and only
discovered when players publish a prediction tool.
*Source:* concealed C-3, C-5, C-6, C-8.

---

**O9 ⛔ same deadline as O8 · Does the survey instrument and the assay loop ship with concealment?**

**Ship both or ship neither.** Concealment without discovery removes an activity and returns nothing. The
cost is smaller than it looks and that is the argument for scheduling it: the **assay needs no new
machinery at all** (the item grant already rides the block-edit acknowledgement path), and only the
**survey** is new — and it rides the block-edit carrier already being built at S0.4, whose second consumer
is already reserved.
*So the honest choice is binary:* add a small P6 survey slice alongside the edit pipeline, **or** take
option (A) all-public for everything now and keep the door open by taking only O8's three free
reservations, which cost nothing.
*Source:* concealed C-9.

---

**O10 ⛔ before the first planet is saved · ⚠NEW · Does the generator emit a non-cube surface skin?**

**The finding:** the metre grid plus the cube-only generator make a hillside a staircase of 1.000 m
risers, which a person clears by design and **a wheel cannot**. Paired with: what locomotion class is the
first vehicle?
*Options:* (a) **HOVER** — needs nothing new; four casts under a hull smooth a 1 m staircase to ~0.25 m;
the only option with zero terrain consequences. (b) **TRACKED**. (c) **WHEELED AT SPEED** — requires the
surface-skin field from the FIRST generated world.
*Recommendation:* (a) first, (b) second, and **RESERVE for (c) by shipping the generated-surface-skin shape
as a generator-config field defaulting to `Cube`, named in the P4 terrain-golden digest.**
*Cost of deferring:* turning the skin on later changes generated terrain **under existing player builds** —
the same one-way door as the noise-lattice step. One field and one line in a fixture today; a world-format
epoch later.
*Source:* `collidable_decoration.md` UD-16 and reservation V-R6.

---

**O11 ⛔ before P4's determinism digest is pinned · Is the climate baseline seasonal?**

*Options:* (a) time-invariant; (b) seasonal via an **integer season index**, which **widens a committed
Definition of Done** from `f(seed)` to `f(seed, season_index)`; (c) continuous time — refused.
*Recommendation:* **(b)**, and it must not be taken silently: it is a determinism-contract door that
changes a stated DoD in `roadmap.json`. It is also what the dormant-world pillar and the economy's
forestry and agriculture professions both need.
*Cost of deferring:* one extra fixture parameter now; a re-argued definition of done later.
*Source:* `collidable_decoration.md` UD-9; `roadmap.json` P4 `definition_of_done`.

---

### Band B — before the wire arm exists

Answering late means a protocol-minor bump plus a client migration, and in two cases a re-serialisation of
persisted data. **postcard is positional**, so a field added later is not additive.

---

**O12 ⛔ FIRST SLICE (S0.4), not P9 · The scoped channel key — RULED, and it has no door row and no slice**

R19 already decided this: `ChannelKey = H(scope ‖ name)` over `Realm` / `Construct` / `Net` / `Protocol`.
**What is owed is scheduling, and the recorded deadline is wrong by five phase groups.** Door 34 says "P9
slice 1"; the slice plan lands the content-hashed channel key with its loud collision check in **S0.4, in
Phase 0, before any P4 feature code**. Verified: there is **no `ChannelKey` anywhere in the workspace
today**, so the fix is free right now.
*Cost of deferring:* every fixture, golden pin and persisted binding minted between Phase 0 and P9
re-resolves to a different channel when the scope is added — **and R18 makes it worse than the design
assumed**, because blueprints carry all wiring and are re-sold, so a late key change breaks the blueprint
*library*, which is data that has been copied.
*Action:* add it as a door, put it in S0.4's first commit.
*Source:* R19(1); door 34 against slice S0.4.

---

**O13 ⛔ S0.4, before the block-edit arm exists · ⚠NEW · The requested-tier byte and the tier descriptor**

**There is no terrain-tier field anywhere on the wire today.** `intershard.rs:659`'s `coarsen_level` is a
**pose**-precision hop counter on `OccupantInterest` whose own doc comment says so, and line 303 reserves
the WHAT lane as a *concept*, not a field. Addendum 2 cites it wrongly, and so does a committed repo doc
(see §7).
*Recommendation:* plant `tier: u8` on the request **and** a tier descriptor on the delta payload,
**together**, because postcard is positional. Two bytes in an arm nobody has written, plus the four
structural conformance tests.
*Binding rider:* **three ladders exist and none derives from or converts into another**, each carrying a
doc comment saying so — the chunk key's `tier` (chunk-local CONTENT), `coarsen_level` (a POSE-precision hop
counter), and `tier_floor` (client-declared, server-clamped, **non-authoritative**, egress selection only:
no fence, arbitration outcome, admission decision, scheduling deadline, persisted value or simulation
state may read it, directly or transitively). `observer_tier` and `min_demanded_tier` are **struck** —
nothing is planted and nothing is owed, because the spin-up lead they existed for is now
`lead_time_s = d_demand(r) / max(v_closing(o,r), v_closing_floor)`, both terms server-known since the A5
flip, so the lead is continuous rather than quantised into rungs.
*Blocks:* coarse-first loading — without it a client cannot ask for tier L without first pulling every
tier-0 delta underneath it.
*Cost of deferring:* free today; a protocol migration plus a pyramid re-serialisation afterwards.
*Source:* `[USER DECISION 4-F]`; door 45; D-WLD-9; §7.18 item 17.

---

**O14 ⛔ S0.4 · The four saga step-id constants, allocated once, in one file**

**Verified in code:** `crates/wire/src/intershard.rs` allocates step ids 0–6 (route swap) and 7–18
(`FLUSH_SOURCE` 7 … `RE_SOLICIT` 18). 19 upward is free. **§3.11 reserves 19 = block-store flush and
20 = block-edit forward; §7.18 reserves 20–23 for durable signal-config applies. Both claim 20 and they
disagree about 19.**
*The consequence, in §7.18's own words:* "a step-id collision aliases two different operations in an
`applied_steps` journal. **Silent, durable, unrecoverable.**"
*Recommendation:* §3.11 owns storage and names two P6 steps concretely — take **19 = block-store flush,
20 = block-edit forward**, and move the signal block to **21–24**. Whichever way it goes, all four consts
land in ONE file in S0.4, which is already the slice that reserves them.
*This is the highest cost-to-fix ratio row on the whole board:* four `const` lines against unrecoverable
durable corruption.
*Source:* §3.11 against §7.18 item 2; verified `intershard.rs:59–109`.

---

**O15 ⛔ S0.4 · The signal message family's shape — and the closed set has exactly zero headroom**

**Verified by enumerating the enum: `InterShardFlow` has exactly 27 arms today.** `PLAN.md` HR1 reserves
exactly three more names — `BlockEdit`, `Coupling`, `Signal`. **27 + 3 = 30, which is precisely the number
door 38 names as the point at which "the closed set passes thirty entries and the structural review that
makes the sealed-world rule enforceable stops being reviewable." There is no margin left at all.**
*Recommendation, and it must be stated in S0.4's prose rather than discovered:* everything else rides
**inside** an existing arm as an internal discriminator, never as a new arm — `BlockEdit` carries
`{place, break, fetch-delta, fetch-pyramid}`; the relay header rides `Signal`; any survey/assay carrier
rides `BlockEdit`. State the ceiling arithmetic explicitly, or the first implementer adds a 31st arm and
the sealed-shard review argument quietly stops holding.
*Four things the shape must carry that §8.3's S0.4 text predates:* `BundleAuth{mac, key_epoch, grant}`;
`Option<RelayHeader>{origin ChannelSeq, hops_remaining, expires_at, class}` (R17's hop count, arrival
deadline and dedup identity, which must be in the first slice or they are a migration);
`ChannelDecl.accepts_relay_origin = false` + `max_relay_hops = 0` — **the default IS the security
property**; and `ChannelDecl.group: ChannelGroupId(u16)`.
*Cost of deferring:* door 38 already says P4, and S0.4 is earlier than P4, so it is already late.
*Source:* door 38; PLAN.md HR1; signal Tier-1 rows 3, 4, 5, 8, 9; verified arm count.

---

**O16 ⛔ before the wire ABI settles further · The render origin becomes a rigid POSE**

*Question:* does the server-told render origin stay a translation, or become a full server-authored rigid
pose (translation + rotation)?
*Recommendation:* **take it now.** It is what makes the atmosphere correct on a sphere; the alternative is
a permanent engine fork. The character controller wants it independently.
*Status — this door is already half through:* verified in code, `pin` / `pin_abs` / `anchor_epoch` shipped
at **PROTO_MINOR 7** on `RealmRegistry` and `RealmSceneDelta`, with consumers across `wire/channels.rs`,
`client/view.rs`, `client/net.rs`, `client/render_snapshot.rs`, `devproto/state.rs`,
`client-harness/capture.rs`, `harness/client.rs`, `connection-plane/gateway.rs` and two integration smoke
tests. **Every additional consumer raises the price**, and it gets more expensive every week the
absolute-position path runs in production.
*It is not in §8.2's 48-door table. It should be.*
*Source:* `stunning_look_plan.md` SL-2; verified `crates/wire/src/version.rs:54`.

---

**O17 ⛔ before the bulk chunk-delta lane is routable · Widen the chunk-state wire entry now**

*Question:* does the per-chunk state entry ship as `(local_index u18, stage u3)` = 21 bits, or as
`(local_index u18, kind u2, value u4)` = exactly 24 bits?
*Recommendation:* **the wider form.** It is *exactly the same width* (3 bytes), the same message, the same
batching — and it decides whether vehicle marks, snow depth and crack stages share one lane or need a
second one.
*Cost of deferring:* a protocol bump plus a client migration — precisely the retrofit this design refuses
to pay everywhere else.
*Source:* `collidable_decoration.md` UD-11.

---

**O18 ⛔ before the signal wire shape freezes · Does distance ever delay a signal, and how fast is the relay plane?**

*Options (delay):* (A) no delay anywhere — remote control works at any range and the sense of distance is
lost; (B) delay everywhere by light speed — at one AU that is eight minutes, so every cross-system
interaction becomes mail and real-time fleet play dies; (C) a zero-delay bubble, growing delay beyond it,
outright refusal past one light-hour, relays exempt with their own much shorter latency.
*Recommendation:* **(C) with the bubble at TWO light-seconds, not one** — one light-second does not even
reach a planet's own moon, so the canonical example the feature exists for would fall outside it. Under
(C) a continuous control stream is never *delayed*, only **refused**, which is what lets the refusal be a
legible game rule rather than a silent degradation. Note (A) would also delete both the trilateration
position leak and the State-refusal rule that keeps the relay plane honest.
*Options (speed):* `propagation_c_multiple` for the trunk relay class — **M = 10⁸** (1 ly in 0.32 s,
1,000 ly in 5 min 16 s) / M = 10⁷ (1,000 ly = 53 min, which converts the plane into an inbox) / M = 1
(interstellar comms literally impossible). Recommend 10⁸. The field is config so the *shape* is not a
door, but the **feel** is: once fleets and markets organise around a latency, changing M by an order of
magnitude is a live-service incident.
*Cost of deferring:* players build around whichever answer ships; removing zero-delay remote control
afterwards is a community incident and adding delay later invalidates every station and fleet design.
*Source:* register row 18; signal D-22; door 33.

---

### Band C — before art is authored

Answering late costs **months and money**, not a migration. Art lead time is the longest pole in the whole
set: a commissioned original prop set is months and roughly €15–40k for 60–80 coherent props.

---

**O19 ⛔ before the first material is authored · The texture tile, its resolution, and the roughness bake**

Three documents ask this about one artefact; it is one door.
*Options:* 4 cells at 512² (128 texels/cell, 128 MiB for 128 materials) / **4 cells at 1024²** (256
texels/cell, 512 MiB) / 8 cells at 2048² (256 texels/cell, 2 GiB) / 4 cells at 2048² (512 texels/cell,
not shippable at scale). Separately: ship authored scan-based data, or synthesise every texel on the
client from the seed. Separately: a texel-density band of **±1 stop over [128, 512] texels/m** — ±½ stop
forces a second 2048² array trio that does not fit.
*Recommendation:* **4 cells at 1024², a hard cap of 128 materials in the shipping set, ship authored data,
±1 stop.** Two-cell tiles read as almost per-block and lose the multi-block effect entirely. Synthesising
moves real GPU cost to the moment a world spins up, which is exactly where the seamless rule forbids a
stutter.
*Not optional and on the same schedule:* **the roughness mip chain must be generated variance-aware from
the pre-compression three-channel normal map**, not box-filtered from the roughness map. Mipmapping a
normal map preserves the mean and destroys the variance, and the lost variance must reappear as increased
roughness or the specular lobe stays too tight and the surface scintillates under a moving sun. **The
classical runtime fix is mathematically impossible for us** — §5.3.4 locks two-channel BC5 normals, so
every fetched normal is unit length by construction and the averaged length the technique needs is
unrecoverable. So it must be baked, and **the bake tool does not exist** (D-RND-13).
*Cost of deferring:* re-authoring hundreds of materials at a different period is a re-do, not a refactor —
and every material authored before the mip rule exists must be re-baked.
*Source:* register row 6 / `[5-C]`; SL-5; doors 18, 46; D-RND-13.

---

**O20 ⛔ same sitting as O19 · The display transform, the exposure policy, and the look statement**

*Recommendation:* a neutral, hue-preserving tonemapper plus **MANUAL exposure** from a server-authored
per-realm illuminance, with auto-exposure only as an accessibility option the harness forces off — because
automatic exposure makes **stars vanish on warp arrival**, a direct violation of the seamless hard rule,
and makes every pinned image gate history-dependent. And the look statement in one sentence:
**physically-lit naturalism — "physical, blocky, old."**
*Blocks:* what every material's albedo should be. It is the root node; every other art decision is
downstream.
*Cost of deferring:* choosing after 128 materials exist is a re-paint.
*Source:* SL-3, SL-4.

---

**O21 ⛔ before terrain art · The crumbling-edge amplitude, and whether 0.20 m is a ceiling or a default**

*Options:* amplitude none / 10 cm inward-only / **20 cm inward-only** / 25 cm with outward permitted /
33 cm. Who wobbles: everything, or **natural material only with every constructed material exactly zero**.
Plus: is `A_max` **asserted at build time** (so the rock class gets its outcrop look from the composite
lane) or may the rock row raise it?
*Recommendation:* 20 cm inward-only; natural 8–20 cm, constructed exactly zero; **`A_max` ASSERTED at
0.20**. Letting the rock row raise it taxes grass, dirt, sand and everything else at the coarse rungs
where the ladder's whole saving lives, and silently reopens a decision already answered. Inward-only is
the difference between "the world is solid and I sometimes hit slightly more than I see" and "the world
lies to me".
*Tier rule, itself a decision:* the amplitude is **in metres, is not scaled by tier, and is exactly ZERO
at tier ≥ 2** — a 20 cm feature is sub-pixel past ~300 m, and dropping it removes the merge barrier from
every coarse chunk. Measured: 2.2× merging retained on a small hand-built box, 15.7× on a full-size
surface.
*Cost of deferring:* close to a one-way door — all terrain art is authored against whatever amplitude is
chosen, and the merge win is otherwise lost a rung at a time with nobody able to say which change did it.
*Source:* register row 11 / `[5-E]`; UD-5; finding A7; door 21.

---

**O22 · Where the prop and composite art comes from, and the licence confirmation**

Asked three times across the set; it is one decision.
*Options:* (A) **buy** stylised nature packs as greybox now — ~€46–200, days of pipeline, zero art lead
time, with the risk that greybox becomes ship; (B) **commission** an original set — months, ~€15–40k for
60–80 coherent props, full control; (C) **free** public-domain scan-based props — zero cost and zero
licence risk, small incoherent sets, no nature biome set; (D) fully procedural from the seed.
*Recommendation:* **trees — bought greybox with a DATED written re-authoring commitment; rocks —
permissively-licensed scan-based** (scan-based rocks blend into a physically-lit voxel world where
stylised ones do not); **leaf clusters — procedural on necessity**, since a cluster's shape depends on its
neighbours and a fixed bought mesh structurally cannot serve it; **schedule (B) to land before any public
build.**
*Two live licence clauses:* source files must never leave the team, and the assets must never be fed to
any generative tool. **A baked impostor atlas is a derived work under the same clauses.**
*A free quality win whichever way it goes:* **hash-derived per-instance shape variation** (scale, lean,
branch phase, hue jitter) from the cell key rather than randomised — deterministic, identical on every
client, a few bytes per instance, no new assets, no new material slots.
*Paired:* do decorative props ever become **carryable items**? Recommend no for the first release; if yes,
obtain written vendor confirmation **before** any of their meshes becomes the placeable one.
*Cost of deferring:* one-way in one direction only — shipping recognisable bought art establishes an
identity that is expensive to change. Buying the packs and never shipping them costs ~€200.
*Source:* register rows 25 and 33; SL-14; door 41.

---

**O23 · Do distant props get baked impostors? — and two documents give different answers and different numbers**

**Two live contradictions, and they must be adjudicated before either is scheduled.**
*Contradiction 1 — the verdict.* Register row 39 and `[6-8]` both **recommend** baked hemi-octahedral
impostors for a curated archetype set; `stunning_look_plan.md` §9 explicitly lists "impostors for
composites" under **what we are not building** — 7.97 ms at the fill denominator over a forest, 11.5×
overdraw, and "the rung underneath is real cubes, which is strictly better and already paid for". Under
O26's ruling a tree **is** a composite, so the very case row 39 exists for (a 12 m tree vanishing at the
cull radius, "the most jarring pop available") is the case the look plan refuses. What survives is
impostors for **non-composite props only**, which is a much smaller prize.
*Contradiction 2 — the numbers.* Row 39 says 1024² atlases, ≤32 archetypes, 4.67 MiB each, **149 MiB**
total, extending props ~3× past a stated **80 m** radius. `[6-8]` option B says one 64-layer array at
1024×512 per layer, 32 archetypes at 128², **151.5 MiB**, and that props today end at **~33 m for a bush,
114 m for a boulder** — so the 80 m baseline was *further out* than reality. Two configurations, two
baselines, two totals.
*Options after adjudication:* **(A) nothing** — props stop at their radius and the far field is terrain
only; defensible if the archetype set stays small and the prop radius is pushed instead, and it **deletes
a tool**. **(B) impostors for the non-composite classes only.** **(C) cluster impostors / billboard
clouds** — rejected in both documents.
*The sizing that decides resolution:* 2048² is 18.67 MiB/archetype → **597 MiB** for 32, more than the
entire 512 MiB material set and enough alone to breach the 3.0 GiB cap on top of the 1.41 GiB on-foot
baseline. 1024² is 4.67 MiB → 149 MiB, which fits. A 128²-texel frame stays convincing to ~128 screen
pixels, and a 20 m tree subtending 128 px at 1920 px / 70° is at ~250 m.
*Cost of deferring:* moderate and one-directional — **the bake consumes the archetype set**, so deciding
after it is authored means re-authoring for bake-friendliness (clean silhouettes, no interpenetrating
geometry — door 48).
*Source:* register row 39 / `[8-C]`; `[6-8]`; `stunning_look_plan.md` §9; D-DEC-7.

---

**O24 ⛔ the face-hiding and AO halves are due at P4.2, not "just before implementation" · R14's deferred collider pass**

The owner deferred four questions to just before implementation. **Two of the four are not collision
questions at all**, and both are due earlier:
- **whether a pass-through cell hides faces behind it** is the mesher's face-mask culling rule, which lands
  at **P4.2**, and whose *pinned quad list* becomes the declared regression baseline for P4.4 and P4.5;
- **whether it counts as solid for ambient occlusion** is P6.5's third occupancy mask, whose threshold
  §8.6 already flags as stated in two incompatible units.

A pass-through cell class (R13 bushes, R14 canopy) arriving after P4.2 changes occupancy semantics
underneath three pinned gates and one 100%-branch-covered crate.
*Recommendation:* **answer the face-hiding and AO-solidity halves before P4.2**; the collider-shape and
fitted-rock halves can genuinely wait for P5.2. Splitting the deferral costs two questions now instead of
four later.
*Source:* R14's deferred pass against slices P4.2 and P6.5.

---

**O25 · Pin the composite tree's cell count, and re-issue the four numbers that depend on it**

*The finding:* the tree has **two published sizes 19.4× apart** (162 vs 3,146 cells), and the composite
figure **fails its own content gate by 3.07×** — which means as published a tree **cannot fall**.
*Recommendation:* **≈998 cells** — the only value consistent with the 121 m² footprint, the 1,024-cell
gate, the 8,192 B blob and the solid-canopy rule simultaneously. Re-derive from it: the collider budget,
the footprint write, the anti-conjuration bound, the felling table and the deforestation figure.
*Blocks:* everything else in the composite lane. **And the economy has already been handed a yield figure
3.1× too high** — whoever owns the economy needs telling, not just the block design.
*One-way in art:* every archetype is modelled to whichever answer wins.
*Source:* `collidable_decoration.md` UD-3, finding A1.

---

**O26 · R4 itself — the authoritative multi-block composite subsystem is undesigned**

The one owner ruling marked OPEN. Placing several tree blocks yields a real tree from an art pack; more
blocks yield a bigger one; rock blocks yield a non-blocky rock; **all of it must collide.** It crosses
§6's hard promotion boundary and **must not be built by relaxing it**.
*What the owed pass must cover, at minimum:* the pattern-recognition rule; where the collider comes from
(the server has no meshes); what happens when the pattern is broken; the per-realm budget; and the
art-direction question of stylised low-poly assets inside a physically-lit voxel world.
*The proposed rendering half, which discharges R4's visual half and is itself awaiting you:* **composites
are RECOGNISED, never overlaid.** Every composite cell stays a real cell — stored, collidable,
harvestable, identical for every player — and a recognised composite's cells are **excluded from the chunk
mesh's emit step at tier 0 only**. The trunk you see is the mesh because it is the only thing drawn; the
cubes remain fully real for collision, mass, raycast and persistence. This deletes the overlay, the derived
collider and the second occupancy authority in one move, and it is what makes the decoration layer emit no
leaf cards on cells a composite owns — one canopy, ever.
*The footprint rule that rides with it:* **(a) Conservative everywhere**, mesh strictly inside its cells,
≤ 0.20 m downward only — recommended, because it satisfies the owner's own sentence, makes "render ⊆
collide" hold with **no art discipline**, and resolves two settled documents that contradict each other.
Replace the 92% coverage ratio with a per-cell distance bound in the same commit (the ratio fails every
rounded archetype at 84.5%). `crown_tolerance_m` and `canopy_clear_m` become per-archetype bake columns —
a forest's mean free path at canopy height is **6.9 m for a 2 m rover**, a contact every 0.23 s at 30 m/s,
against 33 m for trunks alone. And the far cube forms must be baked **from the coarsener**, not from the
art mesh, so the two agree by construction.
*Blocks:* O23, O25, and the tree, rock and vehicle-clearance numbers. **The emit-exclusion rule has a P4.2
deadline while its subsystem has no design** — see §4, where P4.2 lands it as a no-op hook.
*Cost of deferring:* re-modelling every archetype at 2.5–5.2 h each; retrofitting an emit rule into a
100%-covered mesher and re-baking every recipe.
*Source:* R4 and the note beneath it; SL-6; UD-4; findings A10, A12, A14.

---

### Band D — before the subsystem is built

Answering late costs a rework, not a migration.

---

**O27 ⛔ decide the DEADLINE, not the format · What does one mesh vertex look like, and when does the packed path land?**

**The register's "de-escalated" status is stale.** Addendum 1 promoted it to blocking (Earth-sized on foot
= 53,110 chunks, 2.1 GB packed against 25.5 GB at today's 24-byte format); Addendum 2 de-escalated it
(~4,000 chunks, ~160 MB); §8.1 ranks it 11. **Three later documents re-escalate it with harder numbers,
and the register reflects none of them:**
- a **city's tier-0 disc alone is 355 MB** at today's vertex, against the **273 MB budgeted for an entire
  100 km view** — 32 MB packed;
- video memory has **no usable margin** without the packed format (0.19 GiB), upload is already breached
  under motion, and the composite instance path wants a packed record anyway — so pull it from P8 to **P6**;
- if the target speed band is 450+ m/s the packed path becomes a **P4 blocker**;
- row 21's own caveat already concedes the 100 km flight view is **1.3–3.0 GB at 16 bytes against ~270 MB
  packed**, because per-chunk mesh cost is close to **tier-invariant** (62² = 3,844 top-surface cells at
  every tier; dropping three octaves removes ~37% of the slope and only trims side faces).

*Adjudication:* the de-escalation is **correct for the narrow claim it makes** (whether a planet can exist
on foot) and **wrong as a ranking**. This is not a data migration — it is a mesh-encoder and upload-path
change — so it is a **scheduling** decision, and it carries three mutually exclusive deadlines today:
"while there is still one producer" (S0.3), "pull to P6", and "measured, at P11". By P6 there are four
producers plus their golden fixtures, which is exactly the state the one-producer argument exists to
avoid.
*Options:* **(B1)** 16 bytes/vertex through the standard engine pipeline now with the packed 8-bytes-per-
quad path declared the endgame; **(B2)** the hand-written path immediately — an 18× win that costs the
entire engine-provided GPU-driven path (bindless, multi-draw, two-phase occlusion culling, depth prepass,
shadows, motion vectors) on a renderer that migrates roughly every four months; **(B3)** stay on 24 bytes.
*Recommendation:* **B1 ships first, and the packed path is scheduled honestly at P6 as a four-producer
migration — delete the "one producer" justification or take B2 at S0.3. Do not keep both sentences.**
*Source:* register row 21; storage D-14; SL-8; D-RND-3.

---

**O28 ⛔ P5.1, before the physics phase commits · The physics and collision spike, widened**

Five documents converge on one spike. Neither library is in `Cargo.lock` today — and a roadmap line naming
the physics engine is **not** the owner's consent for the *collision* library's new sparse voxel shape,
which is what is actually escalated.
*Options:* **(B1)** adopt both and use the voxel shape on FLAT realms only — ships and stations — with
hand-built convex composites everywhere else (~1 byte/block with built-in removal of the seam-snag problem,
against ~120 bytes/block hand-rolled); **(B2)** adopt both, our own composite everywhere, one uniform lane;
**(B3)** the engine-native alternative, same collision library underneath; **(B4)** hand-roll.
*Correction that must not be lost:* the voxel shape is **structurally incapable of representing a planet** —
a planet's cells are sheared and tapering and the shape assumes a uniform square lattice. This is a genuine
planet-versus-ship divergence and must not be sold as a run-anywhere win. And one case is **unverified by
any shipped implementation anywhere**: a cube next to a slope is not recognised as an internal face unless
a phantom block is inserted, which then collides where nothing is.
*Three arguments the original options table did not have, and the spike's scope must be re-cut before it
runs:*
- **the scripted topple REMOVES an argument** for the voxel shape — a falling group is a server-authored
  rotation with a closed-form final pose and an integer settle, **no collider, no transfer envelope**, live
  and dormant paths writing identical bytes — while a rigid-body topple ADDS one (up to 32,768 voxel
  colliders at the 64-group cap, a non-reproducible landing, a second world representation);
- **per-cell FRICTION** would need either one voxel collider per friction class (~8, sparse, plus 8 BVH
  roots) or the greedy-box fallback on all four lanes, which carries per-box material naturally — a new
  argument that was not in the table when it was written;
- **the HULL collider is the biggest unowned item in the physics half** and blocks landing, docking,
  boarding-from-outside and all of P11 combat. Coarse voxel occupancy at a declared rung (~1.9 kB for a
  100 m ship at a 4 m rung) hands your ship's shape **including internal voids** to whoever hosts you — a
  free hull scan at a player-owned station. Reusing the existing ghost-collider lane avoids that if it can
  carry an owned body; a convex hull is wrong for any ship with a hole in it.
*Recommendation:* **(B1) with (B2) pre-designed as the fallback**, scripted topple for v1, and the spike
widened to cover friction and the hull before it runs.
*Ledgered consequence:* **collision is TIER 0 ALWAYS, as a refusal in the collider builder**, not a policy
in a document.
*Source:* register row 16; §8.6 item 2; R6-4; UD-15; signal D-16.

---

**O29 ⛔ with the geometry-seam slice · What does the run-anywhere gate mean for geometry code?**

A cube-face-seam test has no meaning on a ship grid, so demanding equal results on both would either test
nothing or test the wrong thing.
*Options:* (A) **split the gate** — bind the equal-results rule to everything *above* the geometry seam and
give the seam itself a stronger **exhaustive-enumeration** gate; (B) hold the rule literally and accept
that the seam is untested by it. Second half: when a ship rests on terrain, does the planet own the contact
or does the ship?
*Recommendation:* **(A), with the PLANET owning the contact.** The seam table does not depend on planet
size, so exhaustive enumeration at the smallest configuration is a **complete proof** and fits in a
unit-test budget; planet-owns-contact matches the standing law that the containing realm authors its
children's positions.
*Second exhaustive obligation, cheap:* the nine operations take the **tier** as a parameter and must be
enumerated at **more than one tier** — a mapping correct at tier 0 and wrong at tier 3 is a class of bug
the existing gate would not see, and it is a loop over the same smallest configuration.
*Cost of deferring:* an unresolved gate quietly stops being a gate.
*Source:* register row 12 / `[3-B]`.

---

**O30 · The in-realm authorisation model — nine coupled rows, one sitting**

R15 closed the *outside*; this is the *inside*, and the design currently asserts a write ACL three times
and specifies it zero times. **The warning worth keeping in front of you: "the most dangerous outcome of
this review would be concluding it is fixed because the design contains the words grant and ACL."**
- **Default write scope on a new channel** — `write=Owner, read=Owner, plane=Local`, `Public` an explicit
  opt-in, versus permissive-then-tighten, **which ships the owner's attack as a default**. A saved-world
  door in both directions.
- **Grant scope shape** — enumerated keys only (brittle) / hash per path segment (a hard door on the
  channel key) / **`ChannelGroupId(u16)` plus optional explicit keys** (recommended). Must be settled
  before ANY grant row is written, or every persisted grant and declaration is re-authored.
- **Grant conditions** — `WhileDocked` / `WhileSeated` / `WhileInRealm` **plus a mandatory hard
  `not_after`**, versus a time-only expiry. This decides whether undocking automatically ends a station's
  authority over your ship, or you must remember to revoke — the class of thing a player gets wrong once
  and is furious about.
- **Priority is GRANTED, never declared** — `min(declared, owner's ceiling)`, non-owner default band 0.
  Under the alternative the shipped catalogue contains a block that **out-ranks your seat** with no check.
- **Revocation has no grace period** — revoking an unseated flight grant cuts thrust to the safe state,
  with a UI warning before confirming. The temptation to soften will recur; **a revocation that can be
  delayed is not a revocation.**
- **Ownership transfer revokes every grant and role atomically** under the new owner's fence, riding the
  transfer saga — this decides whether a stolen ship is CONTENT or a BRICK. Under the alternative the
  previous owner keeps flying his stolen ship remotely.
- **Docking is a BRIDGE, not a MERGE** — a revocable, grant-scoped bridge of one named channel group,
  versus opening the resource and signal buses, which hands a docked stranger write access to your bus.
- **The realm ACL's default role for an unlisted principal** — `Visitor`: no interact, no build, no bind.
  A gameplay-feel decision as much as a security one.
- **The grant's shape, re-taken** — plain HMAC grant rows with `parent: Option<GrantId>` and a subset rule
  (recommended), versus an attenuable-token library. **The tree already gives delegation and subtree
  revocation with no new dependency**; the library's only remaining case is offline attenuation across
  trust domains, which no scenario needs. Four persisted tables: realm owner, principals, acl, grants with
  `key_epoch` and the already-reserved `token: Option<TlvBlob>`.

*Why bind-time and not per-message, quantified:* the cheapest conceivable per-signal check **triples the
subsystem's headline number** (3.0% of a tick against a 1% budget); a MAC costs **more than one whole
tick**; Ed25519 costs **18.86 seconds per 50 ms tick**. Bind-time costs exactly zero on the hot path.
*Three live specification defects that must be fixed in the same pass:* **Pass B must NEVER write
`front[c]`** — otherwise engines cut 500 ms after any autopilot disengages, on a channel the pilot is
actively writing; **authorisation is durable and spin-up RE-AUTHORISES** (the ~2 s reap otherwise undoes
every revocation); and **`on_block_removed` becomes a REGISTRY** over config, port, grants, ACL, key and
relay queue (a destroyed Access Reader leaving a live grant is a security failure that presents as a
storage leak). Plus `max_replay_entries_per_peer = 64` with per-origin queues, because the anti-replay
control as specified is a **164-second denial of service on legitimate traffic**.
*Source:* signal D-1, D-4…D-8, D-10, D-11, D-14 (D-14 supersedes register row 34).

---

**O31 ⛔ before P4.10's governor is written · The target flight speed band, and the terrain radius floor**

**Door 42 conflates a gameplay INPUT with an engineering OUTPUT, and two registers own it.** §8.2 and §8.4
make maximum sustainable flight speed an *output* of the mesher-throughput and residency gates at
P4.2/P4.10; the latency analysis independently derives **240 m/s from the terrain UPLOAD budget as the
tightest of four ceilings** and owns it as its own row. They agree in substance — upload is the binding
term either way, and §8.4's residency row explicitly measures "generate + mesh + **upload**" — but a
one-way door owned in two places gets shut twice at different values.
*Recommendation:* record it **once**, split into **"target band (gameplay input, owner, due before
P4.10)"** and **"achieved ceiling (engineering output, due at P4.2/P4.10)"**, carrying 240 m/s as the
current V1 estimate.
*The input you must supply:* (a) ≤240 m/s — the 16-byte vertex is fine; (b) 450+ m/s — the packed format
becomes a P4 blocker (O27).
*And one binding constraint on the governor that is currently owned by nobody:* **minimum drawn terrain
radius ≥ v_max × 4 s.** This is the one rendering-budget decision that can make the no-prediction law
**unsafe** — draw less than ~1 km of terrain and the human control loop is visibility-starved regardless
of latency. It is a hard floor alongside `tier0_min_radius`, and without it the governor is written and
then decides the cheapest thing to shed is the disc the pilot needs.
*Source:* door 42; §8.4; `high_speed_flight_latency.md` §3.5, D-5, D-10.

---

**O32 · The residency governor's fields, the fade direction, and the detail knob as an ANGLE**

*Options:* **(H1) fixed on-screen target, no governor** — in a dense built region it blows the resident-byte
cap and the client hitches or evicts something visible; **this is the failure the title the owner cited
actually shipped**, where raising quality settings paradoxically *worsens* pop-in. **(H2, recommended)
fixed target plus a smoothed, hysteretic governor** that raises the target toward a ceiling when resident
bytes exceed the cap. Tens of lines. **It must be smoothed or it becomes a pop source.** **(H3)** H2 plus a
user-facing terrain-detail slider, at the first settings pass.
*Fields and defaults, all on the render tuning struct:* `px_per_cell_target` 2.0, `px_per_cell_ceiling`
4.0, `resident_chunk_byte_cap`, `governor_hysteresis` 0.15, `crossfade_band_fraction` **0.15 — which is
not a taste value** (see §1: 0.20 fails the residency gate).
*The fade-direction rule, which is not tuning and must be written into the slice:* **the COARSE rung fades
in early over the fine rung's outer band; the fine rung dies at its own range — never the reverse.** At
f = 0.2 that is ~45 extra columns per boundary (+10% across seven boundaries) against ~182 (+40%) the
other way; at f = 0.15 it is ~+7%.
*Three constraints:* the governor clamps the shadow distance; it may never shrink tier 0 below **maximum
tool reach + one chunk edge**; and it may never change collision, physics or any gameplay-visible fact.
*A superseding amendment you should take with it:* **the knob must be expressed as an ANGLE with the
reference tier as the one denominator**, and the rendering header's 1080p sentence restated as a derived
figure — otherwise every published residency, terrain-memory and geometry number in the design is
resolution-dependent and means nothing. **That is a measurement door: taking it re-baselines every gate.**
*Source:* register row 38 / `[8-B]` (absorbs 1-A, 5-H); SL-1.

---

**O33 · How far do player-built structures cast real shadows?**

*The constraint, verified in the vendored crate:* `MAX_CASCADES_PER_LIGHT = 4` is a **compile-time
constant** backing a fixed four-element array in both the Rust light uniform and the WGSL view struct;
raising it is an engine fork carried across every release. Shipped defaults cover 0.1–150 m. **The ladder
draws to 100 km, so 99.5% of the view has no shadow-map coverage.**
*Options:* (I1) accept 150 m; **(I2, recommended)** four cascades out to a DERIVED horizon plus long-range
terrain self-shadowing marched along the sun direction from the octave-dropped height function — no shadow
map, no cascade, no memory, and **it cannot disagree with the drawn mesh because it is the same function
that produced it**; (I3) fork the engine — rejected on maintenance.
*The relation, which must be a derived clamp and never two numbers side by side:*
`R₀ = 1 / (px_per_cell_target × θ_px)`; `shadow_max_distance = min(authored, (1 − crossfade_band_fraction)
× R₀)` — **R₀ = 786 m and shadow_max_distance ≤ 668 m at the defaults**, re-asserted whenever the governor
moves the pixel target.
*Why:* verified — `VISIBILITY_RANGE_DITHER` is set into the pipeline key in exactly three places (main
pass, prepass, wireframe) and `render/light.rs` never sets it, so **the engine does not dither the shadow
pass** and a naive crossfade makes both rungs fully opaque casters — permanent double-darkening and
self-shadow acne at exactly the distance the player is flying toward. The derived horizon closes it **by
construction**. *Two struck formulations:* the authored "300–500 m" (a literal where the relation is
derivable) and "cast shadows only from tiers 0–2", which reaches 3.14 km and therefore **contains** both
the tier-0/1 band (668–786 m) and the tier-1/2 band (1.33–1.57 km) — it guarantees the artefact twice
rather than preventing it.
*Known limit, stated rather than hidden:* builds cast real shadows only inside the cascade horizon.
*Source:* register row 41 / `[8-E]`; D-RND-11, D-RND-12.

---

**O34 · Where does world simulation live in the roadmap, and which six subsystems are in scope?**

*Where:* fire, fluids, growth, weathering, freezing, falling blocks and structural collapse have **no
committed phase**, and the phase they were informally assigned to is actually the player-checkpoint and
crash-durability phase. Options: (A) insert a new phase between block edits and checkpoints; (B) fold it
into the final combat and scale-hardening phase; (C) split — support and collapse early, the rest late.
**Recommend (A)**: the slow-tick scheduler and the shared propagation engine are prerequisites of *both*
the dormant-world pillar and the auto-decoration layer, and the checkpoint phase cannot save per-block
state whose shape has not been decided. **See §7 for the coupling this creates with P7, which the slice
plan currently loses.**
*Which:* the block catalogue quietly requires a power grid, a fluid and pressure network, a thermal model,
item logistics, a crafting recipe graph and a structural stress model. None is in the roadmap; three are
already load-bearing inside the design's own arguments. Options: (A) two now, two with ship interiors, two
behind the economy — **structural stress and the power grid first**, then thermal and fluids, then
logistics and recipes; (B) all six before functional blocks ship; (C) only logic, interface and machinery,
permanently. **Recommend (A)** — both are cheap passes over a graph the block plan already builds, and
structural stress delivers most of the physical-ship feeling people wrongly try to get from joint
simulation. Rider: (C) would remove the power gate that makes a relay cost something to run, which the
relay design depends on.
*Cost of deferring:* each is comparable in size to the signal bus itself; left unnamed they get discovered
one at a time during a phase whose stated goal is a switch toggling a lamp.
*Source:* register rows 13 and 20; D-BLK-1, D-SIG-1.

---

**O35 ⛔ before P6 freezes the saved side tables · Is there a temperature per block at all?**

*Options:* (i) purely event-driven, fire is a dice roll — **which needs a random-number stream synchronised
across machines**; (ii) **a sparse ACTIVE SET** of only those blocks currently hotter or colder than their
surroundings, evicted when they settle; (iii) a temperature value for every block, which provably cannot
work at planet scale.
*Recommendation:* **(ii).** It removes an RNG stream that would otherwise have to stay in step across
machines, and it is what makes the damage model's heat threshold legible — a wood fire at 1,100° simply
cannot melt granite that needs 1,500°, with no rule anywhere saying so.
*Cost of deferring:* one-way. Discovering it after fire and melting are built forces a rewrite of both.
*Source:* register row 14 / `[2-B]`; door 25.

---

**O36 · The cryptographic primitives — one row, four sub-questions, asked in three documents**

Each is a library-adoption call and therefore the owner's by standing rule.
- **(a) The tag on a cross-world message batch.** The keyed hash already used in the workspace and named in
  the phase deliverables (~0.14% of a tick) versus a faster modern hash that is not a dependency here
  (~0.04%). **Recommend the existing one** — the measured 367-fold gap that rules out per-message
  public-key signatures says nothing about these two at per-batch rates, and the tag is the same width
  either way so a swap stays one line.
- **(b) The PRF that places concealed ore.** **HMAC-SHA256** — `hmac` 0.12 and `sha2` 0.10 are already
  declared and used, so zero new dependencies and the project ends with one keyed-MAC primitive. Keyed
  BLAKE3 buys 0.03% of a chunk's generation cost and would be a new declaration. **Forbid keyed
  SplitMix64 / `child_seed` explicitly and in writing** — one observed output recovers the master secret,
  demonstrated — because it is the shortcut an implementer will take.
- **(c) The cipher, if the player-built cipher block ships.** A keystream from the existing `hmac`/`sha2`
  (zero new deps, ~11 µs per 1 kB) / keyed BLAKE3 (~0.8 µs per 1 kB, **extrapolated not measured**, one new
  dep) / a real AEAD such as `chacha20poly1305` (one new dep, the correct tool). **Defer to P9, reserve the
  sealed-blob SHAPE now**, and if it ships prefer the AEAD over a hand-rolled keystream. The 16-byte tag
  against `MAX_SIGNAL_BLOB_BYTES = 1024` makes the sealed-plaintext cap **1008 bytes** — free to set today,
  a format migration once player chat and fleet nets are persisted.
- **(d) The key-agreement primitive.** **The autopilot handshake as specified asks to ENCRYPT TO AN
  Ed25519 KEY, which is not an operation Ed25519 has**, and `x25519-dalek` and every AEAD crate are absent
  from `Cargo.lock`. Either a separate X25519 identity key alongside the signing key, or a signed ephemeral
  Diffie–Hellman. **Take it once for the grant handshake and the radio codebooks together.**
*Measured:* keyed BLAKE3 MAC ≈ 51 ns; HMAC-SHA256 ≈ 345–570 ns; Ed25519 verify ≈ 18.86 µs.
*Cost of deferring:* (a) and (b) are near-free; **(d) is currently an impossible operation in a written
design** and must be fixed before the handshake is implemented.
*Source:* register row 35 / `[7-D]`; concealed C-7; signal D-15, D-32.

---

**O37 · Blueprint paste policy — reframed by R18 and needing re-adjudication**

The original row said an imported configuration **may never WIDEN**, and widening becomes an itemised
confirmation — because the blueprint attack never relied on ownership: **it relies on your own block being
pointed at a channel a stranger can also talk on, because the design said so.** Bind-time authorisation is
defeated by it precisely because the pasted blueprint makes the *victim* the authoriser, so every check
returns yes. R18 then ruled a different mechanism — the dock's connection manifest, informed consent
rather than stripping — and explicitly rejected the strip fix.
*What is open:* R18's manifest covers the **dock** path. **Does paste OUTSIDE a dock still refuse to
widen, or is the manifest the only control? And is the itemised confirmation the same artefact as the
dock's manifest, or a second one?** The two are compatible but they were written in ignorance of each
other.
*Three consequences R18 names that need owners:* ship construction gains a place, a price and a
profession; **the payment is the anti-abuse brake on realm spin-up**, which is a real server cost that
currently has no brake at all; and author royalties require a **durable author identity that survives
copying** — a small design in its own right, because a blueprint is data and data gets copied.
*Source:* signal D-9 against R18.

---

**O38 · The four unowned physics and decoration holes**

Small, unowned, and each is the kind of thing that becomes an unreproducible bug rather than a visible
failure.
- **⚠NEW · A per-realm DECK NORMAL.** "Up" is undefined in a zero-g ship interior — the gravity direction
  can be **zero**, and every up-derived rule then has an undefined branch. It reaches at least three
  shipped rules directly (the upward/non-upward triangle classification, the surface mark's downward-only
  sign, and snow's "up"). A declared per-realm deck normal is the answer and **nothing declares one**. It
  blocks the ship arm of every decoration and surface-response feature.
- **⚠NEW · The per-session EDIT-RATE CAP.** The client's decoration cost is gated **per edit and never per
  second**, and at the design's own density it breaches the main-thread contract by **3–21×**. The missing
  value is *named* in the block-store tuning struct and never given a number, and that number decides
  whether the meadow case ships at all.
- **⚠NEW · The collider cluster MERGE/SPLIT rule.** The 27 MB budget is per **connected component** of the
  cluster overlap graph and the rule has not been designed. Merging is **required** — two colliders for the
  same cells in one physics world double every contact impulse — and 128 players 60 m apart form **one
  component spanning 7.6 km**.
- **⚠NEW · The collider BUBBLE RADIUS against body size.** It was derived against swept motion and never
  against body extent: a projectile drags a 64 m bubble (116 chunk-colliders/s at 800 m/s), and the same
  rule caps a player-built ship at **~118 m before its nose leaves its own bubble**. Both get 2.3× worse at
  once because the forest ruling takes the per-chunk figure from ≤1 MB to 2.3 MB.
*Source:* `collidable_decoration.md` findings A8, A15, A16, A17.

---

**O39 · The remaining subsystem-shaped rows, each with a recommendation and no surprises**

| # | Question | Recommendation | Cost of deferring |
|---|---|---|---|
| **a** | Where does the mesher come from? | **Write our own** (~300–800 lines) — every off-the-shelf option assumes full axis-aligned cubes and would be forked at the first slope, the first seam strip, the first warped rim **and the first skirt**; one candidate is unmaintained since 2022 with a released correctness bug stranded on a branch. Measured: binary greedy meshing ~65 µs/chunk; `block-mesh-rs` ~3 ms | Low and reversible — the integer chunk-geometry output is the seam. **Contradicts `roadmap.json` P4's "binary-greedy-meshing adapter"** |
| **b** | Which shapes can a player walk on? | **(C1) everything except vertical.** The geometry has already removed an option: the same 45° wedge presents 43.5°–56.1° across one planet while the corner-cut facet drops to 53.4°, and those ranges **overlap**, so "wedge walkable, facet not" is inexpressible by any threshold. (C3) different answers on planets and ships is rejected outright — it breaks the run-anywhere gate | Close to one-way in content terms; players build around whatever ships |
| **c** | The collision threshold | **REPRESENTABILITY** (Panel 0.125 m), **not step height** — and this supersedes the collision investigation's own recommendation, because our controller auto-steps (`autostep.max_height` = 0.53125 m) and Valheim's does not, so the Valheim failure mode that is the entire evidential basis for the step-height threshold **cannot occur here** | One comparison against one existing table |
| **d** | Do materials blend softly or meet at hard decorated edges? | **(D1) soft blending across a one-metre strip** — the owner's words are "blend into each other, like real life", and the trimming machinery is needed for the crumbling edge and the shaped lane anyway. Honest counter-argument stated: a smooth blend on a surface that visibly steps in one-metre cubes can read as inconsistent | Not one-way — switching later replaces one shader, not the mesher. ~1 month of shader work either way |
| **e** | Is radio interceptable? | **Open by design — and the row as written answers only READ.** Re-scope it: radio is open for reading **and publishing**; authorisation is evaluated at the **destination realm's ingress**, never at the relay; and **subscription is consent to receive and never consent to actuate.** The relay cannot enforce a read ACL — it never holds the declaration, and the catalogue ships a repeater, so a legitimate reader can lawfully rebroadcast: not merely expensive but **semantically impossible.** Ship the confidentiality wording with it: the server holds every key | Asymmetric — open→private later is additive; private→open later breaks everyone who built around secrecy |
| **f** | Cockpit camera: bolted to hull attitude, or free-look? | **Free-look. The single largest feel lever in the entire latency analysis, and it costs nothing up front.** If a cockpit rigidly inherits delivered hull attitude, the pilot's *view* rotation inherits the full 162 ms loop and flying feels dramatically worse than walking does now | Decided after the ship phase starts, it is a camera-rig rewrite |
| **g** | Is the speed limit derived or authored? | **Derived envelope** — dynamic pressure `q = ½ρ(h)v²` per hull, plus a continuous 0..1 assist scalar where **0 means genuinely raw**. A flat cap is a magic number and violates the seamless law if it manifests as a MODE | Late adoption retrofits per-body ρ₀ and scale height and per-substance thermal limits across the P4 and P6 tables |
| **h** | Where does the flight computer live? | **Split** — the hard limiter as a realm property on the physics-authority shard, the terrain-following autopilot as a functional block. All-engine loses the buildable/upgradeable/economy hook and the run-anywhere fixture | Low |
| **i** | What happens when a ship hits something? | **Energy-based per-block destruction** reusing block LIFE, with a per-substance destruction energy in J/m³. A ship hitpoint bar is a magic number and loses the salvage/repair economy hook. Paired reservation: a **contact/free-flight flag** on the authoritative own-entity state — three consumers, one bit | Cheap now as a substance-table field, a schema migration later |
| **j** | Does the client ever replay its own craft in free flight? | **No, and re-open only after the control-loop fix is measured in-game** (§6). Four mechanisms share the name "prediction" and only this one would help flying; it is genuinely the same *family* as the thing that gave the owner a bad experience, and the load-bearing difference is **contact versus free flight**, not ships versus characters — a walking character is in contact every moment, contact solving is chaotic, one branch flip is ~0.5 m, and that is where correction snaps come from | **Cheap if three reservations are taken** (a trailing applied-sequence field on the snapshot datagram, the contact flag, and a written constraint that own-craft free-flight integration is a shared closed-form path that never enters the physics solver); expensive if they are not |
| **k** | Does the client show a preview block before the server confirms? | **Protocol fixes only, now and unconditionally; measure; adopt a translucent preview only if it still feels soft.** The protocol fixes alone take click-to-pixels from ~250 ms to 73–123 ms. A preview touches **no** standing law — no simulation state, no collider, nothing to roll back — but it *looks* like prediction, so it deserves a ruling rather than a quiet implementation. Applying the edit locally and re-meshing is rejected: its failure mode is the world visibly rewriting itself | Nothing foreclosed by starting with the protocol fixes |
| **l** | Amend the no-prediction law's wording | **Yes — fence the VIEW TRANSFORM as client-local while all simulated state is server-authored.** The client already turns the camera locally on mouse-look and labels it "NOT prediction" in two files; writing the boundary down **blesses what ships and fences the precedent** so it cannot later be stretched. Costs a sentence. Two paired written constraints, currently written down nowhere: own-craft free-flight integration never enters the physics solver, and **client block edits are applied at their STAMPED TICK, not on arrival**, or terrain changes ahead of the drawn ship — 19.4 m at 240 m/s | A sentence now; a precedent argument later |
| **m** | The relay plane's eight open rows | Does a fresh galaxy ship with relays (seed the starter system, or make the first interstellar link a player achievement — the purer design and the harsher first hour, and **not really reversible** once players organise around the vacuum); does destroying a relay destroy its queue (**yes** — emitters learn from lifespan and an undeliverable receipt); is the first message to a new destination visibly slower (**reactive discovery at 267 kB/s** beats proactive flooding at 4.05 MB/s cluster-wide — the cheap answer and the good-feeling answer coincide); the trust model (**untrusted by construction**, with an end-to-end authenticator on grant-bearing items distinct from the hop-by-hop MAC); is the cost purely time (**yes at P9, with a `power` input port DECLARED from day one**); do padding and cover traffic ship (**only WITH byte-denominated pricing in the same slice** — without it padding is a **25× amplifier costing one token** and relay fan-out is ~10 MB/s from one session); is the position leak declared content (**declare it before players believe the cipher block hides them** — any transmitter beyond the bubble is locatable to ±15,000 km with three posts **regardless of encryption**); may player social text ever be sealed (**no** — a validated `sealable=false` on a reserved `chat.*` namespace, **with the residual stated honestly**: players can still build private chat from a keypad, a codec, an antenna and a display) | Each is players building around whichever answer ships |
| **n** | Ship physics and coupling — six rows that arrived through the signal document | The realm frame is **inertial** body-centred (at the sphere of influence a body-fixed frame's centrifugal term is **1,049× real gravity**); `UNIFORM_EXTERNAL` gravity supplied by the grandparent as **24 bytes** cuts the boundary discontinuity **8,412×**, and without it "your orbit changes when you cross an SOI boundary" must be declared as a game rule and a 459 m/s-per-cycle free engine accepted; **lift is per-block** wing and control-surface blocks, because lift is not derivable from a silhouette and a generic lift fudge is a magic number (declared cost: a two-tick, 100 ms, control-surface lag); **quantise the coupling at the PRODUCER** — fixed-point on the wire, exactly one conversion site; **buoyancy in v1** as displaced volume + centre of buoyancy (32 bytes) with an interim occupied-cell approximation; and the aero/geometry fields ride **one ~128 B struct** at 2.56 kB/s per ship, no second flow | Wire-shape decisions on a reserved arm: free today, a protocol change later |
| **o** | Two live specification defects in the signal design | **`Neighbourhood { hops: 2 }` as specified cannot route** — nothing is re-advertised upward, so a grandparent has no index entry and the emission terminates at the first hop. Either the Long-Range Antenna (a **shipped catalogue block** on this plane) does not work, or a 1,000-child system node performs 1,000 lookups and ~13 kB per emission, unpriced. **Recommend cutting to hops = 1** and making cross-planet reach a relay concern, which composes with R17. And **`remote_expanded` is undefined** — the symbol appears exactly once in 17,401 lines and carries the entire pricing claim of the export heartbeat; left undefined the first implementer prices a parent fan-out of K as 1, and human-chosen names collide at ~100%, not 2.7e-12. **Fix: the parent reports fan-out width back to the child on the existing reconcile** | Both must close before the signal wire shape freezes |

---

### Band E — cheap, answerable any time

Each of these is a field, a function call or a taste call. **None blocks a slice**, and every one has a
shipping default so nothing waits on it. They are listed so they exist rather than being typed as literals.

| # | Question | Recommendation |
|---|---|---|
| **O40** | Snow: block family, step, and the derived/authoritative split | **8 layers at 1/8 m** (matches the existing 3-bit width, 8 shape rows = 224 B) **plus the split** — derived cover everywhere, authoritative depth at tier 0. Reserve eight contiguous shape ids now; it costs nothing today and `ShapeId` is persisted, so a later step change halves every saved world's snow |
| **O41** | Are surface marks server-authoritative, and do they survive a spin-down? | **Hybrid** — the server owns 4 bits of displacement, the client renders the tread: 3.07 kB/s, 7.7% of budget, zero new message types, against 10 B/mark fully-detailed (26%) or client-local decals (no shared trails, no relog survival). **Reserve the per-chunk table name now**, populate at P6/P7 |
| **O42** | What radius does the *starter* world get, and what range may the generator draw from? | Residual of a dissolved row. Not blocking P4. Pairs with O43 and O45 |
| **O43** | What *band* of surface gravity may the generator produce? | A feel decision, not architecture |
| **O44** | The enrichment factor — how much advantage does reading geology buy? | **Target 2–4×**, declared as a field of the ore table with a property test measuring digs-to-first-hit with and against public knowledge. Not a one-way door; cheap while it is a field, expensive once players have learned the geology |
| **O45** | Launch biome archetype count and the starter world's set | **10 natural plus 1 identity layer**, 21 slots free. One-way only for the starter world's set — **it is the first thing every player sees** |
| **O46** | The seven provenance and collapse residuals | **Is `Terrain` ever writable** — representable but **not writable** at P6, rejected on the write path as a covered branch a future terraforming tool can relax ("writable when fully enclosed" is a laundering vector). **The physics-source drop gate** — as written every tool-less break yields NOTHING, so landings, explosions, fire and collapse **destroy all matter they touch**; gate only tool-carrying sources, and post every undropped unit as a typed destroyed-matter fact, because it is a very large unpriced sink in a player-driven economy. **What a player does with a landed group** — felling saves only **1.9×**, not 50×, so it is believability rather than an economy lever; the eventual answer is a second production step (logs → planks), which creates a tradeable intermediate and a second profession. **Does falling matter damage builds, and does a landing pass edit admission** — yes and yes; a landing IS an edit subject to reach, the rate cap and every future claim check, or felling onto a protected build bypasses every permission check by construction. **Are grown and planted things `Feature`** — yes regardless of who planted the seed, which keeps the re-fell exploit closed. **`FALL_DP_PER_KJ = 5`** as the fifteenth scale constant. **Gravity-moved matter keeps the provenance it fell with** — monotone, and it launders nothing |
| **O47** | Is hiding in grass a mechanic, and does deep foliage slow you? | **Answer both together — same machinery.** Concealment is one call per player per tick, proportional to players not blades; slowdown is 0.0004 ms/tick at 128 sessions. **The one-way portion is already closed** — the evaluator's placement and the server-side sampling path are committed and paid for by R12. Ladder clause: the concealment sample is evaluated **at tier 0 on the server**, never from whatever tier a client happens to be drawing |
| **O48** | The body bend, and the abundance neighbourhood radius | **Build the bend behind a tuning toggle** (0.05 ms budgeted, 0.006–0.032 ms expected) **with the explicit rule that it is the FIRST thing cut** if the terrain-opaque line needs its full allowance. **R = 8 m** as a per-row field (2.6 µs, 1/289 sensitivity; 4 m reads a 9 m patch as a full meadow, 16 m costs 49% of the typical edit budget) |
| **O49** | Who works out the grass? | **Answered in substance: client-derived, with the server-side sampling path and its dependency edge declared from the first slice.** Shipping it from the server is ~6 MB per player per region, ~150× the entire snapshot budget, and R12 eliminated it permanently on a reason that "will not change with better hardware". **What genuinely remains is narrower than the row's title** — where the evaluator lives (crate layout and the equality gate) |
| **O50** | Does block damage heal, and how far does chemistry go? | **Natural terrain heals slowly, built materials never** (measured: damage drains with a ~9-minute time constant, so the table only grows unbounded above a sustained 146 damaged blocks/s in one chunk — and the memory safeguard is needed regardless). **Rusting and weathering plus a corrosive damage type; no reacts-with web** — it grows the balancing burden as the *square* of the material count for the least gameplay return of any property group in a multiplayer game |
| **O51** | How far does player programmability go? | **Logic blocks plus a bounded arithmetic expression now, a real scripting language gated behind MEASURED demand and written gate conditions.** Three shipped games each failed differently: one killed by a quarter of a million accumulated threads, one running scripts on the main frame budget and commonly disabled by server operators, one moving scripting to the player's machine — which this project's authority law forbids. **One-way in one direction only:** free to move up at any time; once players have written scripts, throttling the runtime is a community incident |
| **O52** | Voice and video | **Text only now with the media handle RESERVED on the wire; self-hosted later; never a managed vendor** (which hands enforcement of the proximity cap to a third party that does not know our world geometry). One talking player costs ~4.5× an entire world-to-world signal link, and voice is the largest recurring infrastructure line in the product. **Attached rider needing its own answer: is voice RECORDED?** That is a consent question, and adding recording after launch means re-collecting consent from the whole player base |
| **O53** | Which tool bakes the text atlas? | **The pure-Rust generator as a BUILD dependency only** — it never links into the shipped client because the atlas is a baked artefact, so the runtime dependency surface stays at zero. Reject the C++ bindings: a C++ dependency undermines cross-binary reproducibility |
| **O54** | Compression, and an analytics export tier | **No compression now** — boxes (O6) remove most of the redundancy and what remains is high-entropy packed indices at ~1.05× — **provided O6's codec tag is taken.** `flate2` is the fallback with **zero new crates** (already transitive via png/tiff); `zstd` is rejected because `zstd-sys` adds a C toolchain to the k3d images, the stable product build **and** the pinned coverage nightly. A columnar analytics export tier is legitimate **eventually and strictly outside** bins → node → sim → wire → core; `arrow`/`parquet` must never enter a Tier-A crate |
| **O55** | ⚠NEW · The panel reduction rung's SHAPE | **Reserve it in the first decoration slice even if the answer is "not yet".** `compose(&self, sources, band)` versus `compose(&self, sources)` is a change to a **closed registry** implemented once per widget; widening it later touches every widget in the catalogue and every golden fixture. **The band argument is a one-way door in shape; the behaviour behind it is not** |
| **O56** | How far can a stranger read your status lights? | **The mechanism is ruled** — a transient per-realm state-light lane composed by the realm owner and shipped at the observer's clamped rung, never in the edit pyramid and never via the signal bus (a client is not a subscriber). **Only the RANGE is open.** Recommend visible at **every rung**, which makes lighting a real build decision — a running-dark warship is a stealth choice a player MADE. Weaker intel than it looks (a stranger sees a lamp is red, not what drove it) but it does compete with the Sensors catalogue and the jammer. Cost: one `STATE_VISIBLE` flag bit plus `state_priority: u8`, with a coherence assertion that every flagged row has a function reference with an emitting port. **This row deliberately stayed with §7 and therefore never appeared in a register the owner reads top-down** |
| **O57** | The HR4 fixture for biome features | A Cartesian test profile has no generator and therefore no features, so the identical-fixture rule is unsatisfiable as the profiles stand. **Give the Cartesian profile a TEST-ONLY generator emitting three boulders**, so the fixture is genuinely identical — legitimate, since the roadmap already commits that the chunk math is shard-agnostic. Weakening the gate instead leaves the project's strongest hard rule partially unenforced on a new subsystem |
| **O58** | The pyramid budget's sizing and its failure mode | **Size against 4 entries per edit, not 1.1** — planetary scatter costs `log₄(A/n) + ⅓` entries per edit, which is **6.07 at 100 M edits** on the starter body, and the cost table has no scatter row. **And make the failure hierarchical** rather than realm-wide: today a breach is a typed refusal of ALL edits on the planet, so **one megaproject bricks editing for everyone.** Per-claim sub-budgets need no realm-model change and no new grid |
| **O59** | The remaining render taste rows | **Anti-aliasing:** temporal resolve plus adaptive sharpening, multisampling **excluded permanently with a start-up assertion** — otherwise someone proposes it for "crisper block edges" and silently deletes occlusion, the temporal resolve, the tier crossfade and the impostor band at once. **Global illumination:** none beyond sky-visibility ambient through P7; revisit at P8 *after* the terrain line is measured. **Engine version:** stay on the current release through P4, take the next as a **dedicated slice before the first custom render pass is written** — it replaces the render-graph API every custom pass would be written against, so the only question is whether the rewrite happens once or five times. **120 Hz:** no at the reference tier, yes as a high-tier preset — saying "60 Hz is the contract" stops the whole gate matrix doubling. **Per-material merge cap at tier 0** (water at 4 cells, wave amplitude ramping on the crossfade band) or accept a permanently flat sea. **Per-chunk rung bias from measured quad density**, bounded to ±1 rung — **but it must also read the composite instance count**, because composites DELETE the quads the bias triggers on, so as specified it cannot fire in a forest, and **0.93 ms of the 1.32 ms frame reserve depends on it**. **Forest density** as an authored per-biome field with a stated ceiling plus a build-gated far-form quad ceiling. **The art contract binds ENTITY art too** — players, ships, projectiles, in-world panels: the entities line is 10.7% of the frame and is currently outside every rule, and a player model authored at a different texel density with baked occlusion in its albedo is the most visible possible violation. **Greybox exemptions become a named, dated, EXPIRING file that warns then fails the build.** **A six-name tolerant-albedo list**, every addition reviewed |

---

## §3 The one-way doors

§8.2 claims to list "every door this design opens". **It lists 48 and eight more have been created or
confirmed since it was written; three of those eight must shut in the FIRST slice, and two are already
partly through.** Two of the existing 48 also carry deadlines that are wrong — one by five phase groups.

The table below is the delta plus the corrections. **The other 40 rows of §8.2 stand unchanged.**

### Doors missing from §8.2 entirely

| # | Door | Must shut | Cost of retrofitting |
|---|---|---|---|
| **49** | **The scoped channel key** (R19) — `H(scope ‖ name)` over Realm/Construct/Net/Protocol, with `dock.*`/`sys.*`/`hull.*`/`chat.*` as compiled constants mintable by nobody | **S0.4, the first slice** | Every fixture, golden pin and persisted binding minted beforehand re-resolves to a different channel — **including the blueprint library**, which is copied and re-sold data, not just live ships |
| **50** | **The three-state provenance ENCODING** — bits 23..22, `0 Terrain / 1 Feature / 2 Placed`, value 3 rejected on decode | **Before P6 writes one edit** | Door 6 covers the record's *width* but not this field's *numbering*. Renumbering later reinterprets every saved cell — and because the values are ordered by anchor privilege, a renumber silently inverts the "provenance never gains privilege" comparison |
| **51** | **The render origin as a rigid POSE** (translation + rotation), server-authored | **S0.3/S0.4** | **Already half through** — `pin`/`pin_abs`/`anchor_epoch` shipped at PROTO_MINOR 7 across nine files plus two integration smoke tests. A wire ABI and protocol-minor change that gets more expensive every week the absolute-position path runs in production |
| **52** | **The generated-surface-skin field**, defaulting to `Cube`, named in the P4 terrain-golden digest | **Before the first planet is saved** | Turning it on later changes generated terrain **under existing player builds** — the same class as the noise-lattice step. A world-format epoch |
| **53** | **The chunk-state wire entry's width** — `(local_index u18, kind u2, value u4)` = exactly 24 bits | **Before the bulk chunk-delta lane is routable** | A protocol bump plus a client migration, for a field that is **exactly the same width** today |
| **54** | **Snow's eight contiguous shape ids** at a 1/8 m step | **Before the shape catalogue freezes** | `ShapeId` is persisted, so a later step change **halves every saved world's snow** unless epoch-gated |
| **55** | **The seven storage-format freezes** (O6) — the box-encoding record layout, the codec tag, the derived-column prune rule, the `state_bits` split, the blueprint-stamp record and its snap lattice, the pyramid storage-block key, and the `CellIndex` linearisation | **Before the first world is written** | Seven separate migrations over every saved world. **The linearisation door is ALREADY SPENT** and is stated only in a doc comment — the risk is that it was spent silently |
| **56** | **The legal radius set** (O2's coarse-lattice divisibility rule) | **Before any P4 code** | A body authored under one rule cannot be re-radiused without regenerating it |

### Doors in §8.2 whose stated deadline is wrong

| # | §8.2 says | The truth | Why it matters |
|---|---|---|---|
| **34** | The channel identifier's width: **"P9 slice 1"** | **S0.4**, in Phase 0, before any P4 feature code — the slice plan lands the content-hashed channel key with its loud collision check there, and there is verifiably no `ChannelKey` in the workspace today | Wrong by five phase groups, and only discovered when a saved binding stops resolving |
| **45** | The requested-tier byte: **"before the P6 arm exists"** | **S0.4** — which already lands the reliable discrete-action carrier **with the block-edit arm** and then adds the two tier bytes itself | A door table that states a deadline *later* than its own slice plan is an invitation to defer, and postcard is positional so the deferral is a protocol bump plus a pyramid re-serialisation |
| **38** | The signal message family's shape: **"P4"** | **S0.4** lands the shape as prose in Phase 0, so P4 is already late — and the shape must now carry four things §8.3's text predates (O15) | Same class |
| **42** | Maximum sustainable flight speed | **Split it.** "Target band" is a gameplay **input** due before P4.10; "achieved ceiling" is an engineering **output** due at P4.2/P4.10. Two registers currently own it | A one-way door owned in two places gets shut twice at different values |
| **2, 16** | The warp curve and the noise source: **"before P4"** | Sharper: **at P4.1's done-when**, which pins a byte-identical 128-bit terrain digest across three targets at two optimisation levels and is also a **committed phase definition of done**. After that pin, any change to generated output is a **world-format epoch**, and the inherited fail-safe on an epoch mismatch is **discard** | Six things shut at that moment, not two: the warp curve, the noise source **and its octave parameter**, O2's divisibility rule, O10's surface-skin field, O8's placement conditioning, and the coarse-lattice cave step |
| **24** | "Whether the server light field carries colour — before P6b" | **Shut.** R2/R3 answered it | A shut door listed as open costs the owner reading time, which is the scarcest thing on this board |

### Doors already passed

| Door | Status |
|---|---|
| `CellIndex` linearisation (`(c*62 + b)*62 + a`, `a` fastest) | **Spent**, in a doc comment only. The chosen branch is also the better one (~2× fewer runs than radial-fastest on terrain). Give it an explicit line so nobody re-opens it |
| The render origin's absolute-position path (**A5 THE FLIP**) | **Committed.** PROTO_MINOR 7, nine consumer files. Door 51 is the *widening*, and its price rises with each new consumer |
| **The 30-arm ceiling on `InterShardFlow`** | **Reached exactly.** 27 arms + 3 reserved names = 30, which is door 38's own stated review ceiling. **There is no headroom left** — everything further rides inside an existing arm as an internal discriminator, and S0.4 must say so in prose |
| The one-metre cell | Answered by the owner |

---

## §4 The slice order

**Forty-two slices** across the roadmap's committed phases, re-derived. Sizes are rough implementation
effort, not calendar: **S** ≲ 400 lines, **M** ≈ 400–1,200, **L** ≈ 1,200–3,000, **XL** > 3,000, and they
price the **production** half only — at HR5's 100% region+branch the test volume historically runs at or
above it.

**The standing rule, stated once:** every phase re-runs the *entire accumulated* transfer / crash / chaos
suite, the RLM demand-loop end-to-end runs and the visual-universe scenarios. A slice that lands a new
persisted table, a new wire arm or a new realm kind has not shipped until they are green on top of it.

**What changed from §8.3**, in one list, so a reader of the old plan can diff it:
1. **S0.6 moves to the FRONT** — it is the only slice blocked on nothing.
2. **Two new Phase-0 slices** — the persistence seam (which does not exist and which two published gates
   are literally unexpressible through), and the control-loop fix (§6).
3. **P4 splits around P5** — the collider spike moves ahead of ~10,000 lines of appearance code.
4. **P4.7 splits** — its declaration half stays early; its reactivity half moves after P6.1, because its
   own acceptance gate needs block placement, which does not exist until then.
5. **Three slices that build superseded designs are rewritten** — P4.1's ore overlay, P6b.2's support rule,
   P9.4's engine relay plane.
6. **P7 is restored** — §8.3 skipped it, and the block system is what makes it hard.
7. **Every slice names the decisions that block it.** §8.3 has 34 "Lands", 34 "Done when", 17 "Must not
   regress", 6 "Planted here because" and **zero "Blocked by"** — which is the one line that makes an
   answer feel worth giving.

> **On the PLANT items below.** They are the cheapest thing in this document and the most expensive to
> skip: a reserved field costs bytes in a file nobody has written; the same field added after the arm
> exists costs a protocol-minor bump, a client migration and, twice, a re-serialisation of persisted data.
> **Every ▣ PLANT line is free today and is not free next slice.**

### Phase 0 — plant slices, before any P4 feature code

---

**S0.6 — Split the single sim shard file along capability lines.** *Size: L (mechanical).* **← DO THIS FIRST**
*Blocked by:* **nothing.** It is the only slice in the entire plan with no external dependency.
*Lands:* a mechanical decomposition of a **verified 15,550-line file** — 60% of the whole `vd-sim` crate,
already working around an engine arity limit twice.
*Done when:* coverage is unchanged and the arity workarounds are gone.
*Must not regress:* Tier-A 100% region+branch; the demand-loop end-to-end runs.
*Why first:* the project's founding diagnosis blames three god-files for a third of its fix commits, and
this one file exceeds all three combined. **It is also the only slice that gets strictly more expensive
with every other slice** — P4.1, P4.7, P4.8, P6.1, P6b.1, P6b.2, P9.1 and P9.2 all add sim systems and a
large fraction land there. It converts owner-thinking-time into progress.

---

**S0.0b — Widen the persistence seam. NEW.** *Size: M–L.*
*Blocked by:* nothing; it is work nobody scheduled rather than a decision anybody owes.
*The finding, verified in code:* `crates/sim/src/io/mod.rs`'s `Store` trait has **exactly five methods** —
`put`, `delete`, `scan(prefix) -> Vec<(Vec<u8>, Bytes)>`, `commit`, `flush`. **There is no point read, no
bounded range and no cursor**, and `MemStore::scan` is a **linear filter over every committed key**, not a
range. **Three of this design's own commitments are unexpressible through it:**
- §8.4's gate "≤ 1 entry write and ≤ 1 pyramid storage-block read per rung" needs a point read that does
  not exist;
- P4.8's "store named by an immutable birth identity", "writable handle obtainable only through a directory
  read" and "double-open detected by the fence" need **per-realm store handles**, and the seam has one
  opaque key-value store per node;
- the WAL's sorted deterministic drain and the three-step compaction checkpoint need a bounded range scan,
  and `scan` materialises the entire prefix run into a `Vec` — which on a planet with 100 M edits is the
  whole delta set in memory.
*Lands:* a point read, a bounded range scan (or a cursor), per-realm handles, and the memory twin for all
of them.
*Done when:* the twin and the redb backend agree on every new method under the existing property tests, and
the block-store fixtures in P4.8 run **entirely on the twin** — no real file.
*Must not regress:* every existing `Store` caller across sim and node; Tier-A 100% region+branch.
*Why here:* `Store` is a Tier-A trait at 100% region+branch with a mem twin, a redb implementation and
existing callers. **Widening it after P4.8 means re-covering all of them.** And the project's hard rule is
that sim/node code gets persistence ONLY through this seam — so the block store is not allowed to be a
private redb handle, and without a twin every Tier-A block-store test needs a real file and the "fast
deterministic suite, virtual clock, no sockets" property is gone.

---

**S0.1 — The geometry seam.** *Size: L.*
*Blocked by:* **O1, O2, O3.** Nothing here can start until the grid family, the legal radius set and the
warp curve are answered.
*Lands:* the one addressing type packed into a single machine word with its **five** detail-tier bits and
its zero-checked spare bits; the cell index type; the one chunk container with palette and bit packing;
the geometry mapping with **all nine operations** — the six already specified plus the fractional
cell-point map, the cell-width bounds and the chart-independent corner identity — **each taking the tier
as a parameter, so a tier-L cell is 2^L metres and nothing branches**; the seam table derived and pinned;
the frame-space reshape (the sector-carrying address plus the anchor-invariant global cell method); the
determinism newtype that makes platform-varying maths unreachable on the generation path; and a
frame-space shape that *admits* more than one anchor while defaulting to one.
*The bit arithmetic, corrected:* three bits give tiers 0–7 and four give 0–15; **neither expresses the
ladder's true top rung**, which is data-derived and is **12** for the 162 km starter body, **18** Earth-sized
and **20** for the largest addressable. Five bits give 0–31 with eleven spare. Layout **3 + 5 + 20 + 20 + 10
= 58 of 64, six spare**, all zero-checked.
*Done when:* the exhaustive mapping gate passes — neighbour lookup involutive on every seam cell, the
four-step lateral loop closing everywhere except the eight corner vertices where it closes in three, every
cell having exactly four lateral neighbours, address round-tripping for every cell, and the apron gather
watertight at all 24 boundary strips and all 24 corner-diagonal slots — at the smallest configuration,
which is a complete proof because the table does not depend on planet size — **and the same enumeration
passes at more than one tier** (O29).
*Must not regress:* the existing pose and coordinate tests; the demand-loop end-to-end runs.
*Why here:* every line written before this seam exists is written against an unknown interface, and the
retrofit is a rewrite of every producer plus its full-coverage tests.

> ▣ **PLANT:** the five tier bits; the six zero-checked spare bits; the nine operations each taking a tier;
> the anchor-invariant global cell address (door 28 — the single thing that makes a world re-centring
> survivable without a whole-world visual regression); the multi-anchor-admitting frame-space shape.

---

**S0.2 — The block registry and the shape catalogue, declared.** *Size: L.*
*Blocked by:* **O5** (the record layout), **O21** (the crumble amplitude, because `A_max` is a
build-asserted registry ceiling), **O24**'s two early halves, and R14's pass-through shape class.
*Lands:* the substance, shape, function and block-type registries in core using the proven
generated-exhaustive-match idiom; the checked-in physical facts as constant data (no parser, no new
dependency); the derivation functions and the fourteen named scale constants; the append-only manifest
with a **build failure if an already-issued row's meaning changes**; the identity digest over
identity-bearing columns only, in the connection handshake and the store header; the **full 23-shape
catalogue declared** with everything but the cube unplaceable; the frozen 24-rotation integer table and the
mirror flag; the golden roster pin; and the deterministic **coarse-shape table** the pyramid's coarse cells
are drawn through — indexed by octant mask and fill bucket, 4,608 bytes, **folded into the registry hash so
a mismatch is a refused connection rather than a silently wrong distant silhouette**, with no tunable
threshold anywhere on the path.
*Done when:* the six-part drift tripwire passes for all four registries, including the round-trip assertion
that a hand-edited row fails the build, and the identity digest is pinned.
*Must not regress:* Tier-A 100% coverage.
*Why here:* declaring the catalogue in full at P4 — even unplaceable — is what makes the
character-controller derivation well-defined from the first commit. **It is the cheapest commit in the
design and the only one that gets more expensive every week.**

> ▣ **PLANT:** R9's `TERRAIN_UNSTABLE` bit in the per-material flag word, **unset on every v1 row with a
> coherence assertion**, plus the per-realm cave-ins toggle — without it, collapsible terrain later becomes
> a table migration over every saved planet **plus a re-balance of every rock row**, instead of a
> one-material change. `A_max` asserted at 0.20 m. Snow's eight contiguous shape ids (door 54). A
> **pass-through shape class** for R13 bushes and R14 canopy. The `STATE_VISIBLE` flag plus
> `state_priority: u8` with its coherence assertion (O56). The foliage shape family and the not-drawn-as-a-
> cube flag bit on the shape row.

---

**S0.3 — The render seam widening.** *Size: M.*
*Blocked by:* **O27** (the vertex format's *schedule*, not its shape), **O32**'s angle re-parameterisation,
**O16/door 51**.
*Lands, alone, while there is still exactly one producer:* rotation on the primitive transform; indices;
the four build-bounded vertex streams; the pass class; the per-chunk header carrying the world-local
origin, the seed-derived salt, the occupancy slot, the biome and a **USED** `detail_tier` byte, plus the two
crossfade margin ranges per chunk entity. The detail tier must be a **per-draw uniform**, never per-pixel
camera distance — a per-pixel fade and a per-entity dither band disagree *inside* the band, which is where
it shows.
*Done when:* the existing proxy scenes render byte-identically, and each of the three unused vertex arms is
constructed and exercised by a synthetic test — **those synthetic constructions ARE the mesher's future
acceptance fixtures, written once and used twice**.
*Must not regress:* every existing visual scenario, pixel-identically.
*Why alone:* a rotating planet or a banking ship is already undrawable today; and after terrain, edits,
shaped blocks and hulls exist, widening is a simultaneous migration of four producers plus their golden
fixtures.

> ▣ **PLANT:** the render origin widened to a **rigid pose** (door 51) — it is what makes the atmosphere
> correct on a sphere, and the alternative is a permanent engine fork; the price rises with every new
> consumer of the nine that already exist.

---

**S0.4 — The wire and transport plant slice.** *Size: L.* **← the densest plant slice in the plan**
*Blocked by:* **O12, O13, O14, O15, O17, O18** and R15/R17/R19.
*Lands:* two new client-facing routing classes — a **reliable event class** and a reliable-paced bulk class
— in one slice with their reliability mappings and pinned discriminants, naming all their consumers in the
doc comments (this discharges ledger item **D-4 🟥**); the reliable client-to-server discrete-action
carrier **with the block-edit arm** and a reserved arm for the later fire trigger; the fixed-point value
type, the value enum, the **scoped** channel and world keys with their loud collision check, and the
copy-preserving dedup key, all in core; two appended world-kind tags for ships and constructions; a
reserved transferable kind for a voxel volume; a reserved permission-token field; and the block-edit,
pyramid-fetch and signal message-arm **shapes** written into the reserved block as prose.
*Done when:* the four structural conformance tests over the closed message taxonomy pass with the new arms,
and the stale prose arm counts in four files are re-synchronised with the code.
*Must not regress:* the whole transfer and chaos matrix.

> ▣ **PLANT — eight items, every one free today:**
> 1. **The scoped channel key** — `H(scope ‖ name)`, four scopes, `dock.*`/`sys.*`/`hull.*`/`chat.*` as
>    compiled constants **mintable by nobody** (door 49). *Not P9. Here.*
> 2. **`tier: u8` on the request AND a tier descriptor on the delta payload, TOGETHER**, because postcard
>    is positional (door 45).
> 3. **Four saga step-id constants in ONE file** — 19 = block-store flush, 20 = block-edit forward,
>    21–24 = durable signal config. A collision here is silent, durable and unrecoverable (O14).
> 4. **The arm-ceiling ruling, in prose:** 27 arms + 3 reserved names = 30 = door 38's review ceiling.
>    Everything further rides **inside** an arm as an internal discriminator (O15).
> 5. **The relay header** — hop count, arrival deadline, dedup identity (R17), or it is a migration.
> 6. **`BundleAuth{mac, key_epoch, grant}`** and **`ChannelDecl{group: ChannelGroupId(u16),
>    accepts_relay_origin: false, max_relay_hops: 0}`** — *the default IS the security property.*
> 7. **The three-ladder boundary, as doc comments on each survivor:** the chunk key's content `tier`,
>    `coarsen_level` as a pose hop counter, and `tier_floor` as client-declared, server-clamped,
>    **non-authoritative**, egress-selection-only. `observer_tier` and `min_demanded_tier` are struck.
> 8. **A trailing applied-sequence field on the snapshot datagram** and **a contact/free-flight flag on the
>    authoritative own-entity state** — three consumers, one bit, and together they are what make O39(j)
>    cheap to re-open rather than expensive.

---

**S0.5 — The within-world area-of-interest reshape.** *Size: L.*
*Blocked by:* S0.4's index key.
*Lands:* the per-cell spatial index and the per-subscriber addressing that replaces whole-world broadcast.
This discharges ledger item **D-9 🟥** (measured: 128 sessions in one realm → 357,248 gateway messages,
peaking 1,920 msgs/tick over 7.6 s).
*Done when:* the density fixture shows message counts proportional to observers-with-that-chunk-resident
rather than observers × events.
*Must not regress:* the density fixture's correctness assertions.
*Why here:* **it is a precondition of block edits, not a follow-on.** The same index is what chunk
residency, delta manifests, edit fan-out, panel addressing and anchor partitioning all key off — one
consumer instead of five. Without it, panel draw lists leak a private bridge's readouts to a stranger
standing in the same station lobby.

> ▣ **PLANT:** chunk residency is per **(chunk, tier)**, so the index key gains the tier — free here, a
> re-key later.

---

**S0.7 — The control loop. NEW.** *Size: S–M.*
*Blocked by:* §6's snapshot-rate question, which must be settled first because it changes the correct
buffer depth by 2.5×.
*Lands:* all three fixes together (see §6) — the interpolator actually running, the buffer set to two or
three steps rather than six, the command rate raised 20 → 50 Hz, a **held-input resource plus a per-tick
integrate** on the shard, and the buffer depth moved into the tuning struct as a **per-context** field with
an adaptive control law behind it.
*Done when:* a two-sample track at one-tick spacing driven by consecutive render ticks shows the sampled
pose **changing** every frame; input applied with a dropped command packet still moves at configured speed;
the measured control loop is no worse than today's ~167 ms.
*Must not regress:* the no-prediction law — **nothing here predicts anything**; it is one operational
parameter in one config struct, which the no-magic-numbers rule already requires.
*Why all three in one slice:* **fixing the interpolator alone takes the loop from ~167 ms to ~257 ms, 55%
worse — and the no-prediction law will be blamed for it.**

---

**S0.8 — Format version and migration skeleton. NEW.** *Size: S.*
*Blocked by:* nothing.
*Lands:* a format-version byte per persisted record family, and an offline `vdctl world migrate` skeleton
with a no-op v1→v1 transform and a test.
*Why:* the design's stance is that every door is shut correctly so no migration is ever needed. The
inherited mechanism is "every record tagged `universe_epoch_id` + `epoch_schema_version`, **mismatch →
discard/abort, fail-safe**" — which is genuinely fail-safe for terrain, regenerable from the seed, and
**catastrophic for player deltas, the edit pyramid, blueprints and ship interiors, none of which is**.
Two pieces of evidence that "never migrate" is optimistic: **R6 spent one of R1's reserved bits within a
day of R1 being taken** (exactly what the slack was for, and exactly what a first migration looks like),
and O6 names **seven separate storage-format freezes** with no door row between them. This slice converts
the first mistake from a wipe into a day.

---

**S0.9 — The harness and benchmark tooling. NEW.** *Size: M–L.*
*Blocked by:* nothing.
*Lands:* the tools §8.4's 46 gates run on, none of which is currently a slice or has a size — `decor_pop`
with its three traverses (a 200 m forward walk at eye height at 5 m/s, a 5 km flight at 100 m/s, a 300 m
walk-away-and-return, asserting no frame-to-frame discontinuity in per-band luminance or coverage,
evaluated separately in the lower screen half); a **terrain-tier** variant of the same detector;
`vdctl world verify-pyramid`; the per-tier bench harness over 13 rungs; a 10,000-configuration
golden-vector runner; a build-time shader-variant compile check; a software-adapter parity runner; and a
**scheduled real-hardware parity run on each shipped backend with a named owner** (D-DEC-6 🟥).
*Also lands the four measurements that must precede any budget being spent:* the large-opaque-quad
throughput rate; **the terrain-opaque shading line (2.51 ms bottom-up against a 4.60 ms allowance — the
largest unmeasured number in the design)**; the composite and forest fill lines at the depth-rejected
denominator; the generation reference benchmark on the **reference** CPU (the budget has 19.3% headroom
against the wrong denominator); the forest collider benchmark (**"colliders: zero" was an accounting
error** — the real figure is 12–20 MB of a 27 MB per-cluster budget in dense forest, and a forested surface
chunk is 1.35–2.25× the published per-chunk worst case the whole budget derives from); and **pinning the
40 kB/s snapshot budget, which is INFERRED, not measured**, and which every wire percentage in three
documents rides on.
*Also lands, and it is what keeps the inner loop usable:* **a wall-clock budget per gate tier** — inner
loop ≤ 2 min, pre-merge ≤ 15 min, scheduled = everything else — with all 46 gates assigned. Several are
plainly not pre-merge: the cross-target digest over **every legal tier** (~832 digests per target-config
on the starter body), the real-hardware parity run, the 300-frame GPU-timestamp run, the
10,000-configuration golden table.

---

### P4a — the ground you can stand on

**The reordering, stated plainly.** §8.3 puts four cosmetic slices (material seam strips, the crumbling
edge, transparency, decoration — ~9,000–11,000 production lines) **ahead of the collider spike**, which is
itself labelled "S but blocking" because the cube-next-to-slope classification case **is validated by no
shipped implementation anywhere**. If that spike comes back badly the mesher's output contract moves, and
everything authored against it moves with it. So P4 splits around P5.
*Honest counter-argument:* the crumbling edge changes the merge ratio and the merge-ratio regression
baseline is pinned at P4.2, so deferring it means **re-pinning once**. That is one regression re-pin
against de-risking the collider seam before ten thousand lines are written on top of it.

---

**P4.1 — The generator.** *Size: XL.* *Blocked by:* **O1, O2, O3, O7, O8, O10, O11.**
*Lands:* seed-derived planet parameters through the existing integer generator with closed-form samplers
and no rejection loops; the height field sampled as 3-D noise on the unit sphere (so it is continuous
across all twelve edges and eight corners **by construction**); the strata table with saturating depth;
biomes; **water as an ordinary material**; parametric-tube carvers; **the PUBLIC ore overlay**; and the
conservative generator band with its stated invariant that it bounds *generated content only* and must be
unioned with the column's edit-presence bitmap wherever used as an emptiness answer. **Octave dropping:
tier L evaluates the same sampler with the top L octaves omitted** — no second sampler, no precomputed
pyramid, no second code path.
*⚠ CORRECTED from §8.3, which builds a superseded design:* §8.3 lands "the ore overlay split into a public
half compiled into the client **and a concealed server-only half**". **R11 rules the opposite** — the
concealed half is **a drop table, evaluated only at tier 0 on the server at break time, and it never enters
the block grid.** That single rule is what makes the cost zero and what structurally closes the pyramid
leak, the CAS break, the crack-timing oracle and the **rim-warp tell** (revealing an exposed ore cell would
be a *geometry* change, because ore's 0.08 m rim-warp amplitude differs from loose stone's 0.10 m and the
corner amplitude is a minimum over the corner's solid blocks). Building it as a grid overlay reopens all
four and reintroduces the refuted reveal-traffic cost model.
*Two obligations that ship with octave dropping:* the conservative band **must be derived per tier**,
because dropping octaves narrows the height range and a band computed at tier 0 and reused at tier 5 is
wrong in the unsafe direction; and the **envelope property test** — the tier-L height at a point must lie
within the tier-0 height's min and max over the containing coarse cell. That obligation **fixes the SIGN**
(the crack can only open downward, so a downward apron always closes it) and its companion,
`terrain_tier_agreement`, **fixes the MAGNITUDE**. A skirt needs both; either alone leaves a hole.
*Done when:* the same seed produces a byte-identical 128-bit digest on three targets at two optimisation
levels, and the client crate and the sim crate produce equal digests — **with the digest set indexed by
(seed, chunk key, TIER) and the cross-build diff covering EVERY legal tier of the fixture body** (0 …
`tier_depth − 1`; 0–12 on the starter world), because `h(dir, L)` is a different function per L and a tier
that passes proves nothing about its neighbours.
*Must not regress:* the existing celestial determinism pin.
*Roadmap scenario this phase owes and §8.3 never picks up:* **"transfer-near-terrain" — a point crosses a
boundary while both shards are generating terrain, and terrain CPU load must not blow saga timeouts.** That
is the one scenario connecting the block system to the transfer machinery it must not break, and 0.885 ms
per chunk × a horizon fill is exactly the risk it exists for.

> ▣ **PLANT:** `concealed_key_id: [u8; 8]` on `RealmDescriptor`, refused loudly on open (O8). The
> generated-surface-skin config field defaulting to `Cube`, **named in the digest** (door 52). The season
> index if O11 takes it — it widens the DoD from `f(seed)` to `f(seed, season_index)`.

---

**P4.2 — The integer mesher, cube lane only.** *Size: L.* *Blocked by:* **O39(a), O24's two early halves, O21's tier rule.**
*Lands:* a new Tier-A crate **added to the coverage list in the same commit** (a crate omitted from that
list is silently exempt); occupancy masks, face-mask culling, greedy runs, and the integer chunk-geometry
output; **the shared border-trimming operation, specified totally and exactly for every rectangle down to
1×1 — written first, because three separate cut rules consume it**; the two-material reduction by lowest
identifier, which is total, tie-free and chunk-independent; and **skirts** — a short downward apron per
chunk on the **four lateral borders only** (tier is assigned per chunk *column*, so vertical neighbours are
always at the same rung — a free 33% saving putting skirt geometry under 1% of a chunk's quads). The
apron's depth is the derived `skirt_drop`: the relief across one cell of the **next coarser** rung, read
from the derived, never-persisted relief envelope, **plus the registry's maximum rim-warp amplitude**,
times `skirt_safety` (default 1.25).
*Why skirts and not vertex snapping:* transvoxel is correct rather than concealed, but it is a second
lookup table (13 sample points, 512 cases) built for smooth iso-surfaces, and adopting it means a second
mesher and a second shape catalogue — an HR3 violation. **Our surfaces are axis-aligned quads at every
tier, so two rungs meet in a vertical STEP, not an interpolated curve, and a skirt closes it exactly.**
Held in reserve if skirts prove visible (D-SHP-6).
*Done when:* a fixed chunk fixture's quad list is **pinned**, and the throughput gate holds **as a per-tier
curve over every legal rung** (§8.4, promoted).
*Run-anywhere fixture:* one fixture meshing a chunk with a material seam, on a spherical planet profile and
a flat ship profile, with equal results.
*Must not regress:* nothing yet — **but the pinned quad list becomes the regression baseline for P4.4 and
P4.5**, which is exactly why O24's face-hiding half is due here.

> ▣ **PLANT:** **the composite emit-exclusion hook as a NO-OP** (O26/SL-6). A recognised composite's cells
> are excluded from the emit step at tier 0 only; the recogniser is deferred with R4, but the hook is not.
> Retrofitting an emit rule into a 100%-covered mesher later means re-covering it and re-baking every
> recipe. **The hook costs one branch and one always-empty set today.**

---

**P4.3 — The encoder, the material and the texture arrays.** *Size: L.* **Preceded by a spike.**
*Blocked by:* **O19, O20, O27.**
*Lands:* the spike first — one extended material with one storage buffer and one per-instance tag, drawing
two chunks with different occupancy slots, on two graphics backends, to establish whether extending a
bindless base material can carry storage buffers at all (D-RND-9); then the Tier-B encoder, the extended
material with three shared-index texture arrays plus a macro-variation array, the exact integer-modulo
world-space texture coordinates, and the anti-repetition layers with anti-tiling as a per-material opt-in
defaulting to off; **and the variance-aware roughness mip chain in the bake specification** (door 46) — a
bake-tool change with no runtime cost that **must exist before any material is authored** and **does not
exist today**.
*Done when:* one planet chunk renders pixel-identically at the world origin and at a billion metres; total
video memory is asserted at start-up; and the forbidden shader-effect combinations are rejected by a
build-time assertion.
*Must not regress:* the mesher's pinned quad list.
*Fallback if the spike fails, ledgered:* a standalone material reimplementing the lit fragment shader, at
the cost of a re-merge every engine release.

---

**P4.7a — The decoration evaluator's PLACEMENT only. SPLIT.** *Size: M.* *Blocked by:* **O49, O19.**
*Lands:* the crate, the **declared sim-to-decoration dependency edge with a module-boundary check from the
first slice**, the anchor-invariant key, the positional-hash contract and its golden-vector gate, and the
server-side sampling path **declared** (door 27).
*Why split:* §8.3's P4.7 is XL and its own acceptance gate — "place a soil block, place four hundred more,
place a wood column and leaf blocks, force a re-centring" — **cannot run at P4.7, because block placement
does not exist until P6.1 and the build loop is P6.4**. The only ways out are a test-only edit path, which
is a stand-in and collides with the standing law that acceptance tests drive the exact shipped path, or
moving the slice. Decoration's entire value proposition is that *the surroundings react to the player
adding and removing blocks*; scheduling the reactive subsystem before the action it reacts to is the
ordering error. **The one-way half — where the evaluator lives and what its hash reads — genuinely must be
early**, so it stays. The render path and the reactivity scenario move to P4b/P6.
*Done when:* the 4,096-input golden-vector gate is in the pre-merge suite and the module-boundary check
fails a deliberate violation.

---

**P4.8 — The per-world block store skeleton.** *Size: L.* *Blocked by:* **S0.0b, O4, O5, O6.**
*Lands:* the store named by an **immutable birth identity** that never changes when a ship flies between
star systems; the **type-level single-writer proof** (a writable handle obtainable only through a directory
read that returned this shard as owner, and a non-owner handle with **no mutating method at all**); the
tuning struct; the owed real-scale path-resolution inversion, which lands *with* the storage layer, not
after it (D-WLD-7); **and the tier-0-only write invariant expressed in the type signature — a write takes a
tier-0 address, so `write(tier != 0)` cannot be written at all** (door 47).
*Done when:* a store refuses to open against a fence that is not strictly greater than the recorded one,
and **double-open is detected by the fence rather than by a file lock** (advisory locks over network
filesystems are unreliable and nothing may depend on them for correctness) — **and the whole fixture runs
on the memory twin**, per S0.0b.
*Must not regress:* the durability suite.

> ▣ **PLANT:** the per-session **edit-rate cap** as a named field with an actual number (O38) — it is named
> in this struct today and never given one, and that number decides whether the meadow case ships.
> `pyramid_storage_block_cells`. The format-version byte from S0.8.

---

**P4.9 — The shape bake and the derived controller constants.** *Size: M.* *Blocked by:* **O39(b), O39(c).**
*Lands:* the bake — templates authored in **cell** coordinates, the 48-transform orbit deduped by exact
integer vertex equality with winding reversed for improper transforms, the **two** coverage rasterisations
(conservative for the occluder, liberal for the occludee, so the only possible error is under-culling and
therefore overdraw, **never a hole**) and the occlusion matrix, and the five closure assertions (total
mirror closure, total group action, exact volume equality **with no tolerance**, cell containment on the
thirty-second lattice, and the connect envelope); plus every character-controller length derived as an
exact integer number of fine lattice cells from the catalogue and the warp bounds.
*Done when:* all five assertions pass and the derived constants are checked in as pinned values.
*Must not regress:* the registry digest.

---

**P4.10 — Tier selection, the crossfade, and the material feature ladder.** *Size: M–L.*
*Blocked by:* **O31, O32, O33.**
*Lands:* the client-side half of the ladder, all of it reusing machinery that already exists — tier
selection by screen-space error computed from **the same angular-size rule the locked area-of-interest
model already uses, one level down** (no second scheduler, and nothing branches on realm kind or tier);
O32's five tuning fields with the resident-byte governor, **its floor on tier 0's radius**, and the
assertion that it touches nothing gameplay-visible; the crossfade through the engine's own dithered
visibility margins, **with the fade-direction rule enforced in the code that assigns the margins**;
coarse-before-fine as an ordering constraint on the request queue (one resident-tier bitmask per column,
near-zero CPU — **this is what makes "the realm proxy is the coarsest rung" operationally true**); and the
**material feature ladder keyed on the per-draw `detail_tier` uniform** — fade the 4 m detail tile out and
the macro layer in, drop the normal map, drop parallax, drop the seam blend, all as branchless lerps.
**Required, not optional:** at tier 5 the tile repeats 496× across one chunk and hardware mip selection
returns its average colour, so without it the coarse rungs look *worse* than no ladder.
*Done when:* the tier-count-under-load gate (**≤ 7,500 resident chunks INCLUDING crossfade residency,
measured WHILE MOVING**) and the no-pop gate hold, **including the forward-walk traverse at eye height**,
not only the flyover — that is the case the reference title visibly failed.
*Must not regress:* the frame budget; the demand-loop end-to-end runs, whose thresholds this shares.
*Note the size correction:* §8.3 prices this M. **A hysteretic governor is a state machine in `vd-client`,
which is Tier-A at 100% BRANCH** — every clamp, every floor and every direction of every hysteresis
transition needs a directed test. Budget L.

> ▣ **PLANT:** `minimum drawn terrain radius ≥ v_max × 4 s` as a hard floor beside `tier0_min_radius`
> (O31) — the one rendering-budget constraint that can make the no-prediction law unsafe.

---

### P5 — physics, moved ahead of the appearance work

**P5.1 — The collider spike.** *Size: S but BLOCKING.* *Blocked by:* **O28.**
*Lands:* a reproduction of the cube-next-to-slope classification case and the chunk-seam case against the
pinned library version, plus the physics-state snapshot-and-restore determinism check that already gates
this phase (**SPIKE-6a, a committed P5 deliverable that §8.3 does not name**).
*Scope that must be re-cut before it runs:* the scripted topple **removes** an argument for the sparse
voxel shape while per-cell friction and the hull collider **add** two — see O28.
*Done when:* the answer is recorded and O28 is closed. **This must run before the phase commits to a
collider architecture.**

**P5.2 — The four collision lanes.** *Size: L.* *Blocked by:* P5.1.
*Lands:* cube cells on flat worlds through the library's voxel shape (subject to the spike); cube cells on
planets through greedy-box decomposition in cell space with boxes capped at eight cells per axis, each box
emitted as a convex hull over its eight mapped corners; shaped cells on flat worlds through baked collider
templates with shared-shape deduplication; shaped cells on planets through per-cell warped convex parts;
collider residency bounded to a radius around dynamic bodies; and the rule that rigid-body mass properties
are computed from the **block field** with cell-nominal volumes, never from collider geometry. **Plus
O38's two unowned rules: the cluster merge/split rule, and a bubble radius derived from body extent as well
as speed.**
*Plus one assertion, not a feature:* **collision is tier 0 only, and the collider builder refuses a chunk
whose address carries a non-zero tier.** Not a policy in a document: a refusal in the code.
*Done when:* a character walked across a chunk boundary at twenty sampled offsets receives **no vertical
impulse**; the collider budget per dynamic-body cluster holds **including the forest figure** (12–20 MB of
27 MB, not zero); collider construction from the block field is bit-identical across binaries; and the
tier-0 refusal is exercised.
*Must not regress:* the crash and chaos matrix.

**P5.3 — The character controller.** *Size: M.* *Blocked by:* **O39(b).**
*Lands:* every threshold derived from the catalogue and the warp bounds rather than typed; the separation
assertion between walkable and non-walkable classes; the single trigonometric call evaluated **once per
world at activation**, quantised and carried in the world checkpoint so a re-home cannot change a player's
climb behaviour.
*Done when:* the derived constants match their pinned values and the separation assertion holds.

---

### P4b — appearance, after the collider seam is proven

**P4.4 — Material seam strips.** *Size: M.* *Blocked by:* **O39(d).**
*Lands:* the seam predicate fired on the shared trimming operation, and the two-material ring shader path
with height-based selection, **dropped at tier ≥ 2** — a blend across a strip one cell wide is meaningless
when the cell is 32 m. *Done when:* a dirt-to-gravel boundary renders correctly at three quad sizes and the
tap count is asserted from a shader-variant compile check at tier 0 (≤ 32 taps) and at tier ≥ 4 (≈ 2).
*Must not regress:* the merge ratio measured in P4.2 outside seam regions.

**P4.5 — The crumbling edge.** *Size: M.* *Blocked by:* **O21.**
*Lands:* the rim-only displacement computed on the processor in the Tier-A mesher and baked into the
vertex, keyed on the chart-independent corner identity from S0.1; the rim predicate as a 256-entry constant
table; the inward-bias derivation quantised to the existing fine lattice; the third cut rule — any lattice
point with a non-zero amplitude is a barrier in the greedy sweep — reusing the same trimming operation;
**and the tier rule: metres, not scaled by tier, exactly zero at tier ≥ 2.**
*Done when:* the 10,000-corner golden table passes including a mandatory cube-face-seam continuity block
and all eight cube corners; the exhaustive 256-configuration test asserts **no vertex moves outward through
its own face plane** and no displacement exceeds the amplitude; the merge-ratio regression is inside its
bound; **and a tier-2 chunk's quad list is byte-identical with the crumble path enabled and disabled**,
which is the assertion that the tier rule is real.
*Must not regress:* the merge ratio on constructed materials, **exactly** unchanged because their amplitude
is exactly zero.

**P4.6 — Transparency.** *Size: M.*
*Lands:* the second and third occupancy masks; the five face-emission rules; three pass classes with the
translucent class writing no depth and casting no shadow; water as a translucent material with an animated
scroll and depth-based absorption. *Done when:* a water surface, a glass pane and a leaf card each render
correctly against every other class.

**P4.11 — The long-range lighting answer.** *Size: M.* *Blocked by:* **O33.**
*Lands:* the **derived** shadow horizon across the engine's four cascades (a compile-time ceiling, not a
setting) — `shadow_max_distance = min(authored, (1 − crossfade_band_fraction) × R₀)`, **668 m at the
defaults**, asserted at start-up and **re-asserted whenever the governor moves the pixel target**;
long-range terrain self-shadowing computed by marching the octave-dropped height function along the sun
direction, either per pixel or amortised into a per-chunk-column horizon-angle texture — **no shadow map,
no cascade, no memory, and it cannot disagree with the drawn mesh because it is the same function that
produced it**; the assertion that **a chunk's shadow-caster mesh IS its visual mesh, one object, both
passes**, and that any separate shadow rung ever introduced is selected by the **light's** screen-space
error rather than the camera's; and runtime geometric specular anti-aliasing (~10 ALU and two derivative
pairs per pixel, no memory, no bake), needed *specifically* because the design deliberately warps rim
geometry — a whole planet's worth of high-frequency geometric normals viewed from 1 m to 100 km.
*Done when:* the shadow double-cast gate passes at every tier boundary **and the negative control at 900 m
SHOWS the acne band**, or the gate is not testing anything; and a mountain shadow falls correctly across a
valley at 20 km.
*Known and accepted limit:* player-built geometry casts real shadows only inside the cascade horizon.

**P4.12 — The art pipeline. NEW.** *Size: M (tooling) + external lead time.* *Blocked by:* **O19, O20, O22, O23.**
*Lands:* the variance-aware roughness bake tool (door 46 — **it does not exist and cannot be done at
runtime**); the impostor bake tool if O23 takes impostors (door 48); the asset provenance manifest with
**verified** triangle, texture and material counts (D-DEC-5 🟥 — the one-draw-call plan currently rests on
an unverified community claim); and the greybox exemption file that **warns and then fails the build**.
*Why a slice:* art lead time is the longest pole in the set — a commissioned original set is months and
~€15–40k — and it currently appears in the plan as a single register row about where props come from. **Two
one-way doors are gated on tools that do not exist.**

---

### P6 — block edits and durable persistence

**P6.1 — The edit pipeline, and the edit pyramid.** *Size: XL.* *Blocked by:* **O4, O5, O6, S0.0b, P4.8.**
*Lands:* the reliable action carrier from S0.4 carrying a **resolved target** — cell, face, hit octant,
intended identity and orientation, and a nonce — with the server **validating rather than re-aiming**; the
two-stage protocol (an effect-free off-tick acceptance receipt plus the authoritative echo on the applying
tick); the full validation list (reach against the *server's* pose, line of sight, occupancy, permission,
the identity row exists — which **is** the per-material admission check — orientation in the derived legal
set, the shape is placeable, pin permission, structural support, inventory, batch cap, rate limit, nonce
dedup); the append-only edit log with its sorted deterministic drain and the three-step compaction
checkpoint; the chunk record with its two encodings and the exact crossover **plus O6's box arm**; the
forwarding arm for edits arriving in overlap regions; the separate undo journal written in the same
transaction; and the fan-out rule that all edits committed in one tick for one chunk become one message per
observer per chunk per tick, addressed by chunk residency.
**And the sparse edit pyramid** — maintained by the realm owner on commit of each edit batch: mark every
touched cell's parent dirty, then for each rung recompute each dirty parent from its **eight children's
summaries**, resolving a missing child through the **generator** rather than through disk, **stopping at
the first rung whose summary equals the EFFECTIVE (stored-or-generated) summary** — not the stored one; see
O4 — and *deleting* any entry that comes to equal what the generator would produce. **≤ 1 entry write and
≤ 1 storage-block read per rung per touched coarse cell; zero reads of the edit log; zero reads of any
tier-0 delta.**
*Open sub-question that must be ruled here, not left:* does the walk-up ride the edit log's own write
transaction, or move into compaction? §8.3 and §8.4 say the former; the storage analysis says the latter,
because the pyramid is fully reconstructible so the crash consistency is not needed. **Both cannot ship,
and it is not cosmetic:** under the first the P6.1 gate "pyramid recomputed from the deltas is
byte-identical" is an invariant; under the second it becomes a post-compaction assertion.
*The audit lands here, not later:* `vdctl world verify-pyramid <realm>` plus its test-suite assertion after
every edit scenario — **a derived structure that can silently disagree with its source needs its audit
built in from the start, or the first symptom is a player noticing their tunnel is the wrong shape from a
hilltop.**
*Done when:* edits survive a hard kill and a re-shard; the log's byte output is deterministic; the
click-to-pixels budget holds; the fan-out is measured against the density baseline; the
**edit-survival-across-tiers** gate passes (dig a tunnel, build a tower, fly away, confirm both are visible
at every tier boundary, fly back, confirm nothing popped); and the pyramid recomputed from the deltas is
byte-identical.
*Run-anywhere fixture:* one fixture placing and breaking a block on a planet profile and a ship profile,
**forcing a re-anchor mid-sequence** — the still-owed half of the existing re-anchor ledger item, and the
**first real G-IDENTICAL**, which the roadmap already commits at this phase.
*Roadmap scenarios §8.3 does not pick up:* the **epoch-mismatch fail-safe discard**, and
**edit-from-ghost-region forwarded-not-local-write**.

> ▣ **PLANT:** **prune-on-equality as the ONE universal delta rule, with the provenance comparison**
> (O6.3) — it must exist before the compactor is written. R8's **growth-stage byte** as a persisted
> side-table column, sparse, alongside the already-reserved per-chunk last-ticked stamp (D-BLK-2 reserves
> the stamp; nothing reserves the stage). O41's per-chunk surface-mark table **name**.

**P6.1b — Survey and assay.** *Size: S–M.* *Blocked by:* **O8, O9.**
*Lands:* the assay outcome on the block-edit acknowledgement path (**no new machinery**) and the survey
instrument on the block-edit carrier already built at S0.4 (**whose second consumer is already reserved**).
*Why here:* concealment without discovery removes an activity and returns nothing. **Ship both or ship
neither** — and if the answer is "not now", take only O8's three free reservations and skip this slice.

**P6.2 — Damage.** *Size: L.* *Lands:* the branchless integer gate-and-scale hit resolution; the four
outcomes and their drop rules **with O46's physics-source correction** (gate only tool-carrying sources, or
landings, explosions, fire and collapse destroy all matter they touch); the sparse per-chunk side tables
with their computed dense-promotion crossover and the loud per-world cap; the eight-stage crack overlay as
a change-driven chunk-scoped delta; the repair path as a separate subtraction so the damage path stays
monotone. *Done when:* the resolution benchmark holds, the promotion crossover is **measured rather than
assumed**, and the replication cost at the density baseline is measured.

**P6.3 — The shaped lane.** *Size: L.* *Lands:* plate, ramp, corner and facet families made placeable; the
resolve pass reading a two-cell window and writing a resolved-variant buffer the mesher **and the collider
both consume unchanged**; the six face masks (full and partial per side) rather than one occupancy mask;
**instancing, which is not an optimisation but the mechanism that makes this lane bounded** — without it an
ordinary decorative facade produces 11 MB of geometry per chunk against a half-megabyte budget; sub-chunk
mesh sections so a single edit re-meshes an eighth of a chunk; and the sparse-iteration path for constructs
occupying a small fraction of a chunk. *Done when:* the hard per-chunk ceiling holds; a 10,000-block
construct's full rebuild meets its budget on the sparse path; the adversarial checkerboard wall is inside
the merge-barrier bound.

**P6.4 — The build loop.** *Size: L.* *Blocked by:* **O39(k).**
*Lands:* face-and-octant variant inference; look-direction auto-rotation against the derived legal set;
neighbour-derived connective sub-shapes with the pinned escape hatch living in the sparse state table
rather than the orientation field; symmetry and mirror mode; **batched line and plane placement, which is
both the latency mitigation and the re-mesh amortiser** (four thousand edits, one re-mesh); server-side
undo. *Done when:* the shape-sheet scenario — every distinct rendered placement laid out in a labelled
grid, one screenshot — passes on both a planet and a ship profile, turning a broken template, a reversed
winding, a wrong mirror twin or an off-lattice vertex into a pixel diff.

**P6.5 — Occupancy-based ambient occlusion.** *Size: M.* *Blocked by:* **O24's AO half.**
*Lands:* the third occupancy mask uploaded as a sparse slab, **resident at tiers 0 and 1 and absent above**
(an ordinary rung of the one ladder, not a private radius carve-out); fragment-side evaluation that is
lane-agnostic by construction; and the exact rim sampling using the **unwarped** cell index carried in the
detail vertex, so a displaced position cannot index a neighbouring cell. Use **§4's exact cell-fraction
lattice** for the threshold — it is currently stated in two different volume units, one of which no longer
exists.

**P6.6 — Decoration tier two, and edit reactivity.** *Size: L.* *Blocked by:* **P4.7a, O47, O48.**
*Lands:* declared read neighbourhoods and the reflected dirty set, always recomputed from scratch; the
per-chunk decorated-cell table with span writes and a bulk-rebuild path on the task pool; the expiring
growth log and the unbounded-lifetime override set as **two named new persisted record types**; grow-in
driven by the client's interpolated authoritative tick; the prop tier; **and P4.7's deferred acceptance
scenario, which can now actually run** — place a soil block, place four hundred more, place a wood column
and leaf blocks, force a re-centring, assert the pre- and post-re-centring frames are pixel-identical.
*⚠ CORRECTED from §8.3:* it lands "leaf materials declaring they are not drawn as cubes, **with overhanging
clusters drawn instead**" — the derived-canopy model. **R14 makes canopy cells REAL CELLS** (stored,
authoritative, harvestable, holding the tree's ~120:1 canopy-to-trunk spread) **that carry an empty
collider.** The leaf render model is a flag bit on the shape row plus a foliage shape family, planted at
S0.2.
*Done when:* the edit-latency gate holds (a bounded number of write ranges and bytes per edit), the bulk
path triggers on a crater and never blocks the main thread, both new record types survive a hard kill and a
re-shard, **and the walking pop gate passes** — the most-reported complaint about the shipped title studied
is not tier crossings at all, it is clutter appearing a few metres in front of a walking player, so the
distance ramp must be the same wide ramp as the edit-reactivity ramp.

---

### P6b — world simulation *(pending O34)*

**P6b.1 — The shared propagation engine and the slow-tick scheduler.** *Size: L.* *Blocked by:* **O34, O35.**
*Lands:* one monomorphic attenuating flood over the six-neighbour graph with separate increase and decrease
queues and a per-tick cell budget that carries over, expressed as **data rows** for light, sky exposure,
structural support, heat, radiation and signal strength; the sampling scheduler whose cost is proportional
to loaded sections and independent of world content; and the boundary assertion that every attenuation
table reaches zero at a world boundary, **so a fire on a docked ship cannot become a distributed-systems
problem**. All of it is tier-0 machinery and never sees a coarse chunk.
*Done when:* the flood benchmark's ball sizes match the closed form **exactly** and the boundary assertion
is a harness test rather than a review item.

**P6b.2 — The twelve mechanics.** *Size: XL.* *Blocked by:* **O34, O35, O46.**
*⚠ CORRECTED from §8.3, which builds a retracted design in three independent ways.* §8.3 lands "structural
support **with the rule that generated blocks are unconditionally anchored and the check runs only within
thirty-two blocks of a player's edit**". That sentence is now wrong three times over: **R6 retracts the
radius exception outright** (terrain is unconditionally anchored everywhere, permanently, no radius, no
exception); **R10 removes the 32-block cap's role as the flood bound entirely** (axial-free,
lateral-limited; the flood is bounded by the existing carry-over cell budget) leaving only the lateral
cantilever role — **without which every tree, tower, mast and antenna in the game is capped at 32 blocks
tall**; and it reads the **one-bit** `placed` marker R6 widened to **two bits with three named values**, so
the slice would also write the wrong record. Likewise "falling blocks merged into single rigid bodies" is
superseded by **R6-4's server-authored SCRIPTED TOPPLE** — closed-form final pose, integer settle, **no
collider, no transfer envelope**, live and dormant paths writing identical bytes.
*Lands, corrected:* light, sky exposure, structural support (unconditional terrain anchoring; axial-free
lateral-limited flood; three-state provenance), scripted topple, fire, fluids, pressure and rooms, growth,
weathering, freezing, explosions and radiation; the three catch-up classes with their persisted inputs;
**per-chunk catch-up on first read** using the already-reserved last-ticked stamp (zero extra storage);
and the gravity capability on the shard profile with the support model selected by the capability **value**,
never by a shard kind — `SupportModel::{Anchored, GridRigid}` collapses into **one algorithm** whose load
term is proportional to the realm's declared gravity magnitude, degenerating **with no branch** into grid
rigidity at zero g. That is what makes "nothing about this exists on a ship" true without a special case —
and cutting a ship in half still separates the severed piece, which is the same rule with the weight set to
zero.
*Done when:* the four gravity-dependent mechanics pass their own identical fixture on **two
gravity-bearing profiles** — a spherical planet and a flat spin station — while the gravity-free subset
passes the ordinary planet-and-ship fixture. Four unbounded worst cases (explosions, collapse, mass
falling-block conversion, spall cascades) all sit behind deferred budgeted queues, **never inline in the
edit path**.
*One coupling to name:* every mechanic that changes a block writes through the edit path, so **every
mechanic also updates the pyramid** — free, because it is the same walk-up with the same early exit, and
which is **why the pyramid must exist before this phase rather than after it**.

---

### P7 — player checkpoints and durability *(restored; §8.3 skipped it)*

**P7.1 — Checkpoint coverage for the block system's persisted structures.** *Size: L.*
*Blocked by:* O34's ordering answer.
*Why it exists:* the block system creates **at least ten new persisted structures** the committed P7
definition of done must survive — `block_wal`, the chunk record's two encodings, the edit pyramid, the undo
journal, the sparse damage tables, `state_bits`, the growth log, the unbounded-lifetime override set, the
derived per-column tables (`column_sky` / `snow_depth` / `grass_cover`), plus `config_state` and grants at
P9. P7's DoD is *"kill -9 everything; every player resumes at their last checkpoint with correct
pose/inventory/health, exactly one Owner, no stuck saga, **all block-edit deltas present**."*
*The sharper collision nobody has written down:* P7's committed deliverable is
**"reconnect-without-game-state-replay (fresh snapshot + only the bulk the client lacks, via the chunk
cache keyed by `edit_epoch`)"**. With the ladder the client's resident set is per **(chunk, TIER)** and the
delta payload is tier-tagged — so **the reconnect cache key must gain the tier**, or reconnect either
re-sends the whole resident set or serves a tier-0 cache entry against a tier-5 request. S0.5 correctly
adds the tier to the residency index; **nothing adds it to the reconnect cache, because the reconnect cache
is in the phase the plan skipped.**
*Also:* P7's stated risk is "write amplification from naive checkpointing is real — mitigated by **tiered
cadence (required, not optional)**", and P6b.2's twelve mechanics writing per-block state at slow-tick
cadence are a **new write source that tiered cadence was never sized against.**

> ▣ **PLANT:** the tier on the reconnect chunk-cache key.

---

### P8 — ships

**P8.1 — Construction realms and the weld.** *Size: L.* *Blocked by:* **O26, O37, O28's hull half.**
*Lands:* minting a construction world from a crafted, quota'd anchor item rather than from an untethered
block placement; adjacency joining an existing construction rather than minting a third; the deliberate
weld as a one-time durable transfer of one world's whole edit set into another through the existing
transfer machinery; abandonment reaping as an ordinary owner-authority operation at spin-up, **never as a
background writer**. **A weld also merges two pyramids** — and that is *not* an entry-wise merge, which
could never produce a correct dominant substance or fill: it is **the same walk-up re-run over the merged
delta set**, which needs no new machinery. It must be in the weld's transaction, or the welded structure's
distant appearance is stale until something touches it.
*Done when:* a welded structure keeps every edit across a hard kill and a re-shard, and its pyramid
recomputed from the merged deltas is byte-identical to the one the weld produced.
*Roadmap scenarios owed here:* **ship-interior block-edit durability**, and the **concurrent-saga mutex**
(a ship transferring while a passenger EVAs is serialised).

**P8.1b — The dock, the blueprint manifest, and the payment brake. NEW.** *Size: L.* *Blocked by:* **O37.**
*Lands:* R18's mechanic — bring a blueprint to a dock; the dock reads it, works out the materials, sets a
price; you pay; it spins up a realm belonging to you and the ship is reconstructed with wiring intact —
**plus the connection manifest**, which lists every channel the design reaches outside the construct
alongside the bill. *Three consequences that need owners before this ships:* ship construction gains a
place, a price and a profession; **the payment is the anti-abuse brake on realm spin-up, which is a real
server cost that currently has no brake**; and author royalties require a durable author identity that
survives copying.
*Why it is a slice:* §8.2 already prices four separate doors as "a migration over every saved planet, ship,
station **and blueprint**" — the plan is charging for blueprints it has not scheduled.

**P8.2 — Ship-on-terrain contact.** *Size: M.* *Blocked by:* **O29's second half.**
*Lands:* **the planet owning the contact**, with the hull entering the planet's physics world as a ghost
collider through the existing machinery and the resulting pose shipped back as the ship world's authored
placement. Tier 0 on both sides.

**P8.3 — Kinematic articulation.** *Size: L.*
*Lands:* articulation **computed rather than solved** — each actuator's scalar state from its transfer
function, the child sub-assembly's pose as an exact analytic rigid offset, colliders moved as
kinematic-position bodies, reaction applied analytically; contacts disabled between a parent and its own
articulated child with the commanded motion refused by a discrete shape-cast instead; lock-implies-merge;
build-time cycle detection; and the cap of 512 actuators per assembly, **derived from the honest byte
arithmetic of the entity state blob rather than asserted**. *Done when:* an articulated machine crosses a
world boundary **as one entity** and reassembles identically.
*Also owed here:* SPIKE-10a (dual-frame interior physics), a committed P8 deliverable §8.3 does not name.

**P8.4 — The top rungs and the hand-off to the realm proxy.** *Size: M.*
*Lands:* the continuity check between the coarsest resident chunk tier and the realm proxy — the proxy
becomes the next rung rather than a different kind of thing, so it is the same crossfade with the same band
— and the demand-loop coupling that keeps the proxy resident as the guaranteed no-hole fallback while
coarse tiers stream in behind it. *Done when:* a warp approach shows a body growing continuously from a
dot, through the proxy, into the body's **face-covering rung** (`tier_depth − 1`, per body and derived —
never the literal "tier 7", which is the option O1 rejects) and down to tier 0, with the pop detector
silent at every join.

---

### P9 — functional blocks and signals

**P9.1 — The signal bus core.** *Size: XL.* *Blocked by:* **O15, O18, O30, O36.**
*Lands:* the interned channel table with double buffering and compiled subscriber lists; the two lanes
behind one message arm with a versioned envelope and per-body classification; the three routing planes
each with a bounded radius, and the inverted index at each parent; the channel declaration with its
**validated liveness invariant** (an exported latest-wins channel must declare a republish period at least
twice as fast as its consumer's deadline, or holding the throttle down cuts the engines); the interest set
whose lifetime is bound to the existing demand ledger rather than a second timer; tiered authentication
with one truncated tag per bundle per destination per tick; the bounded replay window with **eviction and
refusal, `max_replay_entries_per_peer = 64` and per-origin queues**; the retained-port table (D-SIG-2,
without which every latch, counter, integrator, battery and timer resets seconds after its world empties);
and the uniform one-tick propagation delay written down *before* any block behaviour is authored.
**Plus the fixes: Pass B must NEVER write `front[c]`; `Neighbourhood { hops: 2 }` cut to 1;
`remote_expanded` defined by the parent reporting fan-out width back on the existing reconcile.**
*Done when:* the load gate holds (ten thousand functional blocks at realistic churn inside the per-tick
delivery budget); "no listener, no traffic" and "no traffic without a heartbeat gap" both pass; the
interest-set and replay-window memory assertions hold. *Run-anywhere fixture:* a switch toggling a lamp
through two combinators, on a planet profile and a ship profile, **plus an anti-vacuity assertion that no
signal code reads the geometry capability** — and its twin, **no signal, functional block or behaviour may
read a chunk's tier** (a beacon must not stop blinking because you flew away).

**P9.2 — Authorisation, the shipping block set and arbitration.** *Size: XL.* *Blocked by:* **O30, O36.**
*⚠ CORRECTED from §8.3, which prices a whole subsystem as one clause.* §8.3 lands "the durable grants table
with the station-autopilot handshake" inside a slice about the shipping block set. **R15 requires: an
ISSUED write capability admitted once at bind time by the realm that owns the channel and never derived
from a name; four persisted tables (realm owner, principals, acl, grants with `parent: Option<GrantId>`, a
subset rule, subtree revocation and `key_epoch`); revocation as a RELIABLE DISCRETE ACTION, not a datagram
(a lost packet otherwise means the pilot believes he revoked and did not); spin-up RE-AUTHORISATION (the
~2 s reap otherwise undoes every revocation); and `on_block_removed` as a REGISTRY over config, port,
grants, ACL, key and relay queue.** R19 additionally re-orders the handshake to **verify → resolve →
display**, with the dialog composed by the **ship's** realm and the effect set re-resolved at mint.
*Also lands:* the twenty-seven shipping blocks against machinery this design defines — six kernels with
closed parameter sub-enums and **zero per-block behaviour functions**; arbitration as a per-port field with
its two validated cross-products; the probe budget with its deterministic round-robin deferral and its
declared sort key, so a sensor's contact list is ordered by **entity identifier** rather than by a physics
data structure's internal layout.
*Done when:* adding a block is one manifest row and zero behaviour code, verified by a loop over the static
table; the probe budget assertion holds; pilot-versus-autopilot preemption is a priority ordering with **no
branch anywhere asking "is this a pilot"** — **and the eleven permanent authorisation gates pass**, headed
by **G-NO-NAME-AUTHORITY**: for ANY name a stranger can type and ANY realm he does not own, publishing
produces **zero deliveries**, driven by an exhaustive `WriteEntryPoint` enum in the existing effect-class
idiom **so that adding a write path without a refusal case does not compile**. Twelve entry points
enumerated (unoccupied seat, non-resident session, valid MAC with no grant, wrong-channel grant, expired
grant, revoked grant, grant against a non-grantable channel, replay inside the window and past eviction,
stale fence after a re-home, config apply by a non-owner, docked/welded block publishing to an owner
channel, foreign sibling subscribing to a private key) — **plus the anti-vacuity half**: the same fixture
*with* a proper grant does drive the ship. Run on planet **and** ship profiles per HR4. Plus
G-REVOKE-ONE-TICK with its durability half (reap, respawn, still zero), G-BLUEPRINT-NO-WIDEN,
G-NO-AUTH-FROM-BUS, G-ARBITRATION-CEILING, G-RELAY-STORM, **G-RELAY-ABSENT** (the relay-absent twin of the
standing economy-absent gate), G-COUPLING-MODEL-CHANGE, G-COUPLING-ANYWHERE, G-BOUNDARY-ENERGY,
G-TWIST-REBASE.
*None of those eleven gates appears in §8.4. A gate that exists only in a document the slice plan does not
reference is not a gate.*

**P9.3 — The block cover.** *Size: L.* *Blocked by:* **O55, O53, S0.5.**
*Lands:* the cover as an optional per-face attachment in the parent block's configuration blob, **never a
block kind**; the closed widget registry (**with O55's band argument in `compose`'s signature even if it is
ignored**); server-side composition into a bounded draw-command list with static per-widget caps refused at
configuration time; one instanced world-space pass with one signed-distance glyph atlas baked at build time
and hashed in CI; the legibility-derived angular threshold; and the aggregate per-client byte budget filled
in descending angular size. *Done when:* five hundred covers, 128 sessions, inside the per-client byte
budget and the shard tick allowance, in one or two draw calls, **and a session receives no draw list for a
panel it is not authorised for.**

**P9.4 — Relay BLOCKS. REWRITTEN.** *Size: M.* *Blocked by:* **O18, O39(m).**
*⚠ §8.3's P9.4 builds the design R17 deleted.* It lands "rendezvous hashing over the fence-versioned relay
set with the dual-read migration window; light-lag held at the routing node in a hierarchical timing wheel;
the refusal-with-hysteresis…; the offline mailbox" — **an engine-served relay plane**. R17 rules relays are
**player-built functional blocks with time as the cost**, superseding it, with an operator-owned realm
running relay *blocks* available at any time as the fallback. So the engine relay set, its
membership-change protocol, its dual-read window and D-SIG-3's "untested at galaxy scale" ledger row are
all building a deleted design.
*Lands instead:* a relay block row in the shipping set; the message header already planted at S0.4 (hop
count, arrival deadline, dedup identity); a `power` input port **declared but ungated** (adding the gate
later is data, not a redesign; gating immediately couples the comms plane to a power grid that does not
exist); and O39(m)'s eight answers.
*The size changes materially:* **L for a distributed relay plane; M for a block row plus a header.**

---

### P10 and P11 — what they inherit

Neither needs slices now; both need a line saying what carries forward, or the accumulated-suite rule
silently grows two phases of unbudgeted work.

- **P10 (system/galaxy warp)** is where the coarsest rungs are actually exercised. The committed demo is
  *"the system shrinks to a dot; a new system shard spins up on demand; they arrive seamlessly"*, and the
  owner's standing warp rule is *"shrinks to a dot on departure, grows from a dot on arrival"*. **P8.4
  lands the proxy hand-off acceptance test**, so the plan currently puts the *warp* test in the *ships*
  phase and has nothing at P10 — where provisioning latency, residency under a 100 km/s-class approach and
  the demand loop's spin-up-ahead interact with tier residency for the first time. *Inherits:* the tier
  ladder, the residency governor, the proxy-as-coarsest-rung join, and the ordering constraint
  coarse-before-fine.
- **P11 (combat)** consumes P6.2's damage channels plus projectiles. **Addendum 2's binding exclusion list
  — "anything that can shoot or be shot never gets a detail level" — is a P11 constraint asserted at P4
  with no P11 slice to enforce it.** *Inherits:* that exclusion as a permanent gate; O39(i)'s energy-based
  collision model; the server-side rewind question, which is a separate mechanism owed for combat
  regardless of anything decided here.

---

## §5 The critical path

**The shortest chain that unblocks everything else**, with the number of slices each link releases:

```
O1 ─┬─ O2 ─┬─ O3 ──┬────────────────────────────────────────────────────────┐
    │       │       │                                                        │
    └───────┴───────┴─→ S0.1 (geometry seam) ─→ P4.1 (generator) ─→ P4.2 (mesher)
                              │                       │                    │
O5 ─┬─ O6 ─┬─ O4 ──→ S0.2 (registry) ────────────────┘                    │
    │       │                 │                                            │
    └───────┴──→ S0.0b (persistence seam) ─→ P4.8 (store) ─→ P6.1 (edits + pyramid)
                                                                            │
O12/13/14/15 ──→ S0.4 (wire plant) ─→ everything after P4                  │
                                                                            │
                              P5.1 (collider spike) ─→ P5.2 ─→ P5.3 ────────┘
                                                                            │
                                                            → a walkable, buildable planet
```

**Nine answers unblock 26 of the 42 slices**: O1, O2, O3 (the geometry chain), O4, O5, O6 (the format
chain), O7 (the generator), and O12–O15 (the wire plant, which are all *scheduling* rather than design,
because R19 already ruled the substance). Everything else in §2 blocks between zero and three slices.

### The three things to do first

**One. Start S0.6 today, while you answer the register.** It is the only slice in the plan blocked on
nothing at all — every other Phase-0 slice waits on a row above. It is also the **only slice that gets
strictly more expensive with every other slice**, because the plan adds ~67,000 production lines and a
large fraction of the sim half lands in the file it is splitting: a verified **15,550 lines, 60% of the
whole `vd-sim` crate**, already working around an engine arity limit twice. §8.7's "if you do only three
things next" does not mention it at all. **This is the one item that converts your thinking time into
progress.**

**Two. Freeze the formats in ONE session: O1, O2, O3, O4, O5, O6.** They are one dependent chain sharing
one deadline, because all six are written into the first world that is ever saved and none can be changed
afterwards without a migration over every planet, ship, station and blueprint that has ever been edited.
O3 rides along with O1 and takes minutes once O1 is settled. **The two most likely to be underestimated are
O4 and O6** — O4 because it *looks* like an implementation detail and is not (it decides whether a player's
dug tunnel still exists when they look back at it from a hilltop), and O6 because it is **seven separate
permanent format freezes with no door row between them**, three of which are worth 244×, 1.84× and
"bounded between 461 MB and 7.7 GB by an unwritten policy" respectively. Rows 4, 5 and 7 of the old
register no longer belong in this session: two became per-body data and one is answered.

**Three. Land the plant slices — and add the four items that were missing from them.** S0.1, S0.2, S0.3
and S0.4 before a single line of generator or mesher code, unchanged in substance. What is new:
- **the persistence seam (S0.0b)** — verified missing, and two of this design's own published gates are
  literally unexpressible through the trait as it stands;
- **the scoped channel key in S0.4's first commit**, not P9 — the recorded deadline is wrong by five phase
  groups and there is verifiably no `ChannelKey` in the workspace today;
- **the four saga step-id constants in one file** — four `const` lines against silent, durable,
  unrecoverable journal corruption;
- **the arm-ceiling ruling in prose** — 27 + 3 = 30 is exactly door 38's review ceiling and there is no
  margin at all.

*And one measurement that is not a decision:* the mesher's per-tier cost curve, sustained columns per
second through generate + mesh + **upload** at flight speed, and the resident-chunk count of a 100 km view
**while moving**. The noise benchmark closed generation as a risk; it did not close the frame. Those three
numbers are what actually derive door 42, none is hard to get, and none exists.

---

## §6 Live defects in shipped code

These are bugs, not plans. They are costing something today, and they are the only items in this document
where the work is already justified.

**D1 — The interpolation smoothing never runs, so the buffer has never been paid.**
Reproducing the exact client arithmetic over a second of play: out of 121 drawn frames the smoothing step
ran **zero** times — the sampled pose equals the previous sample on every frame. **Every hour of in-game
validation has run at ~40 ms of client lag, not 120.** *Falsify it first, in one throwaway test:* build a
two-sample track at one-tick spacing, drive the render clock with consecutive ticks, and assert the sampled
pose equals the previous sample on every frame. The finding is exact arithmetic over two short functions
read in full and reproduces under both tick rates and under 30% loss, **but it was derived from a Python
replication, not from a test against the real crates.**

**D2 — Motion is slaved to the command rate: input is applied at ~40% of configured speed and stops on packet loss.**
There is no held-input resource and no per-tick integrate, so movement is applied only when a command
arrives. Today that is a 40%-speed bug on an avatar. **At P8 it is "my engines cut out when the network
hiccups", and it will be diagnosed as a netcode-feel problem rather than as a missing resource.** It also
blocks any future own-craft free-flight option, which is *unsafe* without it.

**D3 — The snapshot rate: the code and every document disagree.**
Verified: the shipped tick rate is **`tick_hz: 50`**, and a code comment in the sim shard itself still
describes the pose stream as "the 20 Hz latest-wins" while every design document says 20 Hz. The client
*learns* the rate from the wire, so a 120 ms buffer becomes **six ticks at 50 Hz, not 2.4**. **Settle this
before D1**, because it changes the correct buffer depth by 2.5× and every bandwidth percentage in the
decoration, marks and damage analysis by the same factor.

**D4 — The buffer depth has no external override and no per-context value.**
*One correction to the analysis that produced these findings:* it states the buffer is a "compile-time
constant". **It is not** — `ClientInterpTuning` is a proper config struct with an `interp_buffer_ms` field,
exactly as the no-magic-numbers rule requires. What is true is narrower and still worth fixing: the only
production constructor passes `::DEFAULT`, there is no external override, and the value is **not per
context** — at 30 m/s the buffer is 3.0–4.5 m of lag against 0.50–0.75 m walking, so driving at 100 ms
while walking runs at 150 ms recovers 1.5 m at zero architectural cost.

> **The trap, and it is the reason S0.7 exists as one slice.** If someone fixes the interpolator without
> also resizing the buffer, the game suddenly starts paying the eighth-second delay it has **never** paid,
> and the total goes from ~167 ms to ~257 ms — roughly **55% worse**. It will feel awful, **and the
> no-prediction law will be blamed for it.** Fix all three together: smooth the motion, set the buffer to
> two or three steps (40–60 ms) instead of six, and raise the command rate from 20 to 50 per second. The
> total then lands at about a sixth of a second — the same as today — while the motion becomes properly
> smooth for the first time.

**The honest caveat that no arithmetic settles.** The machine loop at ~167 ms is roughly the size of human
reaction time and therefore roughly **doubles** the pilot's effective reaction. The flying-qualities
literature puts ~100 ms of added transport delay at the onset of degraded compensatory control and ~250 ms
at forced move-and-wait with pilot-induced oscillation likely. At 162–247 ms this game sits **on that
boundary for precision tasks** — worst at docking and landing, not at speed. Only flying it will settle it,
which the capture harness can already script. And note the direction is the opposite of the intuition:
latency hurts when you are being **careful**, not when you are being fast — at 10 m/s the delay is a *third*
of your turning circle; at 500 m/s it is under **one percent**, because the mountain and your stopping
distance scale together.

---

## §7 Cross-document contradictions

Each with an adjudication and the document that must change. **Ten of these would be cleared by one
mechanical pass**: walk the Supersedes column of R1, R3, R5, R6, R10, R11, R15 and R17 through the body
sections the way R19 was already walked through §7 — **struck in place, not deleted**, so a reader who
remembers the dispute learns it was settled and by whom.

### The stale text that still argues a settled question

| # | Where | What it says | Adjudication |
|---|---|---|---|
| **S1** | `block_system_design.md` **§2.5.2, lines 1778–1782** | *"Generator-authored blocks (`placed = false`) are unconditionally anchored. The support flood runs only in the taxicab ball of radius MAX_SUPPORT_RADIUS around a player edit, and within it a block loses its anchor if it lost a generator-authored neighbour."* | **The single highest-risk piece of stale text in the set.** It is written as an *implementable rule* and it is wrong in **two independent ways**: R6 retracts the radius exception outright, and it reads the one-bit `placed` marker R6 widened to two bits with three named values. An implementer following it collapses cave roofs and writes the wrong record **in the same commit**. Strike in place |
| **S2** | **§7.5, line 13519** | *"in-realm traffic is implicitly trusted because the realm's shard IS the authority for everything in it"* | **This is the exact sentence that produced the owner's takeover attack**, and R15's Supersedes column names it — but unlike R19's two amendments it has **not** been struck. Replace with the issued-capability model. The amendment style is already established; use it |
| **S3** | **§2.7.1 (~2013–2030, 2481)**, plus **lines 626, 4400–4442, 5289–5301, 9925–9928, 10160, 15077** | The five-byte record diagram, "exactly five bytes", and six further sites asserting the width is open or five | All superseded by R1. Several are inside adjudication notes that were *correct when written*; strike rather than delete |
| **S4** | **§2, lines 2331–2345** | `[USER DECISION 2-A]` still posed as an open question with a one-way-door note | Answered by R2 and R3. Strike and point at the rulings table, exactly as 2-H, 2-I, 5-H and 5-I were |
| **S5** | **lines 867, 1743, 1916** | `MAX_SUPPORT_RADIUS` described as the support flood's bound in three places | R10 removes that role. Left as written, an implementer reinstates a 32-block cap on every tower, tree, mast and antenna in the game |
| **S6** | **§3.8, lines 4323–4342**, and §8.1 row 9 | Option (C) as "concealed and **revealed by exposure or by a survey action**", costing "one `MaterialClass` field and the generator split" | Superseded by R11 and concealed C-2/C-4: the concealed half is **never revealed at any range and never enters the block grid**. The exposure-reveal lane is exactly the design whose cost model was refuted (54–2,171 cores) |
| **S7** | **§6.6, lines 11331–11355** | *"Anything that must collide, **be harvested**, be stood on, or be seen by a sensor is PROMOTED to a block."* | R13/R14 **split those clauses**. As written it licenses giving every harvestable bush and every leaf a collider — the exact behaviour the owner ruled against. Restate: **harvestable ⇒ CELL; collidable is a separate property of the cell's SHAPE, and a shape may carry an empty collider** |
| **S8** | **`docs/design/slice_3_renderer.md`, lines 47–51** — *a committed repo doc* | *"The wire already anticipated this — `intershard.rs` reserves the WHAT lane 'at the LOD the observed realm controls' plus a `coarsen_level` ladder."* | **False**, and worse than the scratch-file instances because an implementer will trust it: `coarsen_level` is a **pose**-precision hop counter whose own doc comment says so, and line 303 reserves the WHAT lane as a *concept*. **There is no terrain-tier field anywhere in the wire today.** *Good news at the same site, verified: the "LOD (never)" retraction and the `realm_scene.rs:582` "NO LOD" retraction are both genuinely landed* |
| **S9** | **Addendum 1 §C.4 and §E** | *"the binding constraint is the drawn radius… roughly 700 m to 1 km of fully meshed terrain is what one-metre blocks with no detail tiers can afford"*; row 7 recorded as "PROMOTED to the top", row 21 as "PROMOTED to blocking" | Correct **conditional on a clause that has been retracted** — "with no detail tiers". Restate rather than delete: the **resident chunk budget** is capped, the drawn radius is not; the flight table (79,277 chunks at 100 m, 7.9 million at 10 km) becomes ~5,000 and ~6,000. §8.6 item 7 orders this; the addendum file itself is unamended |
| **S10** | **Addendum 1 §B / §B.4** | *"The short answer: yes for what it holds… So register row 2 is really: do you accept those two verdicts?"* | Answered directly by R1. **§B.3's substantive content survives intact and should be kept**: biomes are a property of a PLACE and never written into a saved record; functional-block state lives in side tables; and "no spare bits is a discipline" survives as the mechanical guard — reserved bits rejected on decode, spending one a reviewed change, **as R6 demonstrates** |
| **S11** | **Addendum 2 §A, §C.1, §C.4** | The `coarsen_level` wire claim; "at most eight small writes without reading anything from disk"; "a thousand-block tunnel produces roughly 1,140 pyramid entries"; and "tier 7 to 100 km" reading as the top rung | All four corrected: see S8; **≤1 write and ≤1 storage-block read per rung**; **504 entries ≈ 3.9 KiB**; and the hand-off rung is per body and derived, never a literal (literal rung 7 is option G3, which O1 rejects). **Plus a missing row:** planetary scatter costs `log₄(A/n) + ⅓` = **6.07 entries per edit at 100 M edits**, 5.5× the plan and enough to exceed the planet budget threefold |

### The genuine contradictions, adjudicated

**G1 — Register row 21 (the vertex format) was de-escalated, then re-escalated by three documents.**
Adjudicated in **O27**: the de-escalation is correct for the narrow claim it makes (whether a planet can
exist *on foot*) and **wrong as a ranking**. It is a mesh-encoder and upload-path change, not a data
migration — so it is a **scheduling** decision, and it currently carries three mutually exclusive
deadlines. *Changes:* §8.1 row 21's status line, and one of the three deadlines has to go.

**G2 — R13/R14 create the exact object `collidable_decoration.md` forbids, twice and emphatically.**
That document's §3 note 3 says: *"A cell that exists for storage but not for physics is precisely the cell
the composite rule forbids, read in the other direction. **There is no such object and there must never be
one.**"* Repeated at §14.1 item 2 as an outright refusal. R13 and R14 create exactly that object.
*Adjudication:* the rulings table wins by its own header, and R13/R14 postdate the document and name it in
their Supersedes column — **but the reconciliation is narrow and must be written rather than left implied.**
The document's refusal was aimed at **sub-catalogue** objects (a blade of grass, a pebble) being given
storage they cannot justify; it is **not** a general ban on a catalogue shape carrying an empty collider,
and Minecraft's tall grass is the precedent R13 itself cites. **Restate as: "no SUB-CATALOGUE object may be
a cell", and add a PASS-THROUGH shape class to the catalogue.** R14's deferred pass is the right home for
the detail — and **it must not be allowed to re-derive the ban.**

**G3 — `docs/design/roadmap.json` commits two facts that are now wrong.**
(1) P6 deliverables: *"`block_wal` (append-only, **5-byte** dedup-to-final edits…)"* — R1 fixes the record
at **eight**. (2) P4: *"`PlanetParams::from_seed` (**radius 100k–200k blocks**…)"* — the ladder makes radius
per-body data with any real body inside 20 m, and **Earth is 6.371 MILLION blocks, thirty-two times outside
the committed band.** *Adjudication:* the band survives **only** as a constraint on the curated *starter*
world (O42 owns it) and must be restated as such. **Both are one-line edits today and definition-of-done
disputes at the exact phases they gate.** Two further P4 deliverables are contradicted by recommendations
in §2 and must be reconciled deliberately rather than drifted past: *"pinned `noise`"* against O7's vendor
recommendation, and *"the binary-greedy-meshing adapter"* against O39(a)'s write-our-own.

**G4 — The reserved-bit count is fifteen in two places and fourteen in two others.**
R1 and §8.6 item 1 say **fifteen**; §3.9.2 publishes `cell 18 | block state 32 | reserved 14` and always
has; `block_provenance_collapse.md` §3.1 says the reserve goes 15 → 14 when R6 spends the provenance bit.
*Adjudication:* **fourteen is correct as of R6**, and the reconciliation is already supplied — before R6
the declared fields filled only **thirty-one** of the state word's thirty-two bits, which is exactly why R1
counted fifteen; spending the provenance bit makes the state word exactly thirty-two and the reserved run
exactly fourteen. **So R6 resolves a one-bit discrepancy that had existed since the record was first
written.** *Changes:* amend R1's own text and §8.6 item 1's synthesis line to fourteen, citing R6 — or
somebody implements fifteen and then discovers provenance has nowhere to live.

**G5 — Register row 30 (is radio interceptable) answers only READ.** Adjudicated in **O39(e)**: the
recommendation stands and the **scope** does not. Row 30 is not closed, it is incomplete, and the asymmetry
it already records applies to the publishing half too. *Changes:* the row's wording, before the first radio
slice.

**G6 — Rows 34 and 35 are re-taken and re-sourced by the signal ruling.** Row 34's recommendation survives
but its *reasoning* changes materially — the in-realm grant **tree** gives delegation and subtree revocation
with no new dependency, and the "one reserved optional field" rationale must be restated against the tree
plus four persisted tables. Row 35 is **not closed** and has acquired a sibling: the handshake asks to
encrypt to an Ed25519 key, which is not an operation Ed25519 has. See **O30** and **O36**.

**G7 — Register row 16 gains an argument that was not in its table.** Per-cell friction. Adjudicated in
**O28**: the row is not closed, its options table is incomplete, and **the P5.1 spike's scope must be
widened before it runs, or the spike answers the wrong question.** Two other movements touch it in opposite
directions: the scripted topple *removes* an argument for the sparse voxel shape, and the hull-collider
question *adds* one — with a confidentiality edge, because a coarse voxel occupancy hands your ship's shape
**including internal voids** to whoever hosts you.

**G8 — Rows 10 and 32 are answered in substance and stale in wording.** Row 10's option (B) is eliminated
**permanently** by R12 on a reason that "will not change with better hardware", and (C)'s escape hatch is
now *required* rather than reserved; what remains is where the evaluator lives. Row 32's **irreversible
half is closed** — the evaluator's placement and the server-side sampling path are committed and paid for.
*Changes:* restate both, or the owner is asked to re-decide something three documents have settled. See
**O49** and **O47**.

**G9 — P6b is inserted where the roadmap has P7, and the two are never mapped.**
Register row 13's own "Blocks" column says *"…and **what P7 can checkpoint**"* — so the register knows the
dependency and **the slice plan loses it**, dropping P7 from the sequence entirely. The coupling is real
and directional: P6b.2's twelve mechanics write per-block state whose *shape* P7 must checkpoint, and P7's
own stated risk is write amplification "mitigated by tiered cadence (required, not optional)" — a cadence
never sized against twelve mechanics writing at slow-tick rate. *Adjudication needed:* either P6b lands
**after** P7 and P7's checkpoint covers only the P6 tables, **or** P7's definition of done expands to cover
the P6b tables and P6b's side-table shapes freeze before P7 starts. **Silence picks the worse one by
default.**

**G10 — The status of the documents themselves.** The brief this run was written against calls all ten
binding; the folder's own README calls them **not binding — investigation inputs to be re-validated**.
*Adjudication:* the README is the later statement and it is a committed repo file, so it wins. This board
is a decision register over investigation inputs; a row becomes real when it is promoted into
`docs/design/`. **That does not weaken §3** — a one-way door is one-way whether or not the document
describing it is binding.

**G11 — Two small contradictions worth a line each.** The prop-radius baseline is stated as 80 m in row 39
and as "~33 m for a bush, 114 m for a boulder" in `[6-8]`, which changes what "3× further" means (see
**O23**). And **`DroppedBlock` versus the scripted topple**: `PLAN.md` HR2 commits a continuity model for
this matter (*"dropped block → durable two-step LandBlock conversion keyed by `entity_id`, **acked against
`block_wal` before despawn**"*) and `entity_kind.rs` already registers `DroppedBlock` as tag 11, Transient,
RealmAnchored — while R6-4 rules a falling group has **no transfer envelope** and `collidable_decoration.md`
A2 rules that during a topple a composite **is** an entity. **Neither sentence names the other**, and the
unanswered case is what happens when a toppling group crosses a realm boundary mid-fall or the realm is
reaped mid-topple. Small, but precisely the class of thing that becomes an unreproducible bug at a realm
edge.

---

## §8 Scale reality

### The arithmetic

Using the design's own size bands at their midpoints:

| Phase group | Slices | Production lines (est.) |
|---|---|---|
| Phase 0 (remaining, including the four new slices) | 8 | ~13,500 |
| P4a + P4b | 12 | ~19,500 |
| P5 | 3 | ~3,200 |
| P6 | 7 | ~13,300 |
| P6b | 2 | ~6,000 |
| P7 | 1 | ~2,000 |
| P8 | 5 | ~8,600 |
| P9 | 4 | ~11,000 |
| **Total** | **42** | **≈ 77,000** |

**The baseline, verified today: 115,284 Rust lines across the `crates/` tree, built since June 2026.**
(The adversary's figure of 122,913 counts the workspace-level `tests/` tree as well; the difference does
not change the conclusion.) So **the block-plus-signal plan alone is roughly two thirds of everything that
exists** — and at HR5's 100% region+branch across **nine Tier-A crates**, test volume historically runs at
or above production volume. **The honest total is 140,000–170,000 lines.**

**And that excludes**, all of it named above: P10 and P11 entirely; the survey/assay loop if O9 says ship
it; R4's composite subsystem, which is *undesigned*; the atmosphere, sky, water, weather and surface-aging
line — **`stunning_look_plan.md` §4 is titled "why the whole prize is here" and has no slice at all**; the
economy overlay; the dormant-world substrate; and the k3d packaging already owed.

### The verdict, plainly

**The plan as scoped is not achievable for one developer with AI assistance inside any horizon you would
accept, and saying so is the useful thing to say.** The load-bearing half is, and it is a coherent product
on its own.

### What is load-bearing — cutting any of these breaks something structural

| Slice | Why it cannot be cut |
|---|---|
| **S0.1** the geometry seam | Every line written before it is written against an unknown interface |
| **S0.2** the registry with the full catalogue declared | Identity numbering is append-only-never-reused; declaring it in full is what makes the character-controller derivation well-defined from the first commit |
| **S0.0b** the persistence seam | Two published gates are literally unexpressible without it, and P4.8 cannot be tested on a twin |
| **S0.4** the wire plant | The scoped key, the four step-id consts, the two tier bytes and the arm-ceiling ruling. Each is silent-and-durable if missed |
| **S0.5** the AoI reshape | A **precondition** of block edits, not a follow-on — one index, five consumers |
| **S0.6** the file split | Only gets more expensive; blocked on nothing |
| **P4.1 / P4.2** generator + mesher | The world |
| **P4.8 + P6.1** store, edits, pyramid | The pyramid is "the whole of the ladder's real work" and decides whether a player's dug tunnel exists when they look back from a hilltop |
| **P4.10** the ladder's client half | Without it there is no flight, no warp visual and no residency governor |
| **P5.1–P5.3** physics | Standing on the ground |
| **P9.1 + P9.2** the signal bus and its authorisation | The bus without the authorisation ships the owner's own attack |

That is **~34,000–38,000 production lines** and it is a walkable, buildable, persistent planet with wired
blocks — which is the thing worth having.

### What is gold-plating — every item is additive later and none shuts a one-way door

| Slice | Size | Why it can wait |
|---|---|---|
| **P4.4** material seam strips | M | Row 22 prices the decision behind it at **~1 month of shader work**, and the ladder **drops the seam blend at tier ≥ 2 anyway**. Hard edges look fine, and the trimming machinery it shares is needed regardless |
| **P4.5** the crumbling edge | M + a 10,000-configuration golden table + a 256-case containment proof + a merge-ratio regression | Its entire value is that dirt does not look Minecraft-square — and **its own tier rule sets the amplitude to exactly zero at tier ≥ 2** |
| **P4.6** transparency beyond water | M | Additive |
| **P6.6** decoration tier two and edit reactivity | L + **two new persisted record types** | The reactivity scenario is the payoff, but the payoff is not the product |
| **P9.3** the block cover | L + a closed widget registry + a baked glyph atlas hashed in CI + server-side draw-list composition + a per-client byte budget | **This is one of your own explicit asks, so present it as a DEFERRAL, not a deletion** — but it is the largest P9 item that nothing else depends on |

Together **~6,000–7,000 lines and a dozen gates**, and none of them shuts a door. **Take O55's
one-parameter reservation and P4.7a's evaluator placement even if you cut everything else in this table** —
those two are the parts that are one-way.

### Already correctly refused, and worth keeping refused

Virtual texturing; the analytic displaced-sphere shell; columnar block storage; shell-only storage; SVDAG;
1-D run encoding; hardware ray tracing; multisampling; screen-space reflections; planar reflections;
clustered decals; a spectral ocean; billboard clouds; a particle library now; voice and video transport;
per-cell rounded rock templates. Each is closed on a **verified engine or hardware fact**, not on taste —
which is the property that stops them coming back.

### Converging or accreting

**The design is converging; the delivery artefact fell behind it, and the gap is countable.**

*Converging, and not superficially* — each of the nineteen rulings **replaced a fork with a parameter**:
one ladder instead of three private carve-outs; one record instead of three widths; one authorisation model
instead of three assertions; one grid from a block in your hand to a planet seen from another star; one
universal delta prune rule; one support algorithm whose gravity term degenerates to grid rigidity at zero g
**with no branch**. Two risks closed by **measurement rather than argument**. The register deduplicated 40
raised decisions to 36 by three principled merges, then collapsed seven more markers into three rows — a
design that keeps discovering it asked the same question twice is a design whose parts are meeting. And it
now contains real refusals with facts behind them, which is the clearest single sign of convergence.

*Accreting, and it is all in the plan rather than the design* — five doors missing from a table that claims
completeness; one door stated **five phase groups** late; one door stated later than its own slice; three
slices building **retracted** designs; four answered rows still ranked in the visible top 26; eleven
permanent gates from one binding document appearing in §8.4 **zero** times; seven decisions living only in
body text; and **zero of 34 slices naming the decisions that block them**.

**None of that is an intellectual problem.** The corrective is one mechanical pass over six Supersedes
columns plus the merge this document performs. **What it will not clear is scale** — and no amount of
reconciliation makes the plan smaller. That needs a cut, and the cut is: P4 split around P5, and the five
rows in the gold-plating table above.






