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

# Addendum 2 — Detail levels: the ban is retracted, and the smart way to build them

**Date:** 2026-08-03. **Owner ruling:** *"from where do you get no LOD? Please remove it, we will need
LODs everywhere, but in the smartest way."*
**Amends:** decision register row 7 (now **ANSWERED**), row 21, §5.9, §6.2, and Addendum 1 §C.4.
**Also retracts:** `docs/design/slice_3_renderer.md:45` ("LOD (never)") and the "NO LOD" comment at
`crates/client/src/realm_scene.rs:546` — both updated in place.

---

## A. Where the ban came from

Four places, so the record is clean:

1. `docs/design/slice_3_renderer.md:45` — "LOD (never)", written into a landed design doc.
2. `crates/client/src/realm_scene.rs:546` — a "NO LOD" code comment on the realm-proxy tessellation.
3. A standing preference of yours I had recorded as *"no level-of-detail meshes for now"*.
4. A second recorded preference — the "binary render rule": *in range, proper mesh; out of range,
   nothing; no spheres, no proxies*.

Two things are worth saying about that. First, the earlier rendering analysis had already flagged that
the exact phrase "binary render rule" appears nowhere in the repository — it was always a preference,
never a written rule, and it noted that the repo *contains both sides of the argument*. Second, the
whole-goal audit of 2026-07-05 called this out as a HIGH finding almost a month ago: *"never-LOD +
binary render rule collide with the P4 demo itself… under the rules as written the planet either ends
in a visible wall or renders as nothing mid-'seamless' transition."* It asked for exactly this ruling
and declined to make it unilaterally. So this is a tension the project has carried for a month, now
resolved in the direction the arithmetic was already pointing.

**And the wire anticipated you.** `crates/wire/src/intershard.rs:303` already reserves the WHAT lane for
*"terrain/constructions at the LOD the observed realm controls"*, and line 656 already carries a
`coarsen_level: u8` precision ladder, always 0 today. The architecture was built expecting this. Nothing
about adding detail levels is a retrofit of the transport.

All four statements are now corrected. **The new standing rule is: detail levels everywhere, on one
generic ladder, with a small list of things that must never have one.**

---

## B. Why this project's LOD is unusually cheap — the two facts that decide the design

Most voxel games find level-of-detail brutal because they must *stream* coarse data: the server holds
the world, so every detail tier is more bytes over the wire and more storage. Two facts make our
situation different, and the whole design follows from them.

**Fact one: terrain is generated, not stored.** The client has the generator and the seed. It can
produce any tier locally, at any moment, for free. Only *player edits* have to travel.

**Fact two — and this is the good one: for fractal terrain, the correct level of detail is not an
approximation. It is the exact answer, and it is cheaper.**

The surface is a sum of octaves, each with half the feature size and half the amplitude of the one
before. A cell of size *s* physically cannot represent detail finer than about 2*s* — sampling it
produces aliasing, which is the shimmer everyone associates with distant terrain. So the correct thing
to do at a coarse tier is to **stop summing the octaves you cannot see**:

> **Tier L drops the L highest-frequency octaves.**

This is simultaneously the right anti-aliasing filter, the right level of detail, and a saving. The
height error it introduces is bounded by the sum of the dropped amplitudes, which is proportional to the
cell size — meaning the error always scales with the representation, which is exactly the property a
detail ladder is supposed to have. And a tier-3 chunk costs three octaves fewer to generate than a
tier-0 one.

The same argument covers the three-dimensional cave field: caves smaller than a coarse cell disappear at
that tier, which is correct — you cannot see a three-metre cave from ten kilometres.

**So the generated part of the world gets a detail ladder for free, with no stored pyramid, no
precomputation, and no aliasing.** That is the smart way, and it is specific to fractal terrain. The
work is entirely in the other half.

---

## C. The design

### C.1 The ladder

One rung is one power of two. At tier *L*, a cell is 2^L metres and a chunk is still 62³ cells, so it
covers 62·2^L metres. The chunk container, the palette, the mesher, the material tables and the shape
catalogue are **identical at every tier** — a tier is a number carried by a chunk address, not a second
code path. Hard rule three is satisfied by construction: nothing branches on tier.

Tier 0 is one-metre cells. Tier 7 is 128-metre cells. Above the top rung sits the realm's own proxy —
which already exists — so the ladder runs continuously from a block in your hand to a planet seen as a
dot from another star. **The proxy stops being an exception to a rule and becomes the coarsest rung.**

### C.2 Choosing the tier — reuse the rule that already exists, do not add a second

The locked area-of-interest model is already *"angular size above a threshold, one generic rule for
every realm kind, never branching on type."* Tier selection is the same rule applied one level down:

> **Pick the tier whose cells subtend a fixed number of pixels on screen.**

That is screen-space error, the standard correct metric, and it composes with the existing demand loop
instead of introducing a second scheduler. With a target of two pixels per cell at a 70° field of view
across 1920 pixels, one screen pixel is 6.36 × 10⁻⁴ radians, so tier *L* is used out to
`786 × 2^L` metres. Tier 0 to 786 m, tier 1 to 1.6 km, tier 3 to 6.3 km, tier 7 to 100 km.

The pixel target is a field in the render tuning struct, never a literal — it is the one knob that
trades sharpness against everything else, and it must be tunable at runtime for the tuning overlay.

### C.3 What this buys, in numbers

Each tier occupies an annulus between its own range and the previous tier's. The arithmetic falls out
constant: **every tier above the first holds about 379 chunk columns, regardless of which tier it is** —
the annulus grows by four and the chunk footprint grows by four. Tier 0 is a disc and holds about 505.

| View distance | Without tiers (chunk columns) | With tiers | Tiers needed |
|---|---|---|---|
| 786 m | 505 | 505 | 1 |
| 1.6 km | 2,019 | 884 | 2 |
| 6.3 km | 32,300 | 1,642 | 4 |
| 25 km | 517,000 | 2,400 | 6 |
| 100 km | 8,170,000 | 3,158 | 8 |

Taking three vertical chunks at tier 0 and two above, a **100 km view costs about 6,800 chunks** — call
it 270 MB of mesh at the packed vertex format, and 6 s of generation on one core or 0.75 s on eight,
before counting the octave saving. Without tiers the same view is 24.5 million chunks.

The shape of that table is the point. **Visible chunk count now grows with the logarithm of view
distance instead of its square.** Addendum 1's central finding — that the drawn radius, not the planet
radius, was the binding constraint — is thereby dissolved. Both constraints are gone: any planet size,
any view distance.

| | Before (Addendum 1) | After |
|---|---|---|
| Ground, 162 km planet | 1,348 chunks | unchanged (one tier is enough) |
| Ground, Earth-sized | 53,110 chunks, 2.1 GB | ~4,000 chunks, ~160 MB |
| Hover at 100 m | 79,277 chunks, 3.2 GB | ~5,000 chunks, ~200 MB |
| 10 km altitude | 7.9 million chunks | ~6,000 chunks |
| Orbit | impossible | the realm proxy, as today |

Row 21 — the vertex format — accordingly **drops back out of the blocking set**. It remains the right
choice for bandwidth and upload latency, but it no longer decides whether a planet can exist.

### C.4 The hard part: player edits must survive coarsening

This is the whole of the real work, and it is the thing that makes voxel LOD notoriously difficult. The
generated world coarsens for free. **Player edits do not** — and if you ignore that, a tunnel a player
dug vanishes at 200 m and reappears at 100 m, and a base they built pops in and out as they fly. That
artefact is unacceptable under the seamless law and it is the single reason to take this section
seriously.

The fix is a **sparse edit pyramid**, maintained incrementally by the realm owner:

- For each coarse cell that contains at least one edited fine cell, store two small numbers: how many of
  its fine cells are solid, and which material dominates.
- On every edit the owner updates one entry per tier — at most eight small writes. It already has the
  generator, so it can regenerate the affected fine cells, apply the deltas, and recompute the summary
  without reading anything from disk.
- Rendering a coarse tier is then: generate the coarse field by dropping octaves, then **overwrite only
  where the pyramid has an entry**. A dug tunnel becomes a partially hollowed coarse cell rather than
  disappearing. A built tower becomes a solid coarse cell of the right material.

Storage is proportional to edits, not to world size, and sublinear in practice because neighbouring
edits share coarse cells: a thousand-block tunnel produces roughly 1,140 pyramid entries across all
tiers — about 1.1× the edit count, and it shrinks as edits cluster.

Player-built structures — ships, stations, bases — have no generated baseline, so their coarse tiers are
*pure* pyramid. That is simpler, not harder: a ship's distant representation is a downsampled occupancy
and material grid maintained on edit, and it degrades continuously into the realm proxy.

**This is the piece to build carefully and gate hard.** The acceptance test writes itself: dig a tunnel,
fly away, confirm it is still visible at every tier boundary, fly back, confirm nothing popped.

### C.5 Seams between tiers

Where a fine chunk meets a coarse one the surfaces do not exactly agree, so there is a crack. Two
options, and for once the cheap one is right:

- **Skirts (recommended).** Each chunk emits a short downward apron around its border, hiding the crack
  behind geometry that is never seen from above. About twenty lines, no case analysis, no dependency on
  which neighbour is at which tier — which matters, because that neighbour's tier changes as the camera
  moves.
- **Vertex snapping / transvoxel.** Correct rather than concealed: the fine tier's boundary vertices are
  snapped to the coarse lattice. More code, a real case table, and transvoxel in particular is built for
  smooth iso-surfaces rather than blocky quads. Hold it in reserve if skirts prove visible.

### C.6 No pop, ever

The seamless law makes this non-negotiable, and the mechanism already exists in the engine: a
distance-banded dither crossfade costing one pipeline bit. Applied between two rungs of the *same*
terrain — never between a mesh and a substitute — it is a crossfade, not an impostor. Both tiers are
resident and drawn for the width of the band; the coarse one fades in as the fine one fades out.

The band width is a tuning field. The pop detector the earlier analysis asked for becomes the gate that
keeps it honest.

### C.7 What must never get a detail level

The ban was too broad, but it was not baseless. The list of things that stay at full fidelity:

- **Collision. Always tier 0, no exceptions.** Physics never sees a coarse tier. Tier 0 is always
  resident wherever a player is, which is exactly the region physics runs in.
- **Other players, and anything that can shoot or be shot.** No player ever becomes a billboard. This is
  what the original preference was really protecting and it stands.
- **Anything a signal or a functional block depends on.** A blinking beacon must not stop blinking
  because you flew away; its *housing* may coarsen, its behaviour may not.
- **Anything whose appearance carries gameplay information a player could act on** — a docking-clamp
  status light, a hull breach. Coarsen the geometry, never the state.

Two players at different distances legitimately see different terrain *detail*. They must never see
different terrain *facts*.

### C.8 The rest of the ladder, now trivially permitted

The earlier analysis asked whether the ban covered five different things called "LOD". With the ban
retracted, four of them dissolve and need no further ruling: **texture mipmaps** (mandatory — without
them detail normals shimmer at any distance), **shadow cascades** (so a 2 cm feature and a distant ridge
can share one shadow system), **distance-blended detail-normal scales**, and **parameter falloff** (grass
blade count reaching zero on the same mesh). Two things stay banned: substitute geometry for entities,
and impostors for anything a player interacts with.

**Decoration follows by density, not by substitution.** Grass instance count per cell scales smoothly to
zero with distance; props stop being emitted past their own range. Because decoration is derived rather
than stored, that is a parameter change, not a second asset set — and the tier a chunk is meshed at
gives the decoration evaluator its natural cutoff for free.

---

## D. What this changes

### D.1 The decision register

| Row | Change |
|---|---|
| **7 — "no detail levels" scope** | **ANSWERED and CLOSED by owner ruling.** Detail levels everywhere, on the ladder above, with §C.7's exclusion list. This was the top of the register; it is now settled. |
| **21 — vertex format** | **De-escalated out of the blocking set.** Still the right call for bandwidth and upload latency; no longer decides whether a planet is renderable. |
| **4 — planet radius** | Already dissolved in Addendum 1; now doubly so — view distance no longer scales with planet size in any binding way. |
| **new** | **The edit pyramid** — its record layout, its update rule and its persistence — joins the list of formats that must be frozen before the first world is written, because it is delta data and therefore permanent. |
| **new** | **The tier-selection tuning fields** (pixels per cell, crossfade band width) join the render tuning struct. Never literals. |

### D.2 The slice plan

The ladder is not a late add-on; two pieces of it must land inside the terrain slice or they become
retrofits:

- **In the terrain slice (P4):** the tier as a field on the chunk address; octave-dropping in the
  generator; tier selection from screen-space error; skirts. This is small — the generator change is
  *removing* work, and tier selection reuses the existing angular-size machinery.
- **In the edit slice (P6):** the edit pyramid, maintained on every edit and persisted beside the delta,
  with the fly-away-and-back acceptance gate.
- **Everything else** — cascades, mipmaps, decoration falloff, the crossfade — is ordinary rendering work
  that lands with its own subsystem.

### D.3 New gates

| Gate | The number it must hold |
|---|---|
| **Tier count under load** | A 100 km view stays under ~7,000 resident chunks and the stated memory budget |
| **Edit survival across tiers** | A dug tunnel and a built tower are visible at *every* tier boundary; the fly-away/fly-back run shows zero disappearance |
| **No pop** | The pop detector sees no discontinuity at any tier crossing, at walking and at flight speed |
| **Coarse generation is cheaper** | A tier-L chunk costs strictly less than a tier-0 chunk, proving octave-dropping is real and not decorative |
| **Collision is tier-0 only** | An assertion, not a test: physics may not be handed a coarse chunk |

---

## E. The one thing to watch

The edit pyramid is the only genuinely hard part, and it is hard in a specific way: it is **permanent
data**, so its format must be right before the first world is saved, and it is **derived data**, so it
can silently disagree with the truth it was derived from. The mitigations are cheap and should be built
in from the start: recompute it from the deltas on demand in a debug command, and check it that way in
the test suite, so a divergence is a failing assertion rather than a player noticing their tunnel is the
wrong shape from a hilltop.

Everything else here is smaller than it looks, because the generated world coarsens itself.
