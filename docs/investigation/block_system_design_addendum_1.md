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

# Addendum 1 — the grid, the saved record, and planets of any size

**Date:** 2026-08-03. **Amends:** `docs/investigation/block_system_design.md` §3.2, §3.4, §2.7, and decision
register rows 1, 2, 4, 5, 7, 21.
**Why it exists:** the owner reviewed decision-register rows 1–4 and (a) asked for the grid options to be
compared properly rather than voted on, (b) asked whether the saved block record is really big enough,
and (c) **rejected the premise of row 4** — he wants many planets and moons of realistic, differing
sizes, one-metre blocks on all of them, and gravity derived from each body's mass but configurable.

Point (c) is correct and the design was wrong. This addendum retracts a recommendation, replaces it, and
— in doing so — moves the real problem to a different decision than the one it was filed under.

---

## A. The grid: what the blocky-planet article gives us, and the four alternatives

### A.1 Is it "the blocky-planet solution"?

Partly, and it is worth being exact about which parts, because the article's single most distinctive
idea is the one the design **rejects**.

**Taken from it:**

- **The quad-sphere.** Start with a cube, subdivide each face into a square grid, push every vertex out
  to the sphere. Six square charts, one round planet, and — this is the whole point — every block's top
  face points at space and its bottom face points at the core, everywhere.
- **Pre-distortion before normalisation.** Naive projection squashes blocks badly near face edges; a
  warp applied before normalising counteracts it. The author says so explicitly and shows the
  side-by-side, but never published which curve he used.
- **The four-level address** — face, then shell, then chunk, then block — and its derivation:
  largest-absolute-coordinate picks the face, reverse-project and normalise gives the in-face index.
- **Sampling three-dimensional noise on the sphere surface** rather than projecting a two-dimensional
  field, which dodges both the tiling problem and the projection-distortion problem at once.
- **Gravity toward the centre with the player's "up" smoothly interpolated, never snapped.** He reports
  that snapping per frame caused stuttering and disorientation. That is a free bug avoided.

**Rejected:**

- **The doubling shells.** His signature idea: as you descend, circumference shrinks, so periodically the
  tangential block count quadruples — each shell boundary is a place where one block sits on top of four.
  He is honest about the cost: it breaks the universal assumption that a block has exactly one neighbour
  per direction, his own itch.io page lists "rendering gaps between vertical shells" as a known issue,
  and structures placed across a boundary "wrap in upon themselves".
  **We take a single shell with a build band around the surface instead**, and the reason is the owner's
  own requirement: he wants *realistic* planet sizes. Shells exist to let you dig to the core of a tiny
  planet. On a body of realistic size nobody digs to the core — Earth's is 6,371 km down; even a 200 km
  moon gives 100 km of digging under a single shell. The shells solve a problem realistic sizes delete.
- **His unpublished warp.** Replaced with the JCGT 2018 optimal fifth-order odd polynomial, which is both
  lower-distortion than the usual tangent warp *and* uses only addition and multiplication — no
  trigonometry — which is what lets terrain be bit-identical on every machine. That property is not
  optional here; it is a hard gate.

**One thing to be clear about, because it may be the source of the question.** The shells are not what
lets one-metre blocks work at any planet size. What does that is choosing the face resolution in
proportion to the radius — see section C. The shells are about depth, not about scale.

### A.2 The four families, compared

Every scheme pays the same unavoidable tax. Gauss–Bonnet says a sphere's total angular defect is 720°,
no matter how you tile it. You can only choose **how to spend it**: a few big defects or many small ones.

| | **Cube-sphere** | **Literal cube** | **Hex / Goldberg** | **Flat grid, analytic surface** |
|---|---|---|---|---|
| Planet looks | round | a cube | round | round |
| Blocks aligned with "down" | everywhere | only at face centres | everywhere | never |
| Block size uniformity | 0.71–1.00 m (1.41× spread) | exactly 1.000 m | ~0.95–1.00 m | exactly 1.000 m |
| Where the defect goes | 8 corners, 90° each | 8 corners + 12 edges, as *terrain* | 12 pentagons, 60° each | nowhere (there is no surface grid) |
| Same grid as a ship? | **yes** | yes | **no** | yes |
| Greedy face merging | works (runs in index space) | works perfectly | needs re-derivation | works perfectly |
| Shape catalogue | one | one | **a second hex catalogue** | one |
| Seam bookkeeping | 12 edge pairings × 4 orientations, 8 corners | none | 30 edges, 12 vertices | none |
| Building a straight wall | works | works | works | **becomes a staircase** |

**Cube-sphere — recommended.** It is the only family that gives all three of: a round planet, blocks
aligned with gravity everywhere, and *one* grid shared with ships and stations. The price is real and
worth stating precisely:

- Cells are up to 41 % non-square. The worst are at the midpoints of the twelve cube-edge arcs; at the
  eight corners they are equal-sided but small. The gradient is spread over a quarter of the planet's
  circumference, so it is imperceptible locally — you would never see a block change size as you walk.
  It only becomes visible if you compare a structure built in two distant places.
- **This is why §4 rules that a cell is the unit of matter.** One block mined yields one unit and weighs
  one nominal cell's worth wherever it came from. Without that rule, mining in the right place would be
  worth 2.5× as much for the same effort — invisible to players, durable, and poison in a scarcity-driven
  economy.
- At eight points, three blocks meet where four should. Box-shaped prefabs cannot be stamped there; the
  article's author hit exactly this and had to re-encode structures as relative walks. Our answer is the
  same, and it is cheap because blueprints want to be relative walks anyway.
- Neighbour lookup across a face seam is genuinely the hard part. The article's author called it "the
  most challenging part of this project" and admitted "I think I still missed an edge case" after a
  month. **Our mitigation is structural, not diligence:** it is a 24-record table plus 8 known corner
  columns, derived once and pinned by an exhaustive test over all twelve seams; and the main design
  already puts the only chart-dependent line of the edge-crumbling function below the geometry seam so
  the mesher above it never learns the word "face".

**Literal cube — the honest runner-up, rejected on look and on "down".** It is cheaper on every technical
axis: exact 1.000 m cubes, no warp, no seams, trivial addressing, perfect merging. Two problems. The
planet is visibly a cube, which fails the believability standard the whole project is written to. And
gravity has no good answer: point it at the centre and the ground tilts under axis-aligned blocks by
`atan(d/a)` — 45° at a face edge, 54.7° at a corner — so over most of the surface you walk on a slope
while your blocks stay square. Point it per-face instead and you get six flat worlds with 90° cliffs
between them, which is a coherent game but not this one.

**Hex / Goldberg — the aesthetic winner, and the one long-term trap in the list.** It genuinely has the
best cell uniformity. It also cannot be the ship grid: a hex-prism lattice and a Cartesian lattice are
not the same code, so the block system forks in two — two meshers, two colliders, two shape catalogues,
two sets of edge cases, every feature written and debugged twice. That is precisely the structural
failure the greenfield rebuild exists to remove, and it would be self-inflicted at the very first
decision. **The cost is not the hex maths; it is that hard rule four stops being true on day one.**

**Flat grid with an analytic surface — rejected on building.** Blocks never align with the ground, so a
wall built at latitude 40 is a staircase. Worth noting the one place its argument is strongest, because
the owner's new requirement moves the goalposts: **at realistic planet sizes, curvature over a build site
is negligible.** Over a 100 m structure the surface drops 0.2 mm on an Earth-sized body and 7.7 mm at
162 km. So "locally Cartesian" is nearly true anyway — which is exactly what a cube-sphere *is*, six
locally-Cartesian charts. The difference is that a cube-sphere's charts follow the surface globally and a
flat grid's does not.

### A.3 The long-term implications, stated plainly

- **The grid family is the deepest one-way door in the project.** It decides the shape of every planet
  forever, and — through the ship-grid question — whether there is one block system or two.
- **The seam table is the file that will hurt.** Everything else about the cube-sphere is arithmetic; the
  twelve edge pairings with their axis swaps and flips are bookkeeping, and bookkeeping is where the bugs
  live. It must be generated and exhaustively tested, never hand-written.
- **Prefabs, blueprints, and symmetry mode must be relative from day one.** Absolute box stamping works
  everywhere except eight points, which means it works right up until a player builds at a corner.
- **The distortion is permanently visible in exactly one way**: identical structures in different places
  have different real-world sizes. The unit-of-matter rule removes the economic consequence; the visual
  consequence remains and is, in practice, invisible.
- **Nothing here forecloses ships, stations or rooms** — they take the identity mapping and share every
  line of code above the geometry seam. That is the whole reason to accept the distortion.

---

## B. The saved record: is five bytes really enough?

The short answer: **yes for what it holds, because it holds identity and nothing else.** The long answer
is worth having, because "would that be enough for all types of blocks for all biomes, all functional
block states" is really three different questions.

### B.1 What the record actually contains

```
┌──────────────────┬──────────────────┬─────────┬───┐
│ local_index 18 b │ block_type  16 b │ orient 5│p 1│   = 40 bits = 5 bytes
└──────────────────┴──────────────────┴─────────┴───┘
```

- **18 bits — where in the chunk.** A 62³ chunk has 238,328 cells; 2¹⁷ is too small and 2¹⁸ is 262,144.
  Exactly minimal, not a choice.
- **16 bits — which block.** This is the one that answers the question. It is **not** a material id. It
  is an index into a generated table of approved *(substance, form, function)* triples. Sixteen bits is
  **65,536 distinct block types**. For scale: a rich set of 250 substances across 23 shapes is 5,750
  entries, and the whole functional-block catalogue in §7 is about 140 more. We would ship using perhaps
  a tenth of the space, with room to multiply the substance catalogue by ten and still fit.
- **5 bits — orientation.** The 24 proper rotations. Mirroring is encoded as a *separate shape*, not a
  flag, which is what freed the last bit.
- **1 bit — generated or player-placed.** Small and load-bearing: without it, structural support has no
  way to tell a natural cave roof from an unsupported player build, and collapses every cave on the
  planet the first time it runs.

### B.2 Biomes do not live here at all

A biome is not a property of a block; it is a property of a *place*. It is a field evaluated from the
seed and the planet's configuration — temperature, moisture, latitude, elevation — which the generator
consults to decide *which* block type to emit. Two grass blocks in a tundra and a savannah are the same
block type; what differs is what grew around them and how the surface was shaded, which the decoration
layer re-derives from the same field. **Nothing biome-related is ever written into a saved record.** If
biomes were stored per block they would cost more than the blocks do, and they would go stale the moment
a planet's configuration changed.

### B.3 Functional-block state does not live here either — and that is the point

This is the part worth being emphatic about. The record is the **placement log**: what block is where,
which way up, and who put it there. Everything that *changes* lives in side tables keyed by chunk and
cell index, each with its own width and its own file:

| What | Where it lives | Roughly |
|---|---|---|
| Damage, growth stage, wetness, per-material state bits | the block-state table | 11 bytes per *changed* cell |
| Temperature | the thermal table | 6 bytes, sparse active set only |
| Fluid level | the fluid table | 5 bytes, only wet cells |
| **A functional block's player configuration** | the config table → a variable-length tagged blob | a reference plus an evolvable blob, capped per block |
| A HUD panel stuck on a block face | the cover table | face + a reference |
| Sky exposure per column | the column table | not chunk-local, so not per block |

So a cockpit with fourteen named signal bindings, a thruster with its fuel and reactor channel names, a
display panel with a widget and three graphed values — all of that is a tagged, versioned,
skip-unknown blob in the configuration table, sized by what the player actually configured, with a
declared cap to stop griefing. It can grow for years without the placement record changing by one bit.

**Why keep them apart at all?** Three reasons, all measured. Placements are rare and permanent;
state changes are orders of magnitude more frequent — putting them together means rewriting a permanent
log every time a block warms up. Fixed width is what makes the "sort on drain" determinism cheap. And a
record with no spare bits is a *discipline*: it structurally prevents someone from quietly stealing two
bits for a damage stage and thereby putting mutable state on the permanent log.

### B.4 The one genuine dispute

Three sections of the main design specified three different widths — five, six and eight bytes — and the
integration pass recommended eight. §2.7's five-byte layout above is the one that actually closes, and it
closes *because* of two upstream verdicts (mirroring is a shape, not a flag; identity is one combined
number, not separate substance and shape fields). If either of those verdicts is reversed, five bytes
stops fitting and the answer becomes eight.

**So register row 2 is really: do you accept those two verdicts?** If yes, keep five bytes as the roadmap
committed, and the answer to "is it enough" is yes with a bit to spare. If you would rather have slack —
fifteen reserved bits, rejected on decode if non-zero — take eight bytes and pay about 300 MB across a
hundred million edits, which is nothing against the cost of ever migrating the format. Either is
defensible; **what is not defensible is leaving it open past the point where the chunk format freezes.**

---

## C. Planets of any size — the design was wrong, and the real constraint is elsewhere

### C.1 What the design got wrong

§3 recommended a single planet radius of 161,671 m and called it "the unique value" that gives exactly
one-metre blocks with a whole number of chunks per cube face. **The uniqueness was an artefact of an
unnecessary constraint.** The section required the chunks-per-face count to be a *power of two*; nothing
in the addressing actually needs that, because the chunk index requires a division by 62 regardless and
62 is not a power of two.

Drop the power-of-two requirement and the picture changes completely.

### C.2 The radius ladder

A cube face spans 90° of arc, so cell size is `s = πR / 2N` for `N` cells per face edge. For exactly
1.000 m cells with whole 62-cell chunks tiling the face:

> **R = 124·m/π metres, for any integer m** — a ladder with a **39.47 m** step.

Verified against real bodies:

| Body | Real radius | Ladder step *m* | Ladder radius | Error | Chunks per face edge |
|---|---|---|---|---|---|
| Small asteroid | 500 m | 13 | 513.1 m | +13 m | 13 |
| Small moon | 200 km | 5,067 | 199,996.6 m | −3.4 m | 5,067 |
| Ceres | 473 km | 11,984 | 473,013.6 m | +13.6 m | 11,984 |
| Europa | 1,561 km | 39,549 | 1,561,015.9 m | +15.9 m | 39,549 |
| Luna | 1,737 km | 44,008 | 1,737,014.5 m | +14.5 m | 44,008 |
| Mars | 3,390 km | 85,887 | 3,389,996.5 m | −3.5 m | 85,887 |
| **Earth** | **6,371 km** | **161,412** | **6,371,000.4 m** | **+0.4 m** | **161,412** |

**Any planet or moon can be given its real radius to within twenty metres, with exactly one-metre blocks,
on one code path.** The radius stops being a global decision and becomes a per-body number carried as
data — which is exactly what the owner asked for, and it is also what hard rule three requires (a body
type is a configuration, never a code path).

Two consequences to absorb:

- **Register row 4 is dissolved as a global decision.** It becomes: what radius does the *starter* world
  get, and what range does the generator draw from? Far cheaper, and no longer blocking.
- The chunk key grows. Face plus three chunk indices at Earth scale needs about 3 + 3×18 = 57 bits; a
  64-bit key still holds it, but the realm discriminator that §2.7 assumed could share those bits must
  move out. That is a small, contained correction to §2.7.1 and it must land before the key is persisted.

### C.3 The horizon wall — and it is not about planet size

Bigger planet means a farther horizon, and visible chunk count grows **linearly** with radius. Standing
at eye height:

| Body | Radius | Ground horizon | Visible chunks | Generate, 8 cores | Mesh memory (packed) | (at today's vertex format) |
|---|---|---|---|---|---|---|
| The design's pick | 162 km | 741 m | 1,348 | 0.15 s | 54 MB | 0.65 GB |
| Small moon | 200 km | 825 m | 1,667 | 0.18 s | 67 MB | 0.80 GB |
| Ceres | 473 km | 1,268 m | 3,943 | 0.44 s | 158 MB | 1.9 GB |
| Luna | 1,737 km | 2,430 m | 14,480 | 1.60 s | 580 MB | 6.9 GB |
| Mars | 3,390 km | 3,395 m | 28,260 | 3.13 s | 1.1 GB | 13.6 GB |
| **Earth** | **6,371 km** | **4,654 m** | **53,110** | **5.9 s** | **2.1 GB** | **25.5 GB** |

So **standing on the ground works at every size up to Earth — but only if the packed vertex format
lands.** At the format the renderer uses today it does not: 25.5 GB of mesh for an Earth-sized surface is
not shippable, while 2.1 GB is. That is a direct dependency nobody had drawn: **register row 21, the
vertex format, gates how big a planet can be.** It was ranked twenty-first; it belongs near the top.

Now flight, which is where it actually breaks:

| Planet | Altitude | Horizon | Chunks | Generate, 8 cores | Mesh memory (packed) |
|---|---|---|---|---|---|
| 162 km | 100 m | 5.7 km | 79,277 | 8.8 s | 3.2 GB |
| 162 km | 1 km | 18.0 km | 792,775 | 88 s | 32 GB |
| 162 km | 10 km | 56.9 km | 7,927,749 | 15 min | 317 GB |
| Earth | 100 m | 35.7 km | 3,124,103 | 5.8 min | 125 GB |
| Earth | 1 km | 112.9 km | 31,241,030 | 58 min | 1,250 GB |

**At one hundred metres of altitude — a low hover, not orbit — the smallest planet in the table already
needs 79,277 chunks and 3.2 GB.** Planet size barely matters here; altitude does, and it also grows
linearly. There is no radius at which flight works.

### C.4 The finding that reorganises the register

> **The binding constraint is the drawn radius, not the planet radius. Roughly 700 m to 1 km of fully
> meshed terrain is what one-metre blocks with no detail tiers can afford — on any planet.**

The 162 km planet appeared to "work" only because its ground horizon happens to land at 741 m, which is
inside that budget. It was never the planet size doing the work. For comparison, Minecraft's default
render distance is a 512 m radius — so the affordable budget is about twice what the reference voxel game
draws, which is a reasonable place to be, and it is a *fixed* budget independent of how big the world is.

This moves the problem out of row 4 and into row 7 — the already-open ruling on what "no detail levels"
bans — and sharpens what that ruling has to decide:

- **Hold the rule literally.** Drawn radius is capped near 1 km. Every planet works on foot at any size.
  Flight above roughly 100 m either shows nothing beyond 1 km, or is not a thing. On a small planet the
  curve and the atmosphere can hide the cut honestly; on an Earth-sized one they cannot, and the player
  sees a wall of fog at a fixed distance. **This is the only option that requires no new machinery.**
- **Permit server-authored terrain detail tiers.** Flight works. The cost is the genuinely hard part of
  voxel level-of-detail that the earlier analysis already named: a coarsened tier must fold in the delta
  overlay, or a player's dug tunnel vanishes at 200 m and reappears at 100 m.
- **Cap the walkable planet size and make big bodies non-landable.** Cheapest to build, and it directly
  contradicts what the owner asked for in this very review.

I am not going to pre-empt this ruling; it is row 7 and it is the owner's. What this addendum adds is
that **the ruling is now blocking flight, not blocking planet size**, and that the two are independent.

---

## D. Gravity from mass, and worlds that are seed *plus* configuration

### D.1 The one relationship that decides everything

For a body of uniform density, surface gravity is `g = (4/3)·π·G·ρ·R`. Density and the constants are
fixed, so:

> **At the same density, surface gravity is directly proportional to radius.**

A body at Earth's density and 162 km radius pulls at 1/39 of Earth gravity. At a more typical rocky
density of about 3,000 kg/m³ it is 1/72 — which is where the main design's figure came from; the
relationship, not the single number, is the thing to carry. **A small planet with honest physics cannot
have Earth-like gravity. That is arithmetic, not a design choice.**

### D.2 The model, which gives the owner exactly what he asked for

Author **mass** (or density, and derive mass from it and the radius); derive everything else.

```rust
/// Per-body physical definition. Seed-derived by default; every field overridable.
pub struct BodyPhysical {
    pub radius_step: u32,   // m on the 124/pi ladder — the radius IS this integer
    pub mass_kg: f64,       // authored, or density x volume
    // derived, never authored: surface_gravity, escape_velocity, soi_radius,
    // orbital_period, rotation_period, atmosphere_scale_height
}
```

This satisfies the request in both directions. Gravity *is* calculated from mass, so escaping a heavy
world is genuinely harder and the difference between bodies is real and consistent. And it is
configurable, because mass is a field: a curated 200 km starter world can be given an Earth-like pull by
being written as unusually dense.

**The one honest cost of faking density.** A body's gravitational sphere of influence scales as
`(m/M)^(2/5)`, so multiplying mass by 72 to get Earth gravity on a small world multiplies its sphere of
influence by 72^0.4 ≈ 5.5×. That is not a bug — it means the planet's gravitational neighbourhood is
larger than its size suggests, which in practice makes approach and orbit *more* forgiving. It must be
applied consistently from the one mass field, which this model does by construction.

Everything else follows without a second decision: escape velocity, low-orbit velocity and period, fall
speed, jump height, how much thrust a ship needs to lift off, and how far out the realm-lifecycle demand
loop must wake the planet. All derived, all from one authored number, no magic constants.

### D.3 Seed plus configuration

The owner: *"I don't think that we can get that with pure seed, so for some planets I'd love to have seed
+ configuration for size + biome types."* Right, and it costs less than it looks.

The model is **derived-by-default with explicit overrides**: a body definition whose every field has a
seed-derived value, and an optional authored table, consulted before generation, keyed by the body's
position in the realm tree. Procedural worlds carry no overrides and cost nothing. A curated world
carries a handful.

**The determinism gate survives, and this is the part that must be got right.** The rule is that terrain
never crosses the network and only the seed does; if a planet now depends on authored configuration, that
configuration is part of the planet's *definition*, not its terrain. Definitions already stream to the
client as part of the realm description. So: the override table is content-hashed, the hash joins the
protocol handshake exactly as the block registry's does, and a client whose table disagrees is refused at
connect with a named error. **The failure is loud at the door instead of silent in the terrain** — which
is the whole reason to hash it rather than trust it.

Biomes then become per-planet configuration in the same way: a body's definition names its biome *set*
and the parameters of the temperature and moisture fields; the assignment over the sphere is computed
from those fields, sampled in three dimensions on the sphere surface so that no biome boundary can ever
follow a cube edge.

### D.4 The starter world

*"Players appear on an Earth-like planet with a variety of biomes, in the same spaceport"* decomposes
into four things that already exist or are cheap:

1. **A curated body definition** — a radius on the ladder, an authored mass giving comfortable gravity, a
   temperate biome set. One row in the override table.
2. **A habitability score** derived from a body's own parameters — gravity in a comfortable band,
   temperature in a liquid-water band, an atmosphere. Cheap, derivable, and reusable later for
   exploration and for the economy's notion of desirable real estate.
3. **A spaceport stamped deterministically** at a known place. The mechanism should be the *same* one
   blueprints use — a relative walk from an origin block, which is what the cube-sphere corners force
   anyway — written into the planet's delta store at world creation. It is then a normal player-visible
   build: persisted, damageable, repairable, and requiring no special case anywhere.
4. **Spawn selection** that picks the highest-scoring reachable body. One sort.

---

## E. What this changes in the decision register

| Row | Was | Now |
|---|---|---|
| **1 — grid family** | unchanged | unchanged, but section A gives the full comparison that was asked for. Still the first door to shut. |
| **2 — saved record** | "three sections disagree; recommend 8 bytes" | **Reframed:** five bytes closes *if* you accept that mirroring is a shape and identity is one combined number. The real question is those two verdicts. Section B. |
| **4 — planet radius** | "pick one radius; recommended 161,671 m" | **Largely dissolved.** Radius is per-body data on a 39.47 m ladder; any real body fits within 20 m. What remains is a much cheaper question: the starter world's radius and the generator's range. **No longer blocking P4.** |
| **5 — gravity** | "fake Earth gravity, or be honest" | **Reframed:** mass is an authored per-body field and gravity is derived from it. Both behaviours are available per planet. What remains is the *range* of surface gravity the generator may produce, which is a feel decision, not an architecture one. |
| **7 — "no detail levels" scope** | one of several rendering rulings | **Promoted to the top of the register.** It is now the single constraint deciding whether flight above ~100 m exists, at every planet size. Section C.4. |
| **21 — vertex format** | ranked twenty-first | **Promoted to blocking.** It decides whether an Earth-sized body is renderable on foot at all (2.1 GB packed versus 25.5 GB today). |
| **new** | — | **The chunk key must be re-derived for variable radius** before it is persisted: Earth scale needs ~57 bits for face plus three indices, so the realm discriminator §2.7.1 packed alongside it must move out. |

**Net effect on the critical path:** two of the four questions that blocked all terrain work (radius,
gravity) are downgraded to per-body data. One question (the detail-level ruling) is promoted and is now
the one that decides whether ships can fly low over a planet. The grid family remains the first door, and
the saved record remains the format that must be frozen before the first world is written.
