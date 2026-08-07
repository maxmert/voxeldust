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

# Collidable decoration — what collides, what responds, and what the server owns

**Status:** settled ruling, awaiting the owner decisions in §12. Binding on `docs/investigation/stunning_look_plan.md`
§3.5 and §6.4, and on `docs/investigation/block_system_design.md` §3.1, §4.9.4, §6.2 and §6.6, where those sections
disagree with it. It does **not** touch, weaken or reinterpret the composite subsystem's single rule.
**Date:** 2026-08-03.
**Phases:** the shape and table reservations land at **P4**; the collision half at **P5**; marks, snow depth
and the edit path at **P6**; the drawn rungs at **P6/P7**.

**Produced from** five independent investigations (the collision line; density scaling; surface response;
the overhang problem; vehicles and the true cost of server-side derived collision) and one adversary pass
that attacked all four rulings. §13 adjudicates the adversary line by line. Where the adversary is right
against an investigation, the adversary wins and it is said so; where an investigation is right against the
adversary, the same.

---

## 1. The owner's ask

Verbatim:

> *"Would it be possible to use decorative elements, that also would be collidable? Stones appearing on the
> flat block, grass, snow with effects when I walk. Or trees when I put leaves, it takes more than the
> square, so looks believable? … if I put a lot of grass blocks, grass becomes bigger in that area, and it's
> 3d and interact with the player. Same with stones — the size of the stone on top is growing. In theory
> I'd like all collisions to be server side driven, so server should be able to understand what decoration
> was generated and do collision calculations. Especially when we will implement vehicles."*

It contains five distinct requests, and they have five different answers.

| # | The ask | The short answer |
|---|---|---|
| **1** | **A stone sitting ON TOP of a flat block, that you collide with.** There is no block there today. | **Yes — and it is a block.** The block is the air cell above the flat one. |
| **2** | **Density scaling.** More grass blocks ⇒ taller, denser grass. More rock blocks ⇒ a bigger stone. A continuum, not a switch. | **Yes**, by two different mechanisms that meet at one line: below the line a continuous field, above it a rung ladder. |
| **3** | **Surface response.** Snow with effects when you walk — footprints, sinking, sound, movement cost. | **Yes, all four** — and only one of them is collision. |
| **4** | **Canopies that overhang their blocks**, so a tree "takes more than the square". | **Yes — in cells, not in mesh.** The tree is already ~120× wider than its trunk. |
| **5** | **All collision server-authoritative, including vehicles.** | **Already true, and strengthened.** Vehicles get a sampler, not colliders — and they raise a much bigger problem than collision, named in §8. |

---

## 2. The answer in one page

*No engineering words on this page. This is the page to read.*

**Yes to the stone, and it works in a way you will like better than what you asked for.** When you put a
stone on top of flat ground, what you are really doing is filling the empty space above that ground —
and empty space is already a real part of the world, exactly as real as the dirt under it. So a stone on
the ground is not a decoration bolted on top of the world; it *is* part of the world, in a space that
already existed. It draws as a proper rounded rock from the art pack, it does not draw as a cube, and you
walk into it, stand on it, shoot it, mine it and carry it away. Nothing new had to be invented for any of
that, and the world does not have to remember it: on a wild planet the rocks are decided by the same
recipe that decided where the mountains are, so a whole continent covered in boulders costs the game not
one byte of storage and not one byte sent over the network.

**Grass is the one thing on your list you cannot have.** Grass will never physically stop you, and this is
the only place where the answer is a flat no. Six shipped games we can point at all refuse it, one engine
documents refusing it as a permanent limitation, and the one well-known game that *did* give small ground
clutter real solidity — Valheim — has a years-old complaint thread about exactly that: you are running
away from something, you clip a bush, and you stop dead. The problem is not that it is expensive. The
problem is that solid grass makes the ground lie to you. What you get instead is better and is genuinely
physical: grass bends and parts around you as you walk through it, it whispers when you brush it, it hides
you if we ever want it to, and deep grass slows you down — and that slowing is decided by the server from
the actual soil you are standing on, so nobody can turn grass off on their own machine and run faster.
You will feel the meadow. You will just never be stopped by a blade of it.

**Yes to grass getting bigger where you plant more of it, and yes to the same for stones — but they are
two different effects and you should know which is which.** For grass, the world counts how much soil is
around and grows the grass smoothly to match. There is no moment where it flips: put one more soil block
down anywhere and every blade near you grows by less than a millimetre. Put down a twenty-metre meadow
and the middle of it is knee-high while the edges thin out over the last eight metres, which is what a
real meadow looks like. For stones it is not smooth and it should not be: one stone block is a rock you
can sit on, two is a bigger rock, eight is a boulder, thirty-two is an outcrop. That is a ladder with
steps you can see, and that is right, because a stone you can climb on has to be honest about where its
edges are in a way that grass does not.

**Yes to snow you sink into, and yes to footprints, and yes to it slowing you down.** Snow you can wade
through and shovel is made of real, thin layers of world — eight of them per metre, so twelve and a half
centimetres at a time, which is the same slicing Minecraft settled on for the same reason. The white
dusting on a distant mountain top is *not* made of anything; it is computed from the climate wherever the
sky can reach, costs nothing, and reaches all the way to the horizon. Your footprints are real, the server
knows about them, and other players see the same ones — a tracker can follow your trail, and it survives
you logging out. Walking through deep snow is slower than walking on rock, and that is a property of snow
rather than a property of a footprint. The one thing you asked for that we deliver differently is the
*feeling* of sinking: rather than making the ground secretly lower than it looks, we press a dent into the
snow's surface where you stepped. You see the dent, you feel the drag, and the ground never lies about
where it is.

**Yes to a tree taking more than its square, and it already does — by a lot.** A grown tree is not one
column of blocks. It is roughly one square metre of trunk and one hundred and twenty square metres of
canopy — a hundred and twenty times wider than what you planted. That is what makes it read as a tree
rather than a green pillar. What we will *not* do is let the painted leaves stick out past the part you
can touch. Two documents written the same day contradicted each other on this, one allowing a metre of
leaves hanging over nothing and the other forbidding it, and this ruling settles it in favour of the
stricter one: **you will never see something you cannot touch.** Sometimes you will touch a few
centimetres more than you can see — you will brush the edge of a canopy that looks like empty air. That is
the direction every good game errs in, because "the world is solid and I caught slightly more of it than I
expected" is forgivable and "I fell through what looked like a branch" is not.

**Yes, the server does every collision, and it always did — for a reason that is worth understanding,
because it is the whole trick.** The server never has to guess what decoration was generated, because
everything that can be collided with is made of world, and the server *owns* the world. There is no second
list of invisible obstacles anywhere. This matters most for the thing you have not built yet: it means a
vehicle driving over a boulder field is doing exactly the same thing as a player walking over it, with no
extra machinery, no extra memory and no risk that one of them sees a rock the other does not.

**But vehicles raise a problem that has nothing to do with collision, and you should hear it now rather
than at the end of P5.** A planet made of one-metre cubes has hillsides that are staircases, and every
step in that staircase is exactly one metre high. A person can jump one metre; that is deliberate, and the
jump was sized against exactly that step. A wheel cannot. At a gentle one-in-ten slope, a vehicle doing
thirty metres a second meets a one-metre wall three times a second. No amount of collision engineering
fixes that; it is the shape of the ground. There are three honest ways out — hovering vehicles, which is
what every block game ships and which needs nothing new from us; tracked vehicles, which bridge the steps;
or teaching the world generator to put sloped pieces on the very top surface of the ground, which works
but changes what every already-built world looks like and therefore has to be decided **before** the first
planet is saved rather than after. Deciding it early costs one setting today. Deciding it late costs
everybody's world.

**One last thing you cannot have, and it is not about money.** Anything that is drawn but not made of
world — the grass, the pebbles, the ripples — can never be collided with, at any price. Not because it
is expensive, but because the amount of grass drawn depends on how far away you are standing and on how
much memory *your* graphics card had spare last frame. The server has neither a distance nor a graphics
card. Two players standing in the same field are shown different amounts of grass on purpose, and that is
fine as long as grass is only ever a picture. The moment it becomes something you bump into, they are
standing in two different worlds. That is the real reason, it will not change with better hardware, and
it should be written down once so nobody re-opens it in a year after re-running the arithmetic and finding
that the *cost* looks affordable. It does look affordable. It is still wrong.

---

## 3. The rule

Stated once. It decides every case in this document and every case not yet raised.

> ### THE RULE
>
> **A thing collides if and only if it occupies a CELL, or it is an ENTITY.**
>
> **It occupies a cell if and only if (a) a body can rest on it, or a player can remove it and keep it
> removed; AND (b) it is representable in the shape catalogue at cell scale.** Below the catalogue's
> smallest extent there is no cell to occupy and therefore no collider, ever — the base cell is never
> subdividable.
>
> **Everything else is DERIVED. A derived thing may bend, sound, mark, slow, conceal, and be sampled at a
> point by the server. It may never own a collider, a raycast hit, a fence, or a byte of per-instance
> storage, at any distance, on any machine.**

Three notes, each of which is a thing that was got wrong somewhere in the source material.

**Both arms are load-bearing, and the second one is easy to forget.** `block_system_design.md` §6.6 already
says *"PROMOTED to a real block … **or a real entity**"*. Compressing that to "if it must collide, it is a
block" fails on five things that ride the entity arm and would otherwise be rebuilt badly: a rope or crane
cable (a capsule chain along a curve, which cannot be voxelised without being a metre thick — §7.16.1
already names loose cranes as multibody trees); a hinged door, whose collider must sweep continuously; a
**falling group**, whose cells have deliberately left the block field for the ~2.5 s of a topple; a ship,
whose blocks are cells but whose body is an entity; and a projectile. None of these needs an exception to
the composite rule, because none of them is a composite in the block field while it is behaving that way.

**The threshold is representability, not height — and this is a correction to the collision investigation's
own recommendation.** That pass proposed the character controller's step height (0.53125 m) as the line,
reasoning from Valheim's shipped complaint that a collider below step height is invisible to a pedestrian
and only hurts vehicles. The diagnosis is right and the threshold is wrong, because **our character
controller auto-steps and Valheim's does not**: `autostep.max_height` = 544 FINE = 0.53125 m is configured
precisely so a player can walk onto a `Slab` without jumping. A 0.25 m stone that is a real cell is
therefore *climbed*, silently, and never produces Valheim's bug. What genuinely cannot collide is what
cannot be a cell: the catalogue's smallest extents are `Panel` at 1/8 cell in one axis (**0.125 m**) and
`Nub` at 1/8 volume (**0.5 m in every axis**). A 2 cm pebble, a blade of grass, a sand ripple and a fallen
leaf have no representation at cell scale and never will, because the base cell is never subdividable.
Step height stops being a category boundary and becomes what it actually is: the point above which a cell
stops being a step and starts being an obstacle. It decides *feel*, not *kind*.

**The converse must be stated in the same breath or "derived" drifts.** The investigation into the
collision line offered, as a fallback if the owner overrules the grass ruling, *"a promoted 0.25 m Plate
block with no collider"*. **That is refused outright.** A cell that exists for storage but not for physics
is precisely the *"cell that exists for physics but not for storage or vice versa"* the composite rule
forbids, read in the other direction. There is no such object and there must never be one.

### 3.1 This does not contradict the composite subsystem's binding rule

The rule this document inherits, quoted in full:

> *"Every cell a composite occupies is a real cell in the block field. Recognition selects a RENDERING, and
> nothing else. There is no second occupancy authority, no overlay, no derived collider, and no cell that
> exists for physics but not for storage or vice versa."*

It was tested against every ask above and it survives all five without amendment. Three places where it
was under genuine pressure, and how each resolves:

1. **A stone on flat ground.** The pressure was "there is no cell to own". There is: the air cell above the
   flat one, which is already a real cell in the field holding air. Filling it makes the stone a cell.
   No exception (§4).
2. **A footprint in snow.** The pressure was "an authoritative displacement is a second occupancy
   authority". It is not, and it is proved rather than asserted: a mark changes no cell's occupancy, so a
   bit-identity gate over the collider, the mass properties, every raycast, the structural flood, the
   pressurisation graph and every mount test — with every mark at maximum and with every mark cleared —
   is the mechanical proof. A mark is **block state on an existing cell**, of exactly the same kind as
   `wetness` and `burnt`, and it rides the machinery those already ride (§6).
3. **A tree in mid-topple.** The pressure is real and nobody caught it: a detached group's cells *leave*
   the block field for the duration of the fall, so the recognition predicate cannot see them and the
   falling tree draws as tumbling cubes — on the harvest path the provenance design describes as the
   *intended* one, 1.9× faster than block-by-block. The resolution is the rule's second arm: **during a
   topple a composite is an ENTITY, and §3.2 does not reach it.** Carry `(recipe, variant, orient)` — four
   bytes — on the falling group's blob and draw the art mesh at the entity's pose. Its collider, if the
   reserved rigid-body tier is ever enabled, still comes from the tier-0 occupancy of the detached cells,
   exactly as the provenance design already requires. No second occupancy authority appears at any point.

Two of the composite design's **gates** are amended by this ruling — §3.5's fill threshold and its
containment clause — and both amendments make §3.2 *more* true, not less. They are argued at length in §7.

---

## 4. Ask 1 — a stone on flat ground

### 4.1 The cell that was already there

The tension the brief names is that "the composite rule does not cover this, because there is no cell to
own". It dissolves on one observation: **for a rock sitting on top of a flat block there is a cell — the
one above it.** It is a real cell in the field today, at tier 0, holding `Air`. Placing a `Stone` block in
it with a shape from the existing catalogue makes the rock a cell with:

- no exception to the composite rule,
- no second occupancy authority,
- **no new row in the shape catalogue**,
- a real collider, from the baked `ColliderTemplate`, on whichever of the four collision lanes the realm
  uses,
- real mass, derived from `volume_units` exactly as every other cell's is,
- a real raycast target, a real mount surface, a real structural-support node, a real WAL record,
- and free degradation: mine it and it is gone, because it was always a cell.

The catalogue already carries every collider such a thing needs, with exact integer volumes on the
1/196,608 lattice and baked orientation orbits:

| Shape | Id | Volume | Extent standing proud | Orients | What it reads as |
|---|---|---|---|---|---|
| `Panel` | 3 | 1/8 | **0.125 m** | 6 | a flat shard, a scree plate |
| `Plate` | 2 | 1/4 | 0.25 m | 6 | a low flat-topped stone |
| `Quarter` | 15 | 1/4 | 0.5 × 0.5 × 1.0 | 12 | a leaning slab, a standing stone |
| `Nub` | 17 | 1/8 | 0.5 m cube | 8 | the default rock |
| `Tetra` | 13 | 1/6 | wedge | 8 | a split shard |
| `Slab` | 1 | 1/2 | 0.5 m | 6 | a broad flat boulder top |

**The rock's LOOK is not the shape.** The shape is the collider; the drawn thing is a rank-1 composite's
art mesh, exactly as the composite design already specifies. So "it is a cell" does not mean "it looks like
a cube", and the owner's actual complaint — *"the decorational not-flat rock"* — is answered by the
composite lane rather than by the catalogue.

**Why not grow the catalogue instead.** A new row costs 64 B plus its orbit at ~976 B of baked template
and collider — a 24-orient shape is 23.4 KB, and ten of them take the table from 0.33 MB to 0.56 MB.
Trivial in bytes; not trivial in commitment. Every row is a permanent one-way door on an append-only,
never-renumbered id space; it must pass exact-volume **equality** with no tolerance, total mirror closure,
conservative and liberal 32×32 per-face rasterisation, an octant mask and a `COARSE_SHAPE` entry. The
catalogue already excludes curved shapes by construction for precisely these reasons. **Zero growth is
available and is therefore the answer.**

### 4.2 This does not reopen the rejection of per-cell rounded templates

`stunning_look_plan.md` §3.1 rejected a per-cell rounded render template on the adversary's arithmetic: a
tier-0 chunk's exposed rock surface is about 3,844 cells, and at 24 triangles per template that is 92,000
triangles per chunk against a measured whole-chunk figure of 4,400 — 21×, and 44–131× at believable
template sizes. **That rejection stands and is not touched**, because the stone class here is a different
object by two orders of magnitude:

- density **0.005/m²** against roughly **1 exposed rock cell per m²** — 200× sparser;
- **instanced** from one baked archetype, not emitted per cell into the chunk's vertex buffer;
- and it is *placed*, so it exists where the generator or a player put one, not on every exposed face.

The frame arithmetic is in §9 and comes to 0.11 ms, against the 1.32 ms of reserve the look plan closes
with. §3.1's 21× does not apply and its verdict is not reopened.

### 4.3 What §3.1 got wrong, and the one correction this ruling makes to it

`stunning_look_plan.md` §3.1 rules that *"the rock half is not a composite"* and delivers it as the
existing rim warp *"with a **larger per-material amplitude** and a per-corner hash direction"*. Half of
that is right and free; the other half is refused, on the design's own arithmetic.

**Adopted: the per-corner hash direction.** Free, contained by the existing containment proof, and it is
what makes a rock face read as an outcrop rather than as eroded dirt.

**Refused: the larger amplitude.** §5.6.7's tier fade is deliberately fed the registry **maximum**, not
the per-material value — `w(L) = smoothstep(warp_frac_off, warp_frac_on, A_max / 2^L)` — *"so all materials
fade together"*. One rock row raising `A_max` therefore taxes **grass, dirt, sand and every other material
in the world** at the coarse rungs where the ladder's whole memory saving lives:

| `A_max` | tier 0 | tier 1 | tier 2 | tier 3 | min lattice edge (1 − 2·A_max) |
|---|---|---|---|---|---|
| **0.20 m** (E3, the owner's answer) | 1.00 | 0.50 | **0.00** | 0.00 | **0.60 m** |
| 0.30 m | 1.00 | 1.00 | 0.16 | 0.00 | 0.40 m |
| 0.35 m | 1.00 | 1.00 | 0.32 | 0.00 | 0.30 m |
| 0.50 m | 1.00 | 1.00 | 0.84 | 0.04 | **0.00 m — degenerate** |

At 0.30 m the merge-defeating warp predicate stays alive a whole extra rung for every material, against a
priced merge loss of 2.2× retained quads on warped geometry. Three other quantities move with it: the skirt
depth `(4·k_rough·2^(L+1) + A_max) × skirt_safety`, the minimum lattice edge above, and the standable
chamfer overhang §5.6.6 lists as failure mode 1.

> **RULING.** `A_max` is a **build-asserted registry ceiling at 0.20 m**, discharging the owner's answer to
> **[USER DECISION 5-E]** (E3) as a hard constraint rather than a default. The rock class reads as an
> outcrop through the composite lane — a rank-1..N recipe with an art mesh — at zero cost to the warp and
> zero cost to every other material. Raising `A_max` re-opens 5-E and must be put to the owner explicitly.

### 4.4 Growing it: the rank ladder is already the answer

*"The size of the stone on top is growing"* is a property of the composite recipe table, not a number
anybody types. `rank = core.count_ones()`, asserted at table build and never authored, and the total order
is `(rank descending, anchor cell ascending, recipe identifier ascending)`. So:

| Blocks placed | Rank | What appears | Collider |
|---|---|---|---|
| below the catalogue floor | — | derived pebble scatter, size scales continuously with abundance (§5) | **none, ever** |
| 1 | 1 | a rock you can sit on, ~0.7 m | `Nub` / `Plate` / `Quarter` |
| 2 | 2 | a bigger rock | its two cells |
| 8 (2×2×2) | 8 | a boulder | its eight cells |
| 32 | 32 | an outcrop | its thirty-two cells |

Continuous in look below the line, discrete and legible above it — which is correct, because a stone you
can climb onto must be honest about where its edges are in a way that grass need not be.

**Two clauses the stone recipes need that the tree recipes do not**, both free and both discovered by the
adversary:

1. **`footprint == core` is asserted at table build for every inorganic recipe.** The composite design's
   anti-conjuration bound is *area × time* with maturity *"measured in days of world time"*. That is sound
   for wood and absurd for stone — a boulder that grows over seven days is not a mechanic anyone wants, and
   at any shorter maturity the bound collapses into a matter printer. So an inorganic recipe is
   **rendering-only**: it writes no footprint, materialises nothing, and its cells are exactly the cells the
   player placed. One assertion.
2. **An `isolation` clause on the recipe.** Recognition is over block *patterns*, and a 2×2×2 stone cube is
   the most natural building primitive there is, so without this a player's wall turns into scenery. The
   fix costs one mask and needs no invariant change: **cells in a declared shell around the core must not
   hold the core types**, so a lone cluster recognises and a wall's interior does not. Note that reading
   `Provenance` instead — the obvious alternative — is banned by the provenance design's own invariant
   (*"provenance NEVER gates composite recognition"*), and this ruling does not relax that ban, because the
   pattern answer is strictly cheaper and needs no second reader of a durable field.

### 4.5 The one honest cost of promotion: things grow with distance

The adversary found this and it is a genuine attack on the recommendation, so it is priced rather than
waved away.

The coarsening rule is *"occupancy is non-empty, not half-full"* — which is what stops a wall of `Panel`s
vanishing at range, and which has the opposite consequence for a sparse small object. A `Nub` alone in a
2×2×2 group has one bit of `occ_mask` set and `fill_256 = 4`, so `bucket = 0` and `COARSE_SHAPE` picks a
`Nub` at the coarse scale: **the object doubles in linear size at every rung.** 0.5 m at tier 0, 1 m at
tier 1, 2 m at tier 2, 4 m at tier 3.

The probability a coarse surface cell contains at least one promoted object is `1 − e^(−ρA)`:

| Density | tier 1 (4 m²) | tier 2 (16 m²) | tier 3 (64 m²) |
|---|---|---|---|
| Boulder, 0.005/m² | 2% | 8% | **27%** |
| Pebble, 0.05/m² | 18% | 55% | **96%** |

At boulder density this is **correct behaviour**: a 3.5 m boulder genuinely is a multi-cell feature and
should survive coarsening. At pebble density it would be a lumpy 4 m crust over the entire distant surface
at a range of 6.3 km — a one-pixel-amplitude noise on every horizon silhouette.

**This is therefore a third independent argument for the representability floor**, arriving from a
direction none of the investigations approached from: promoting sub-catalogue objects would be visible as a
distant artefact even if it were free and even if collision were not the point. It also owes a gate
(§9.6): assert that promoting a class does not change the tier-3 silhouette beyond a stated cell count over
a fixture.

### 4.6 Cost, in one line each

| Quantity | Generated stone | Player-placed stone |
|---|---|---|
| Wire bytes | **0** — terrain never streams, only the seed crosses | 8 B, one ordinary block record |
| Stored bytes | **0** | 8 B, and it prunes on equality if the world regrows over it |
| New shape rows | **0** | 0 |
| New record types | **0** | 0 |
| New inter-shard message arms | **0** | 0 |
| New entity kinds | **0** | 0 |
| Collapse behaviour | **already specified** — `Provenance::Feature`, rigid while intact, falls as one piece when undermined | `Placed`, ordinary |

The last row is worth stating plainly: *"mine under the boulder and it drops"* is the **worked illustration**
in the provenance design's own one-page summary. Promoting stones to cells does not add a mechanic; it
activates one that is already designed, tested and gated.

---

## 5. Ask 2 — density scaling

Two mechanisms, because the ask has two halves that produce different things. They meet at the line in §3
and must never be merged.

### 5.1 Above the line: the rank ladder (§4.4)

Already built, already arithmetic, nothing owed.

### 5.2 Below the line: abundance

*"If I put a lot of grass blocks, grass becomes bigger in that area"* is a question about substance density
over tens of metres. The answer is **one derived scalar field**, `abundance`: an integer box-filtered count
of kin cells over a 17×17 m surface neighbourhood (R = 8 m), evaluated on the **same 8 m lattice the clump
field already uses**, bilinearly interpolated in q8, driving three presentation parameters — instance
density, instance scale, variant weight — and nothing else.

**The load-bearing decision is where it lives.** Not in the per-cell decoration record; in a separate
4-byte-per-lattice-point buffer. That single choice is what makes the feature exist at all:

| | In the per-cell record | On the 8 m lattice |
|---|---|---|
| Dirty set per edit | 17×17 = **289 cells** | **≤ 9 lattice points**, at any radius |
| Bytes written | 2,312 B in 17 runs | **36 B in ≤ 3 runs** |
| Recompute | 214 µs | **2.7 µs** |
| Against the shipped `decor_edit_latency` gate (p99 ≤ 0.15 ms, ≤ 9 ranges, ≤ 216 B) | fails all three — 10.7×, 1.9×, 1.43× | **13.5% of the typical budget, 1.8% of the p99 ceiling** |

Solving for the record-resident radius that would fit gives R ≤ 2 m — a 5×5 m neighbourhood, which cannot
express a meadow. The lattice is not an optimisation; it is the difference between the feature and no
feature.

**The stability the owner is implicitly worried about is solved by construction, not mitigated.** A box
filter has bounded single-edit sensitivity of exactly `1/(2R+1)²`. At R = 8 m that is 1/289: placing one
block anywhere changes blade height by **≤ 0.75 mm** on a blade that runs 83–300 mm, worst case ≤ 1.5 mm
allowing for the ramp curve's maximum slope. The obvious alternative — a threshold ("≥ 20 grass blocks
within 8 m ⇒ tall grass") — has **unbounded** sensitivity at the threshold: one block flips all 289 columns
by 217 mm at once. That is a factor of 289 and it should be a gate, not a hope:
`decor_abundance_sensitivity` asserts for every shipped row that a single-cell edit moves the size and
density multipliers by at most `ceil(range / (2R+1)²)`.

**Everything in the evaluator is integer**: a per-column kin count capped at `kin_col_max` (so a
sixty-two-storey ship shaft cannot read as a super-meadow), a box sum, a multiply-shift by a
build-asserted reciprocal, a q8 bilerp, and a 256-entry q8 ramp lookup. No division at runtime, no `powf`,
no float crossing between Rust and WGSL. It is Category A′ exactly like the rest of the decoration layer.

**It must share the clump field's lattice, asserted at table build.** `radius_log2 == lattice_log2 ==
clump_lattice_log2 == 3` gives the same addressing, the same fractional coordinates, the same bilerp
helper and the same anchor-invariance (both key on `GlobalCell`, so a `reanchor()` re-rolls neither). Two
lattices is the class of bug this design exists to avoid. One caveat that differs between the two channels
and must be stated: the clump lattice is permitted to be discontinuous at the twelve cube-face-edge
pairings because clump drives only density and variant weight. **Abundance drives SIZE**, so a
discontinuity would be a visible band of short grass along twelve great-circle arcs. The box filter
therefore walks neighbours through `FrameSpace`, making abundance continuous in value; only the
interpolation lattice is misaligned, leaving a C¹ kink bounded by the local gradient — exactly zero in a
uniform meadow. Fallback if the seam walk proves expensive: truncate the box at the face boundary and
normalise by the truncated area, which is C⁰-correct and also gives 255 on both sides of a uniform meadow.

**The edit pyramid cannot be the source, and the reason is worth recording.** It is tempting —
`coarse_summary(addr)` is already an 8 m lattice, already updated by every edit's walk-up, and the ground
tint already reads it. It fails for two independent reasons. `fill_256` is an *occupancy* fraction: a
surface coarse cell straddles the ground plane, so its fill is ~128 whether it is dirt or stone and carries
no abundance signal. And `substance` is the **dominant** substance — a mode over 512 cells — and a mode
flips discontinuously at a tie, so one block placed near the tie flips the whole coarse cell's
contribution. **A mean has bounded sensitivity; a mode does not.** That is the threshold failure mode
reintroduced through the back door at 1/4 instead of 1/289. The collision-line investigation recommended
the pyramid path; the density investigation refuted it, and the refutation is correct. The standing
statement that §6 needs no new pyramid field of any kind survives intact, which matters because it is the
stated reason the pyramid record could spend its authored bits on the octant mask.

**The ceiling.** A derived thing may be drawn as tall as `autostep.max_height` = **0.53125 m** and no
taller, asserted at table build as `feature_h_q8 × scale_hi_q8 ≤ decor_max_h_q8`. Below the step height,
walking through something is visually indistinguishable from stepping over it; above it, you are visibly
passing through an obstacle, which is the complaint that started this. Grass therefore runs 0.083 → 0.30 m
at the shipping row and may be authored to 0.53 m; pebbles run 0.02 → 0.45 m. Neither ever needs a collider.

**Bulk edits are cheaper per block than single edits**, which inverts the usual intuition and is why a
player terraforming a hillside does not stall: the dirty lattice set is bounded by the batch's bounding
box, not by the block count. A 20×20 m meadow placed as one batch dirties 6×6 = 36 lattice points ≈ 10 µs,
against 400 × 2.6 µs = 1.04 ms if processed one block at a time. It needs no bulk path of its own; it rides
the existing batch-union-dedup machinery.

**The apron is the one genuinely new structural cost.** A lattice point's box needs kin data 8 cells beyond
its own position — **8× the mesher's one-cell apron** — so a chunk's outermost lattice points must be
deferred until neighbours load. The machinery already exists and is already specified; what widens is the
bug. Skipped, it produces a ring of shortest-grass at the residency frontier that travels with the player,
which is exactly the walking-traverse complaint `decor_pop`'s first traverse was written to catch. Inside
the 40 m micro radius the neighbours are always resident, so this only bites at the outer prop range and at
warp arrival.

### 5.3 The one legitimate coupling between the two mechanisms

Add `cosmetic_group: u8` and `abundance_bias_q8: i16` to `CompositeVariant`, with a table-build assertion
that **all variants sharing a cosmetic group have byte-identical footprints**. Variant selection at
recognition becomes `hash × (base_weight + (abundance_bias · abundance) >> 8)` restricted to the winning
group. A lone tree in open ground draws as a spreading open-grown oak; the identical core pattern inside a
dense wood draws as a tall narrow forest-grown one. Because the footprints are byte-identical, **nothing
authoritative moves**: the cells, the collider, the mass, the raycast target, the support graph and the WAL
record are all unchanged, so the composite rule is untouched.

**Abundance must never be a recognition-predicate input, must never affect rank, and must never affect a
footprint.** Feeding an 8 m box filter into the predicate would widen the composite dirty set from a tight
≤ 8³ core mask to a 289-column neighbourhood *and* would make persisted, collidable geometry depend on
blocks eight metres away — the owner's stated failure mode at its most severe.

**This is the only true one-way door in §5.** The recipe table's freeze rule makes adding fields to a
shipped row a table migration, and the recipe numbering is already a named one-way door. Two struct fields
and one assertion today; a world-format epoch later.

### 5.4 "It interacts with the player" — four levels, three affordable

| Level | What it is | Cost | Verdict |
|---|---|---|---|
| **L0 wind bend** | already designed: a per-realm closed form, ~12 ALU/vertex, zero stored, zero on the wire | **free** | ships |
| **L1 body bend** | grass parts around a moving body — a 128-byte uniform of ≤ 8 influencers, filled from the positions the client is **already drawing bodies at** (the server-authored absolutes on the 100–150 ms buffer), so no prediction and no new authority | **0.006–0.032 ms**, budget 0.05 ms | **[UD-6]** |
| **L2 movement cost** | deep foliage slows you: one server-side point query per player per tick, multiplied into the commanded speed | 2,560 queries/s at 128 sessions ≈ **0.0004 ms/tick** | **[UD-7]** |
| **L3 per-blade collision** | — | the server has no camera so it must use the plateau set: π·64²·100 = **1.29 M colliders** in one physics cluster, **150 MB** of collider against a **≤ 27 MB** budget, rebuilt per tick | **rejected — 5,500× over, not 5×** |

L2 must damp through the impostor crossfade using the multiplier wind already uses, or the handover is a
visible freeze. And L2 requires one sharpening of the promotion boundary's wording, which should be adopted
explicitly rather than inferred:

> **The promotion boundary is about IDENTITY, not INFLUENCE.** A sampled scalar field derived from the
> block field may influence the simulation. An *instance* may never be acted on, targeted, collided with,
> raycast or persisted.

Walking slower in deep grass is the **soil blocks** slowing you — abundance is a box filter over
authoritative cells, and no blade exists as far as the simulation is concerned. Three properties survive
verbatim: no instance ever enters the broadphase; a modified client that draws no grass is still slowed,
because the server computes the slowdown from the block field; and *"two players may see different
decoration density, they can never see different decoration facts"* holds, because the fact is evaluated
at the plateau, identically for everyone.

---

## 6. Ask 3 — snow and surface response

Three layers. The boundary between them is **"does it change occupancy?"** — not "is it cosmetic?".

### 6.1 L1 COVER — the white on the mountain

A closed-form shading term from the climate field the weather design already commits to:
`f(seed, coarse cell, season_index, sky_exposed)`, octave-dropping with the terrain ladder so it is
available at every rung, all the way to the horizon. **Zero bytes stored, zero on the wire, no collider,
no contact with any of the eight readers.** A sheltered patch clears itself for free because `sky_exposed`
is already derived.

Making it seasonal widens P4's Definition of Done from `f(seed)` to `f(seed, season_index)`, which is a
real change to a stated DoD and must not be taken silently — **[UD-9]**. The safe form is integer:
`season_index = universe_tick / season_ticks`, so the contract stays closed-form and gateable with one
extra fixture parameter. A continuous-time input is refused outright: it makes a saved world irreproducible
without also saving the tick, and it makes the prune rule chase a moving target every tick.

### 6.2 L2 SUBSTRATE — snow you sink into and shovel

**This is a BLOCK.** A `SnowLayer1..8` shape family (eighth-metre steps — Minecraft's shipped
quantisation, and the same eight steps the current three `snow_depth` state bits encode) × `Snow` and `Ash`
material rows. It collides because it *is* a cell, exactly as a composite tree collides because it *is*
cells.

**Why the three state bits cannot do this job, and why saying so is a defect report rather than a
preference.** `stunning_look_plan.md` §4.6 says depth is *"the three persisted bits that already exist …
diggable and collidable on the ordinary edit path"*. That cannot hold. A collider is looked up as
`BlockTypeId → ShapeId → ColliderTemplate`; state bits are not in that chain. Making three state bits
collidable requires either sub-cell collision geometry — the second occupancy authority the composite rule
forbids outright — or a collider keyed on state, which breaks `ShapeDef.volume_micro_m3` as the single
field from which mass, integrity, buoyancy and structural load are all derived. The clean resolution costs
**eight shape rows at 28 B = 224 bytes** against a `ShapeDef` budget of 1.75 KiB at 64 rows, and collision
then comes from the existing four lanes with no new code.

Consequence: the three `snow_depth` state bits are **reclaimed**, taking the reserve from 5 bits to 8 —
direct relief to the fight for a material-interpreted state byte, and where L3's four bits come from.

> **THE RULE THAT KEEPS SNOW INSIDE THE COLLIDER BUDGET, with the number attached.**
>
> **Snow depth may only ever use the box family — `Slab`, `Plate`, `Panel` — never a sloped shape.**
> Snow is the only item on the owner's list that wants *full ground coverage* rather than sparse
> placement, and shaped cells are the expensive lane. Over the 64 m physics disc (π·64² = 12,868 m²):
>
> | Form | Cost | Share of the ≤ 27 MB per-cluster budget |
> |---|---|---|
> | Individually shaped (sloped), spherical lane at ~800 B/cell | **10.29 MB** | **38%** |
> | Box family, merged at the 8-cell greedy cap (201 boxes × ~500 B) | **101 KB** | 0.4% |
>
> **A 102× swing.** Axis-aligned boxes merge under the greedy predicate exactly like flat ground, so
> uniform snow costs what flat ground costs; `Wedge`, `CornerOut`, `RampLow` and `Tetra` do not merge.
> This must be a **build-gate assertion on the snow material's admissible shape set**, not a convention,
> because one drift material with a sloped shape added in a later patch silently reopens it.

**Catch-up over dormancy is class 1 for the baseline and class 2 for the chase**, and both degenerate to a
*delete* pass over any long sleep. The baseline is a closed-form climate evaluation with nothing to
integrate. The chase — the block field converging one layer at a time toward that baseline — steps each
cell independently under the existing random-tick scheduler, so a binomial draw over elapsed sleep gives
the exact number of steps, clamped to the ≤ 8-layer distance. After any sleep longer than a few minutes
the mean vastly exceeds 8, so catch-up becomes *"set every cell to its baseline and prune every diverged
record"*. **Snow makes a sleeping chunk smaller, not bigger.** Storage: baseline 0 bytes; divergence 8 B
per cell in the existing WAL, and a shovelled 20×20 m path is 400 records = 3.2 KB that prunes as the snow
refills.

### 6.3 L3 MARK — the footprint, the rut, the compaction

**A mark is block state on an existing cell.** Four bits of downward-only displacement, in the same field
that already carries `wetness` and `burnt`, persisted by the same mask, replicated by the same per-tick
per-subscriber chunk-state message. It is **not** a new object, **not** a new record type, and **not** a
second occupancy authority — and that last is proved rather than asserted, by the gate in §9.4.

**It is not decoration, and the existing rule already says so.** The promotion boundary states: *"two
players may see different decoration DENSITY; they can never see different decoration FACTS, because at no
rung does decoration carry a fact."* A footprint carries a fact — someone was here, going that way,
recently — so decoration's local tier does not reach it. The split is: the **fact** (cell, quantised
displacement, when) is server-owned; the **art** (which boot, the tread, the crisp rim, the crystal spray)
carries no fact and is derived locally from a hash.

**Cost, at the pathological density.** An entry is `local_index 18 | displacement 4` = 22 bits → 3 bytes,
the same shape as the damage lane's existing entry. A player at 6 m/s makes ~8 quantum-changing entries
per second. At 128 co-located runners, all mutually subscribed: 3,072 B/s of unique data, so **3.07 kB/s
per player** and 393 kB/s of realm egress — **≈7.7% of the ~40 kB/s per-player snapshot budget**, on
**zero new message types and zero new inter-shard arms**. The realistic case (128 players spread over a
region, ~10 within the mark radius) is 240 B/s, 0.6%.

**The per-mark timestamp is not optional.** The client renders on a 100–150 ms buffer with no prediction,
so a naively-rendered mark appears where the wheel *actually is* rather than where the client is drawing
it: 0.9 m ahead at a 6 m/s walk, **4.5 m at 30 m/s**, 15 m at 100 m/s. Carrying `made_rel` and rendering
the mark against the same interpolated clock the body is rendered against puts the rut exactly under the
wheel. The same field removes every decay message: decay is a linear closed-form ramp from `made_at`,
linear for the same determinism reason the wetness drying already is.

**The radius is derived, not chosen.** The detail knob is an angle, `cell_angle_max = 1.273e-3 rad`. A
footprint is a ~0.3 m relief feature and is discriminable at two pixels out to `0.3 / (2 × 1.273e-3)` =
**118 m**. The render-side buffer is one dense byte per surface cell: 3,844 B per column, padded, 24
columns resident with hysteresis = **90 KiB** — three orders of magnitude below the 320 MiB of render
targets.

**Parallax is what makes compaction affordable, and it is the load-bearing insight of this whole
section.** A greedy-merged 62 m quad has four vertices and cannot show a 30 cm dip in its middle, so the
mark must be a per-pixel parallax perturbation — which the surface-craft design has already ruled is the
shipped technique for sub-cell relief. Two consequences. The mesher never sees the mark, so it is **not a
merge barrier** and the chunk mesh stays the camera-free pure function the skirt ruling requires. And the
drawn snow surface is already lower *before* the promotion remeshes the chunk, so **the remesh is visually
a no-op and can be deferred arbitrarily**. Without that, 128 players compacting snow produce up to 128
chunk-dirty events per second at 480 KB of remesh upload each — **61 MB/s**. With a 1,000 ms coalesce
window it is one deferred remesh per marked column per second: 5.5 MB/s at the current 88-byte quad and
**0.46 MB/s at the packed 8-byte format** — a third independent argument for pulling the packed vertex
format forward to P6.

**Promotion closes the loop.** A mark that reaches its quantum ceiling spends itself into an ordinary block
edit — `SnowLayer6 → SnowLayer5` — and resets to zero. That is the only path from mark to block, it is the
ordinary edit path, and it carries an ordinary fence. *Marks below one shape-catalogue step are state;
marks that cross one are edits.*

**Upward is banned at the type level, not by convention.** A downward-only mark obeys the standing rule
that the visual sits at or below the collision surface. Snow *piling up*, mud ridges beside a rut and sand
berms all violate it, so accumulation must always be an edit (a new layer block) and never a displacement
value. The mark's sign is a constraint on the type.

**Minecraft's sinking detail, delivered without a lie.** Minecraft ships a deliberate one-layer mismatch —
five layers collide as a slab, four only look like one — so *"the player is always slightly sunken in the
snow"*. That mismatch is a shape whose collider differs from its visual, which this ruling forbids. We get
the same feel from the **mark**: you press a dent into the surface where you stepped, you see the dent, you
feel the drag, and the collider is exactly the layer you are standing on. Better, and inside the contract.

### 6.4 Movement cost is a separate field from friction, and conflating them silently fails

`friction_q12` is the Coulomb coefficient the **rigid-body solver** uses. The character is a
`KinematicCharacterController`: it moves by a commanded translation and resolves penetration, and **it does
not read collider friction at all**. So "snow is slow" cannot be delivered by `friction_q12` and must be a
separate control-side property the server multiplies into the commanded speed.

Three new fields on the substance row, six bytes on a 118-byte row that already pads to 128 — **table cost
literally zero**: `traction_q12` (control), `deform_q12` (how much one unit of normal impulse displaces the
surface, derivable from `porosity_q12` and `density_kg_m3`, so the compaction threshold is *derived* rather
than typed), and `recover_s_q8`. Vehicles keep `friction_q12` because they are dynamic bodies with real
solver contacts. **One field is physics, one is control, and they must never be merged.** The depth penalty
needs no fourth field: a snow layer's `volume_units` already says how deep you are wading.

**And it is free on the wire and free on the client**, because the server runs the controller and there is
no prediction. Variable-traction movement is the classic prediction-desync source in other engines; here it
costs one integer read.

### 6.5 One byte and one recovery rate serve six surfaces

| Surface | Cover | Substrate | Mark | New content |
|---|---|---|---|---|
| Snow | closed-form | `SnowLayer1..8` | compression | 1 material row |
| Ash | closed-form | **the same shape family** | compression | 1 material row, 0 shapes |
| Mud | — | — | long `recover_s` (~30 min) | soil at `wetness ≥ 2` |
| Sand | — | — | short `recover_s` (~60 s) | 0 |
| Dust | closed-form | — | shallow | 0 |
| Flattened grass | — | — | **already designed** — the decoration override's per-slot bits are already labelled *harvested / trampled / mown* | 0 |

Water splashes are **not** marks — they are stateless events, the shape precipitation already ships. Fold
the splash into the footstep: one `SurfaceImpact { cell, family, impulse_class, attenuation }` = 4 bytes
drives the sound and the particle burst together.

**Footstep sound has worse fan-out than the marks and must be culled harder.** A mark is emitted only when
its quantised value *changes*, so re-treading a cell is silent on the wire; a footstep is an event on every
footfall. At 128 × 8/s × 4 B × 128 subscribers that is **4.1 kB/s per player**, a third more than the mark
lane. It is affordable only because it is short-range: audible to ~30 m, and π·30² against the mark
radius's π·118² is **15.5× less area**, so a realistic ~8 audible sources cost 256 B/s. It must also be
rate-limited per source, and it must be server-authored — the client is a pure renderer, and a
client-derived footstep is silent to everyone else, which defeats stealth entirely.

---

## 7. Ask 4 — the canopy that takes more than the square

### 7.1 The overhang is already delivered, in cells, and it is enormous

A composite tree's footprint is not the player's core. The planted footprint spans a **121 m² plan area**
over roughly **1 m² of trunk** — a plan-area ratio of about **120:1**. Measured against typical stylised
tree proportions (8 m tall, 6–8 m canopy diameter, 0.5–0.8 m trunk) the natural ratio is 78–140:1, so the
design's figure sits in the right band for a large broadleaf. **That ratio *is* "it takes more than the
square, so it looks believable"**, and it is delivered by real, collidable, persisted cells rather than by
mesh overhang.

What remains is the small question: how far may the drawn mesh sit outside those cells. The answer is:
**it may not.**

### 7.2 The contradiction, and which side wins

Two binding documents written the same day say opposite things:

- The composite design's containment gate permits *"no solid part of the art mesh outside the footprint
  dilated by **≤ 1.00 m laterally and downward**"*.
- The shape design's ruling hands the render/collide divergence budget the other way: *"the visual hull is
  a strict subset of the collision hull … the failure direction is therefore always 'a chipped-looking edge
  that still holds you up', never 'you fall through what looks solid'"*.

They cannot both hold. Worse, the containment gate's *"nothing at all above the top plane of the topmost
solid footprint cell"* clause is **per-composite**, so on an 8 m canopy a bough drawn at 6 m height and
overhanging 1.0 m over an empty column is *below* the top plane and therefore legal — and it is an
upward-facing surface a player will try to land on with nothing underneath. That is failure mode 1 from the
eight the standing law exists to prevent, arriving at 1.00 m where the law priced 0.20 m.

> **RULING. The visual hull is a subset of the collision hull, everywhere, for every archetype, with one
> bounded downward-only exception.** The composite design's lateral dilation clause is **struck**. This
> ruling sides with the shape design against the composite design, and with the adversary against the
> overhang investigation.

### 7.3 The mechanism: the footprint becomes CONSERVATIVE, and it costs no art discipline at all

`FootprintRule` gains exactly two representable values and only one is legal:

- **`Conservative`** — occupancy **> 0** claims the cell. **Mandatory for every archetype.**
- **`Majority`** — occupancy ≥ 0.50. Retained in the type for the bake tool's diagnostics only; rejected by
  the table validator.

Under `Conservative`, `render_hull ⊆ collider_hull` holds **by construction**, with no art constraint, no
convention for an artist to remember, and no gate that can be argued about at review time. The composite
design's fill threshold of 0.50 is superseded; the *"nothing above the top plane"* clause becomes redundant
rather than contradictory, because the topmost claimed cell contains the topmost mesh point by definition.

**The one thing it breaks, and the instrument that must be replaced.** The coverage gate demands the art
mesh's silhouette cover ≥ 92% of the footprint silhouette from eight directions. Under `Conservative` the
footprint is *deliberately* larger, so a 5.7 m mesh inside a 6.2 m footprint scores mechanically
(5.7/6.2)² = **84.5%** and every rounded archetype fails the build on day one. The gate's *purpose* is
sound — the original finding was ~13 m² of collider you cannot see — but its *instrument* is now wrong.
Replace it with the dual:

> **No footprint cell may lie further than one cell from the art mesh's surface.**

Satisfied by construction under `Conservative`, checkable from the same rasterisation the tool already
runs, and impossible to game. Keep the coverage percentage as a **reported diagnostic** in the bake
manifest so a regression still names the archetype that got thinner.

### 7.4 The number that reconciles two documents that disagree by 19.4×

The adversary found that the reference tree has **two published cell counts** and that the composite
figure fails the provenance design's own content-build gate by 3.07×. He is right, and it invalidates the
collider, storage, felling and anti-conjuration arithmetic on both sides. The reconciliation is exact and
falls out of the geometry:

| Source | Cells | Status |
|---|---|---|
| Provenance design's reference oak (12 trunk + 150 leaf) | **162** | A *sparse* Minecraft-shaped canopy. Incompatible with the composite design's own solid-canopy requirement (fill ≥ 0.50 **and** simply connected). |
| Composite design's §3.4 / §3.6 | **3,146** | **Fails gate G6 (≤ 1,024 cells) by 3.07× at content build.** Cannot fit the falling-group blob, which is documented as *exactly* 1,024 × 8 B — so as published, **a composite tree cannot fall at all**, which deletes the provenance ruling's own headline example. |
| **This ruling** | **≈ 998** | Derived, and it reconciles everything. |

The derivation, and it is exact:

```
art mesh canopy radius            5.7 m
conservative footprint radius     5.7 + 0.5 = 6.2 m           (half-cell voxelisation residue)
footprint plan area               π · 6.2²   = 120.8 m²        == the published 121 m²
footprint cells (solid ball)      4/3 π 6.2³ = 998 cells       <= feature_max_cells = 1,024
falling-group blob                998 × 8 B  = 7,984 B         <= max_state_bytes  = 8,192 B
                                                                (208 B spare — exactly enough
                                                                 for the 4-byte recipe tag)
```

The published 121 m² footprint and the published 1,024-cell cap **agree, to 2.5%**, on a canopy of ~998
cells. Neither 162 nor 3,146 was ever consistent with both.

**Four numbers move with it, all in the safe direction:**

| Quantity | Published | Corrected |
|---|---|---|
| One-shot footprint write | 25.2 KB | **8.0 KB** (against a 33.5 KB worst-case crater — comfortable) |
| Anti-conjuration yield | 3.7 blocks/m²/day | **1.18** blocks/m²/day |
| 100 m plantation yield | ~37,000 blocks/day | **~11,800** blocks/day |
| Forested chunk feature-cell share | 2.6% *or* 50.7%, depending which document you read | **~26%** of a 62³ chunk |

The last of these is the economy's number, and it is 3.1× smaller than the one the economy design has been
handed. **It must be re-issued to that design, not quietly corrected here.**

### 7.5 The art constraint this creates, and it is the hardest one in the document

Under `Conservative`, a member thinner than a cell claims a **full cell** of collider for a sliver of mesh:
a 0.3 m branch arm produces a metre-wide invisible obstacle along its whole length, and inflates the cell
budget toward the 1,024 cap. Under `Majority` the same member claims **no** cell at all and is 4 m of
geometry with nothing behind it. **Both footprint rules punish thin members, for opposite reasons.**

> **Composite members must be authored ≥ 1 m thick, or not authored.**

That is a hard rejection of a recognisable class of bought asset — gnarled oaks, dead trees, anything with
visible separate branch arms. A round-canopy or conifer archetype passes; a windswept dead pine does not.
Nobody has opened a purchased pack to find out what fraction is affected; §10 owes that measurement.

### 7.6 The one exception, and why it is the *safe* direction

A composite's cells are excluded from the chunk mesh's emit at tier 0, so the composite's art mesh is
**not** rim-warped — but the terrain around its base **is**, inward by up to `A_max` = 0.20 m. The terrain
face directly under the trunk is culled (occupancy is not excluded), so there is no gap there; the visible
seam is the base ring one cell out, where a warped terrain face meets an unwarped trunk base.

> **A composite's art mesh may extend DOWNWARD below its footprint's bottom plane by ≤ `A_max` = 0.20 m,
> and in no other direction.** Derived from the existing warp budget, not authored. The mesh goes *into*
> the ground, never above it, which is the direction the standing law already calls harmless.

---

## 8. Ask 5 — vehicles, and the problem that is not collision

**This ruling does not design vehicles.** It answers the owner's collision question, states the constraint
vehicles place on *this* design, and reserves what is free now and expensive later.

### 8.1 Server-authoritative collision is already true, and promotion is what makes it true for free

The owner's sentence — *"the server should be able to understand what decoration was generated and do
collision calculations"* — is satisfied by construction under the rule, because the server never has to
*understand what decoration was generated*: everything collidable is world, and the server owns the world.
There is no second list of obstacles to reconcile, no derivation to run per body, no parity gate on the
physics path.

### 8.2 Why a derived collider is not merely expensive — it is unbuildable

The cost argument only kills half of it, and anyone who prices only the grass will reach the wrong
conclusion. Priced honestly over the 64 m physics disc:

| Tier | Population | Collider | Share of ≤ 27 MB | Derivation cost |
|---|---|---|---|---|
| **Micro** (grass, 100/m²) | 1,286,796 | **149 MB** | **553%** | 13–21 ms per body per evaluation, against a 50 ms tick |
| **Prop** (bush/boulder, 0.135/m² shipped) | 1,737 | **0.61 MB** | **2.3%** | 0.21 ms once per chunk, on the task pool |
| Prop at the authored ceiling (0.5/m²) | 6,434 | 2.96 MB | 11% | as above |

**The prop tier is affordable.** That is exactly why the rejection must be recorded as a **correctness**
ruling and not a budget one — otherwise someone re-runs this arithmetic in a year, finds it cheap, and
reopens it. Three reasons that survive the absence of any cost argument:

1. **The falloff proof has the wrong polarity for collision.** Property 5 — `S(0) = 1.0`, so the server's
   sampler at the plateau returns *"the union of every client's set"* — is the **safe** direction for a
   concealment sensor and exactly inverted for a collider: the server would collide bushes a driver at
   150 m is not being shown. Evaluating at the observer's rung is not available either, because a collider
   is not per-observer — two players in the same vehicle at different camera distances would need different
   colliders, and there is no such object.
2. **The residency governor makes the drawn set a function of one GPU's last-frame memory pressure.** It
   reads last frame's instance count and impostor coverage and lowers the ranges at 0.08/s with a 0.10
   hysteresis band; its shipped caps differ per hardware tier (bush range 120 m / 178 m / 260 m). A collider
   that exists on a 3060 and not on a 1060 is not a collider, and no gate reconciles the two because the
   divergence is designed in and correct for its actual purpose.
3. **The parity gate that would have to become blocking structurally cannot be.** The pre-merge parity test
   runs on a software adapter and the design says in its own words that it *"cannot catch backend codegen
   divergence, because it exercises a different WGSL compiler from the shipping one"*. The real gate is
   *"real hardware, scheduled not pre-merge"*. Correctly rated when the output is a blade's lean; a phantom
   wall on one backend when it is a collider.

There is also a fourth, quieter reason: **the client never needs a collider at all**, because there is no
prediction. "Evaluated identically on server and client" is over-specified from the start.

### 8.3 The standing rule this generalises

> **Anything the simulation reads is computed on the CPU, in Rust, in a Tier-A crate. The WGSL twin is a
> presentation of it, never the source.**

Abundance satisfies this (the CPU builds the lattice and uploads 4 bytes per point; the shader's q8 bilerp
is integer and exactly reproducible). Anything that does not satisfy it can never become a physics input.
That closes the cross-rung determinism question permanently rather than per feature.

### 8.4 What a vehicle actually needs from a surface

Not colliders. A wheel cast is a BVH descent at ~1–3 µs; sixteen casts per tick per vehicle is 32 µs =
**0.06% of a 50 ms tick**. What it needs is four things a walker does not, and three of them fail silently
if not designed in:

1. **Swept sampling, not point sampling.** At 30 m/s a wheel crosses 1.5 cells per 20 Hz tick and can
   change contact material between ticks. ≤ 2 cells per wheel per tick at 30 m/s, ≤ 7 at 100 m/s — trivially
   cheap, and it aliases invisibly if omitted.
2. **Per-cell solver friction**, which means the collider builder must not merge boxes across a friction
   boundary. Quantise friction into ~8 classes so the barrier fires rarely. **This is a new argument against
   the recommended Cartesian cube lane that was not in that decision's table when it was written**: parry's
   `Voxels` is one collider with one material and has no per-voxel friction, so that lane needs one
   `Voxels` per friction class (sparse, so memory is roughly unchanged, plus 8 BVH roots) or the greedy-box
   fallback, which carries per-box material naturally.
3. **Suspension travel exceeding the largest sub-cell discontinuity.** The rim warp is ≤ 0.20 m and a snow
   layer step is 0.125 m, so **0.325 m is a derived floor** on any vehicle's suspension rather than a
   number somebody types.
4. **Surface roughness as a point sampler, not a collider.** `vd_decor::query::surface_roughness(cell) ->
   RoughnessQ8`, evaluated server-side per wheel per tick, feeding a quantised suspension impulse and a
   tyre/dust cue. Four wheels at 20 Hz at ~200 integer ops is **4 µs per vehicle per second** — 0.0004% of
   a tick, 0.04% for a hundred vehicles. The alternative, a live derived collider set per vehicle, is
   514 KB resident plus 192 inserts and 192 removes per second at 30 m/s, per cluster. **The value must be
   integer/fixed-point**, because it changes a physics trajectory and therefore crosses a physics→control
   boundary; a float sampler would be a determinism break that only manifests as divergent vehicle
   trajectories across hosts.

### 8.5 The finding that matters more than any of this

**A one-metre cube grid serves a walking player well and a wheeled vehicle badly, for geometric reasons,
and no amount of collision engineering fixes it.**

The whole character controller is derived from *"a player can always clear exactly one cell"*:
`jump_clear` = 1024 + 16 + 16 FINE = one cell + skin + margin. Generated terrain is `Cube`-only at every
rung, so a natural hillside is a staircase with exactly **1.000 m** risers and a tread of `1/grade` metres.
A vehicle has no jump:

| Grade | Wall spacing | At 30 m/s |
|---|---|---|
| 10% | 10.0 m | a 1 m wall every **333 ms** |
| 20% | 5.0 m | every 167 ms |
| 30% | 3.33 m | every 111 ms |
| 45° | 1.00 m | every 33 ms |

A rigid driven wheel cannot climb a step approaching ~0.4–0.5 of its own radius, so a 1.000 m riser needs a
**4–5 m diameter wheel** — a mining truck, not a rover. Hover and tracked locomotion are immune because
both average the surface over a contact patch: four repulsor casts under a 4 m hull smooth a 1 m staircase
to about 0.25 m, which is why hovering is the shipped answer in every voxel game.

And the interpolation buffer converges on the same conclusion from the other side: at 30 m/s a vehicle
renders **3.0–4.5 m** behind authority against 0.50–0.75 m for a walk, so every obstacle at issue — a 0.3 m
stone, a 1.000 m riser, a 2 m boulder — has an along-track extent *below* the lag distance and its
collision response arrives where the driver already sees nothing. Note honestly that this indicts the metre
grid exactly as much as it indicts derived decoration: a 0.5 m `Nub` fares identically. What it argues is
that a vehicle's response to sub-vehicle-scale geometry must be a **suspension compliance** — a low-pass
filter, where lag reads as a heavy ride — and never a hull impulse, where lag reads as a bug.

**A forest is a solid maze at canopy height**, and this is a consequence of the composite rule working
correctly rather than a defect. Canopy cells are real cells and collide, exactly as Minecraft's leaves do.
Mean free path `1/(ρ(w + 2r))` at one tree per 100 m² with a 6.2 m canopy: **7.7 m** for a 0.6 m player,
**6.9 m** for a 2 m rover — a canopy contact every 0.23 s at 30 m/s. Trunks alone give 62.5 m and 33.3 m.
**The entire difficulty is the height of the lowest canopy cell**, which is currently an art choice with no
gate. It needs a `canopy_clear_m` bake column, and it is the same lever as the collider cost: a hollow
canopy is cheaper to collide and drivable, a solid one is neither.

---

## 9. What the server owns, and what it costs

All numbers against **VD-REF** — RTX 3060 12 GB / Ryzen 5 5600X / 32 GB, 2560×1440, 15.00 ms of owned
frame, 1.32 ms of reserve after the look package — and the 50 ms shard tick at 20 Hz.

### 9.1 The server owns everything, and the list is short because the answer is "the block field"

| Question | Where the answer comes from |
|---|---|
| Does this collide? | the tier-0 block field. No second source exists. |
| What is its mass, its integrity, its yield? | `volume_units` on the tier-0 cell |
| Can a body rest here? | the collider built from those cells |
| How rough is this surface? | `surface_roughness(cell)` — a sampled integer, never an instance |
| How much foliage is here? | `sample_foliage_density(cell, at)` — a sampled integer |
| How deep is the snow? | the `SnowLayer` shape on the cell |
| How compacted is it? | four bits of block state on the cell |
| How slow is it to walk on? | `traction_q12` on the substance row |
| Was someone here? | the mark's timestamp |

### 9.2 Frame

| Line | ms | Share of the 1.32 ms reserve |
|---|---|---|
| Stone class, near + mid art mesh rungs (0.005/m², ≤ 200 tri near / ≤ 60 tri mid, ×2 passes) | **0.033** | 2.5% |
| Stone class, C1 cube rung 150 m → 786 m (9,351 instances at a ≤ 16-quad ceiling, 20% frustum) | **0.080** | 6.1% |
| Abundance bilerp in compute pass A (+4 reads, 3 lerps ≈ 20 int ops per decorated cell, against ~80 existing) | **0.010** | inside decoration's own 0.15 ms reserve |
| Mark parallax (one buffer read per top-face pixel) | **0.010** | inside the booked parallax line |
| **Mandatory total** | **0.13** | **10%** |
| *Optional:* body bend (L1) | *0.05* | *3.8%* |

**Conditioned on one thing.** §6.4's booked 0.35 ms forest line assumes a per-chunk rung bias that triggers
on *measured quad density* — and the composite emit-exclusion **removes exactly those quads**, so the bias
has nothing to fire on. If it never fires, the forest line is **1.28 ms against 0.35 ms booked** and 0.93 ms
of the 1.32 ms reserve is gone before a single stone is promoted, at which point the mandatory 0.13 ms
above is 33% of what remains. The adversary found this and it is a genuine self-contradiction in a settled
document. **[UD-12]** resolves it.

### 9.3 Memory

| Item | Cost |
|---|---|
| Rock archetypes added to the composite catalogue (8 × 4 variants × 5 rungs) | **+1.9 MiB** |
| Abundance lattice, GPU (324 B/chunk × 114 resident tables) | **36 KiB** |
| Abundance kin-column map, CPU (62² u8 × 114) | **428 KiB** |
| Mark buffer, 118 m radius, 24 columns with hysteresis | **90 KiB** |
| Ramp tables, density-scaling row growth, influencer uniform | **3 KiB** |
| **Total** | **≈ 2.4 MiB**, i.e. **1.2%** of the 0.19 GiB spare at the current vertex format |

### 9.4 Collider — this is the real cost, and the published figure of "zero" is an accounting error

`stunning_look_plan.md` §3.8 states *"**Colliders: zero** — the collider is the block field."* That is a
category error, not a saving: the block field's collider cost goes **up**, because the tree's cells are
solid where air was. Corrected, per 64 m physics cluster (9 surface chunk columns = 34,596 m², 346 trees at
one per 100 m²):

| Contributor | Collider | Share of ≤ 27 MB |
|---|---|---|
| Bare terrain, 9 surface chunks at 100 KB–1 MB | 0.9–9.0 MB | 3–33% |
| **Composite forest** (346 trees × 40 / 65 / 130 boxes × ~500 B) | **6.9 / 11.2 / 22.5 MB** | **26 / 41 / 83%** |
| Stone class at 0.005/m² (173 stones × ~3 boxes) | 0.26 MB | 1.0% |
| Snow, box family, merged | 0.10 MB | 0.4% |
| *Snow, if a sloped shape is ever admitted* | *10.29 MB* | *38%* |
| **Forested + snowy total, central estimate** | **12–20 MB** | **44–75%** |

Two things follow and both must be recorded rather than absorbed. A **forested surface chunk is
1.35–2.25 MB against the published ≤ 1 MB per-chunk worst case** — and that per-chunk figure is the number
the entire 27 MB budget is derived from. And the box-count-per-canopy estimate (40–130, central 65) is the
weakest number in this document; it is one afternoon's spike to settle (rasterise a canopy at 998 cells,
run the 8-cell-capped greedy decomposition, count) and it is owed before the composite lane ships.

### 9.5 Generation, and the honest headroom

Measured: 0.885 ms per 62³ surface chunk, 12.19 ns per noise evaluation — **on an Apple M4 Pro**. The gate
is 1.20 ms *"on the reference machine"*, which is the Ryzen 5 5600X, and `bench-gen-reference` is owed at
P4.1.

| Item | ms | % of 0.885 |
|---|---|---|
| Composite planting, dense forest (already booked) | 0.060 | 7.0% |
| Stone placement hash, 3,844 columns × 12.19 ns | 0.047 | 5.3% |
| Snow climate term, coarse and interpolated (~10 int ops/column) | 0.0135 | 1.5% |
| **Chunk total** | **0.885 → 1.005 ms** | **+13.6%** |
| **Remaining headroom to the 1.20 ms gate** | **1.194×** | |

**The reference CPU may be at most 19.3% slower than the M4 Pro on scalar dependent-chain code. It will
not be.** So `bench-gen-reference` runs **before** any of this generation spend is committed, not after.
There is a lever if it comes back badly, and it is already documented: the cave field's coarse-lattice step
from 4 to 8 buys a further **33% of the 0.625 ms cave line = 0.205 ms**, which is 23% of the chunk and more
than covers the +0.121 ms above — at the cost of visibly rounding tunnel walls. The trap to refuse
explicitly is evaluating the snow term *per column* instead of interpolating the coarse field: 3,844 ×
12.19 ns = 0.047 ms, +5.3%, for a quantity whose real spatial frequency is 64 m.

### 9.6 Shard tick and wire

| Item | Cost |
|---|---|
| Vehicle roughness sampler, 4 wheels × 20 Hz | 4 µs/vehicle/s = **0.0004% of a tick**; 100 vehicles = 0.04% |
| Foliage-density query, 128 sessions × 20 Hz × ~80 ns | **0.0004 ms/tick** |
| Marks, 128 co-located runners | **3.07 kB/s per player** (7.7% of budget), 128 msgs/tick (6.7% of the measured 1,920/tick peak) |
| Marks, realistic (~10 in radius) | 240 B/s (0.6%) |
| Marks, 128 co-located **vehicles** at 30 m/s | 23 kB/s per player — **58% of budget** |
| Footstep sound, 128 co-located | 4.1 kB/s per player (10%); ~8 audible = 256 B/s |
| Abundance, stones, snow baseline, cover | **0 bytes** |

The vehicle case must be bounded by a **shed-loud governor on the quantum, not a rate limiter**: coarsen
the displacement from 4 bits to 2 rather than drop entries, because a dropped fact desynchronises two
observers while a coarser one is a density change, which the promotion boundary already rules legitimate.

### 9.7 Gates this ruling owes

| Gate | What it must hold | First runs |
|---|---|---|
| `surface_no_second_authority` | With every mark at maximum and every mark cleared: collider construction, rigid-body mass properties, every raycast result, the structural-support flood, the pressurisation room graph and every mount test are **bit-identical**. The mechanical proof that a mark is not an occupancy authority | P6 |
| `composite_footprint_conservative` | Every shipped variant's footprint is the **conservative** voxelisation; **no footprint cell lies further than one cell from the mesh surface**; the mesh has no solid part outside the footprint except ≤ 0.20 m downward | build gate, P4 |
| `composite_cell_budget` | Every variant's conservative footprint ≤ `feature_max_cells` = 1,024, and its blob ≤ `max_state_bytes` − the recipe tag | build gate, P4 |
| `snow_shape_family` | The snow and ash materials' admissible shape set is the box family only | build gate, P4 |
| `A_max_ceiling` | `A_mat ≤ A_max = 0.20 m` for every registry row | build gate, P4 |
| `decor_abundance_golden` | A pinned 62³ field + kin table produces the full 9×9 lattice byte-exactly on two target-cpu builds; the reciprocal equals the reference division for every count in `0..=area` | P4 |
| `decor_abundance_sensitivity` | A single-cell edit moves size and density by ≤ `ceil(range/(2R+1)²)` over a randomised corpus | P4 |
| `decor_promotion_coarse` | Promoting a class does not change the tier-3 silhouette beyond a stated cell count | P4 |
| `surface_mark_golden` | Byte-identical linear decay over a pinned corpus on two target-cpu builds | P6 |
| `surface_anywhere` | Walk, mark, decay, promote, shovel — the identical fixture on a **Spherical planet** profile and a **Cartesian ship** profile, with a forced re-anchor. Real: a station's snow-covered landing pad, ash on a burnt deck | P6 |
| `decor_abundance_anywhere` | 400 soil cells on a planet and 400 hull-moss cells on a ship interior with the identical rule row produce equal lattices and equal instance sets; forced re-anchor is pixel-identical | P6 |
| `surface_mark_fanout` | The 128-session fixture, every session walking, asserts bytes/tick and msgs/tick against §9.6 | P6 |
| `surface_dormancy` | Sleep past the longest `recover_s`, spin up: zero surviving marks, and a record set equal to the diverged-from-baseline set | P6 |
| `bench-collider-forest` | The per-chunk and per-cluster collider figures in §9.4, on both geometry lanes | **before P5 commits** |
| `bench-gen-reference` | The generation figures in §9.5, on the reference CPU | **P4.1, before any generation spend** |
| `composite_order_anchor_invariant` | The winning recipe and variant are identical before and after a forced `reanchor()` | P4 |

---

## 10. The divergence contract

Stated as a deliberate, bounded, tested property — not as an accident, and not as a tolerance somebody
picks at review time.

> ### WHAT A PLAYER MAY SEE AND NOT TOUCH
>
> **Nothing solid. Ever. At any distance, on any machine.**
>
> The only things drawn that are not touchable are things that are *obviously* not touchable — grass,
> pebbles, ripples, spray, the visual half of a footprint — and they are drawn as what they are.

> ### WHAT A PLAYER MAY TOUCH AND NOT SEE
>
> | Cause | Bound | Direction | Why it is the safe side |
> |---|---|---|---|
> | Conservative footprint residue on a composite | **≤ 1 cell laterally** | outward from the mesh | you brush the edge of a canopy that looked like air |
> | `crown_tolerance` on an upward-facing triangle | **≤ `A_max` = 0.20 m** by default, per-archetype tightenable to `offset` = 1/64 m for drivable classes | collider above the mesh | you stand slightly above what you can see |
> | Terrain rim warp | **≤ `A_max` = 0.20 m** | inward | a chipped-looking edge that still holds you up |
> | Snow mark | **≤ 4 quanta**, downward only, never upward | visual below the collider | you press a dent and stand on the layer |
>
> **All of these are the same direction: phantom hit, never phantom miss.**

**The composition rule, which nobody had written and which the adversary was right to demand.** Two of
those budgets were each justified alone and nobody stated what happens when they meet:

> **The total visual-below-collider band on any upward-facing surface a body can rest on is
> ≤ `A_max` = 0.20 m, from all causes combined.** One number, one build assertion.

Under this ruling the two budgets never in fact apply to the same triangle — a composite's cells are
excluded from the chunk mesh's emit at tier 0, so a composite triangle is never rim-warped — so the
worst-case stack the adversary computed (0.53125 + 0.20 = 0.731 m, which is 1.38× the autostep height and
41% of a character) **cannot occur**. The rule is stated anyway, because "it happens not to compose today"
is exactly how a shipped physics bug is born.

### 10.1 The two consequences worth quantifying

**Shots that graze a canopy.** Under `Conservative`, a 5.7 m mesh inside a 6.2 m footprint gives
1 − (5.7/6.2)² = **15.5% of shots that visually clip the canopy edge stop on something invisible** — and
**0% pass through what looks solid**. The alternative (`Majority`) inverts both figures. The owner's own
sentence — *"All of it must collide"* — chooses this side, and so does the standing law.

**The selection outline must be the CELL, never the mesh — and this is not optional.** With the cubes not
emitted at tier 0, the block highlight becomes the player's **only** cue to where one cell ends and the
next begins, and he needs it constantly: chopping is cell-by-cell, and the recognition predicate's
first-failure-wins behaviour means removing the wrong cell can drop the composite's rendering entirely and
reveal the cubes. So the selection outline, the placement preview and the reach test are all defined over
the tier-0 cell, exactly as the eight readers already are, and **a composite mesh is never a selection
target**. This is free — it is what falls out of doing nothing special — but "make the outline hug the
pretty mesh" is the obvious wrong instinct and it lands as a two-line polish commit.

### 10.2 The rungs must agree, and one pair currently agrees only by luck

The near art rung, the baked cube rung and the coarsened chunk mesh are three drawings of the same cells.
The first two are authored and baked under a quad ceiling; the third is a pure integer function of the
octant masks of the cells actually present. **There is no mechanism by which the second and third are
equal**, and the only assertion between them is a perceptual no-pop threshold — the one cross-rung claim in
the whole design asserted that way, where every other is asserted exactly.

> **RULING: bake `cubes_t0/t1/t2` FROM the coarsener, not from the art mesh.** Run `coarsen` over the
> variant's footprint at bake and greedy-merge the result. The cube rungs are then by construction the same
> silhouette the chunk mesh will produce at the rung above, and the ≤ 128 / ≤ 48 / ≤ 16 quad ceilings become
> statements about the coarsener's output rather than hopes about an artist's. **Before the first archetype
> is baked.**

---

## 11. The asset pipeline

### 11.1 What this ruling constrains about an authored mesh

| # | Constraint | Derived from |
|---|---|---|
| 1 | The mesh is authored **inside** its conservative footprint. For a canopy: mesh radius ≈ footprint radius − 0.5 m | the half-cell voxelisation residue |
| 2 | **No member thinner than ~1 m.** Both footprint rules punish thin members, for opposite reasons (§7.5) | the 1 m lattice |
| 3 | The conservative footprint is ≤ **1,024 cells**, and the blob is ≤ 8,192 B − the recipe tag | `feature_max_cells`, `max_state_bytes` |
| 4 | The canopy is **solid and simply connected** — which is what forces the ~998-cell figure and what lets the far form meet its quad ceiling | the far-form quad ceiling |
| 5 | Every **upward-facing** triangle sits ≤ `crown_tolerance` below the collider plane above it. For a rock, a log or any drivable class this means a **flat, lattice-snapped crown with all the roundness in the flanks** | the divergence contract |
| 6 | `canopy_clear_m` is a bake column, asserted against whatever ground-vehicle height is eventually chosen | §8.5 |
| 7 | Yaw-only placement. The upward/non-upward triangle classification is baked once and is well-defined **only because** composite placement is yaw-only, so local up and realm up coincide on a planet and inside a ship alike | the recipe's yaw mode |
| 8 | Every archetype carries a fixed pivot and bounds convention (for the impostor bake) and an authored feature height (for the rung rule) | the existing decoration decisions |

Item 7 needs one sentence written down before somebody allows a wall-mounted composite: **if a hanging vine
or a ceiling-mounted cluster is ever permitted, the classification must be recomputed per orientation
across the 48 octahedral transforms at bake** — 48× the gate cost, still entirely offline, still zero
runtime. Reserving that is one sentence in the table-freeze rule; retrofitting it is a re-bake of every
shipped archetype behind a world-format epoch.

### 11.2 The re-authoring cost this adds

The art-direction contract's re-authoring checklist gains an eleventh step (flatten and snap the crown;
thicken or delete thin members) at roughly 0.3–0.6 h per archetype. At 64 archetypes that is **+19–38
hours, or +€950–1,900** on the €8,000–16,600 already booked — about **+12%**. Small, but it lands on the
option whose entire appeal is that it is cheap and fast.

### 11.3 The measurement that is owed and has never been taken

Nobody has opened a purchased pack. The pipeline section already owes a triangle-count and texture
inventory before any pipeline decision; **this ruling adds one column to that same pass: measure branch
member thickness.** If a meaningful fraction of a nature pack's trees are gnarled or dead-branched, that
fraction is unusable as composites without re-modelling, and that is a schedule fact rather than a design
one.

### 11.4 Two positions that do not move

**The licence position is unchanged and absolute.** Source files never leave the team, never enter a public
repository or artifact, and **never go to any AI tool**; no bake stage may use a learned upscaler. A baked
asset inside the client ships normally.

**One clause is now touched twice and should be watched.** The composite art meshes for the stone class are
pack-derived, and if a stone ever becomes a player-placeable *inventory item* it moves adjacent to the
EULA's prohibited "Game Creation Software" use. The standing recommendation is "no for the first release",
and this ruling does not change it: **a rank-1 stone composite is player-CONSTRUCTIBLE from ordinary stone
blocks, which is a different thing from placeable pack art, and it should stay that way.**

**The rock class specifically should stay on CC0 scan-based sources.** That is already the standing
recommendation, on the grounds that photogrammetric rocks blend into a PBR voxel world where stylised ones
do not, and this ruling reinforces it: the rock class is the one being promoted to real, collidable,
mineable world, so it is the one whose look is hardest to walk back.

---

## 12. What must be reserved now for vehicles

Without designing vehicles. Each is cheap today and a migration later.

| # | Reserve | Where | Free now / cost later |
|---|---|---|---|
| **V-R1** | **`physics_substeps: u8`**, and re-derive `snap_to_ground` from `step_dt` rather than `tick_dt` | the operational tuning struct | One field. At 20 Hz a 30 m/s body moves **1.5 m per step** and tunnels 1 m features; wheeled sims run 60–240 Hz. Later: re-derives every character-controller constant and every quantisation grid on a shipped world |
| **V-R2** | **`collider_build_budget_us_per_tick`** and the benchmark behind it | the same struct; `bench-collider-build` | The collider **build rate** is what a vehicle stresses: 0.73 chunk-colliders/s walking, **4.35/s at 30 m/s**, with a 19-chunk burst at a corner crossing. Per-chunk build cost is nowhere in the design, and the derived ground-speed ceiling lands anywhere between **82 and 344 m/s** depending on it. There are three collider gates and **no build-time gate at all** |
| **V-R3** | **A ground-query seam with ADMISSION control**, not deferral | the sim's budget family | Copying the radar-probe budget's shape would fail in the wrong direction: an over-budget probe defers in round-robin order and sets a stale flag; **an over-budget wheel cast cannot defer — a wheel with no ground answer falls.** So the budget caps vehicle *count* (refused at build time), never queries per tick |
| **V-R4** | **`surface_roughness(cell) -> RoughnessQ8`** — integer, q8, never a float | `vd_decor::query` | One unused argument now. Later: a signature change in a 100%-covered Tier-A crate and every caller — exactly the cost already paid to reserve the concealment observer argument. A float here is a determinism break that only manifests as divergent vehicle trajectories across hosts |
| **V-R5** | **`crown_tolerance_m` and `canopy_clear_m` as per-archetype bake columns** | the composite recipe table | **ART one-way door.** Every archetype is authored to the convention, so it must be fixed before the first archetype is authored |
| **V-R6** | **The generated-surface-skin shape as a generator-config field, defaulting to `Cube`, named in the P4 terrain-golden digest** | the generation tuning struct | **This is the single thing that makes wheeled ground vehicles possible** (§8.5), and turning it on later changes generated terrain under existing player builds — the same one-way door as the noise-lattice step. One field and one line in a fixture today; a world-format epoch later |
| **V-R7** | **The interpolation buffer as a per-context tuning field**, default unchanged | the same struct | At 30 m/s the buffer is 3.0–4.5 m of lag against 0.50–0.75 m walking. Per-context lets driving run at 100 ms while walking runs at 150 ms, recovering 1.5 m at zero architectural cost. **This is not a request to relax the no-prediction law** — it is one operational parameter in one config struct, which the no-magic-numbers rule already requires |
| **V-R8** | **The 4-byte `(recipe, variant, orient)` tag on the falling-group blob** | the entity registry | The blob is documented as *exactly* `feature_max_cells × 8 B` and has zero spare at the published tree size; at the corrected 998 cells it has **208 B**. Reserve it with the entity-kind row at P6 or it is a persisted-registry migration |
| **V-R9** | **`SurfaceContract::{Rigid, Permeable}`** as one field on the substance row, consumed only at bake | the substance registry | One enum beside the rim-warp amplitude. After the first recipe row ships it is inside the table-freeze rule |

---

## 13. The decision register

Each row states a **default** that ships if the owner says nothing, so nothing is blocked.

| # | Decision | Options | Recommendation | One-way door | Cost of deferring |
|---|---|---|---|---|---|
| **UD-1** | **Grass never collides — confirm or overrule.** | (a) Confirm, as a standing rule rather than an implementation detail. (b) Overrule. | **(a).** Six shipped titles and one engine documenting it as a hard product limitation, against one counter-example that is a complaint thread. If overruled, the least-bad form is a taller derived grass with a movement-cost field — **never** a cell without a collider, which is forbidden by the rule | No | Not colliding grass is reversible; colliding it is not. The cost of being wrong is asymmetric |
| **UD-2** | **The collision threshold.** | (a) **Representability** — the catalogue's smallest extent (`Panel` = 0.125 m), because our controller auto-steps and Valheim's does not. (b) Step height (0.53125 m). (c) A full cell | **(a)**, and it supersedes the collision investigation's own recommendation of (b). Derived from an existing table, not a new number | No | One comparison against one existing table |
| **UD-3** | **Pin the tree's cell count and re-issue the numbers.** | 162 / 3,146 / **≈ 998** | **≈ 998** — the only value consistent with the 121 m² footprint, the 1,024-cell gate, the 8,192 B blob and the solid-canopy rule simultaneously (§7.4). Re-derive from it: the collider budget, the footprint write, the anti-conjuration bound, the felling table, the deforestation figure | **YES in ART** — every archetype is modelled to whichever answer wins | It blocks everything else here, and the economy has been handed a yield figure that is 3.1× too high |
| **UD-4** | **The footprint rule and the overhang convention.** | (a) `Conservative` everywhere, mesh strictly inside, ≤ 0.20 m downward only. (b) `Majority` on foliage with ≤ 1.00 m lateral art overhang | **(a).** It satisfies the owner's own sentence, it makes `render ⊆ collide` hold with no art discipline, and it resolves two settled documents that contradict each other. Replace the 92% coverage ratio with the per-cell distance bound in the same commit | **YES — ART.** Before the first archetype is authored | Re-modelling every archetype at 2.5–5.2 h each |
| **UD-5** | **`A_max` = 0.20 m as a build-asserted registry ceiling.** | (a) Assert it; the rock class reads as an outcrop through the composite lane. (b) Let the rock row raise it | **(a).** (b) taxes grass, dirt, sand and everything else at the coarse rungs where the ladder's whole saving lives, and it silently reopens a decision the owner already answered | Effectively yes | The merge win is lost a rung at a time and nobody notices which change did it |
| **UD-6** | **Build the body bend** (grass parting around a moving body). | (a) Yes, behind a tuning toggle. (b) No | **(a)**, with the explicit rule that if the frame benchmark reports the terrain-opaque line needs its full allowance, this is the **first thing cut**. 0.05 ms budgeted, 0.006–0.032 ms expected | No | It is the only visible part of "grass interacts with the player" that is not free |
| **UD-7** | **Does deep foliage slow you down?** | (a) Yes, 0–12% at the micro tier. (b) No | **(a).** 0.0004 ms per tick at 128 sessions, no collider, no new state, and it is what makes grass feel physical rather than painted on. Requires adopting the identity-vs-influence sharpening in §5.4 | No | Makes decoration gameplay-relevant for the first time; adjacent to the concealment decision and should be answered with it |
| **UD-8** | **Does snow become a block family, and at what step?** | (a) **8 layers at 1/8 m** — Minecraft's quantisation, matches the existing 3-bit width, 8 shape rows = 224 B. (b) 16 layers at 1/16 m. (c) Not a block — state bits only | **(a).** (c) requires either sub-cell collision geometry or a state-keyed collider, both of which break the composite rule and the mass derivation | **YES** — `ShapeId` is persisted, so a later step change halves every saved world's snow unless epoch-gated | Reserve eight contiguous shape ids now; it costs nothing today |
| **UD-9** | **Is the snow/climate baseline seasonal?** | (a) Time-invariant. (b) **Seasonal via an integer season index**, widening the P4 DoD from `f(seed)` to `f(seed, season_index)`. (c) Continuous time — refused | **(b).** It is a **determinism-contract door**: it changes a stated Definition of Done and must not be taken silently. It is also what the dormant-world pillar and the economy's forestry and agriculture professions both need | No, but it changes a DoD | One extra fixture parameter now; a re-argued DoD later |
| **UD-10** | **Are marks server-authoritative, client-local, or hybrid?** | (a) Client-local decals: 0 B/s, no shared trails, no relog survival, no trail query. (b) Fully detailed server-owned (10 B/mark): 10.24 kB/s per player, 26% of budget, for art the client can hash. (c) **Hybrid** — server owns 4 bits of displacement, client renders the tread: **3.07 kB/s**, 7.7%, zero new message types | **(c).** It is (a)'s fidelity with (b)'s facts, and the existing fact-versus-density rule already decides it | The wire entry shape is (see UD-11) | Without it a tracker cannot follow a trail and marks do not survive a relog |
| **UD-11** | **Widen the chunk-state entry NOW so damage stages and marks share one message.** | Today `(local_index u18, stage u3)` = 21 bits → 3 B. Proposed `(local_index u18, kind u2, value u4)` = **exactly 24 bits → exactly 3 B** | **Take it.** Same width, same message, same batching, and it is free before the bulk chunk-delta lane is routable | **YES — WIRE.** Afterwards it is a protocol bump plus a client migration | Exactly the retrofit the design elsewhere refuses to pay |
| **UD-12** | **Resolve the forest rung-bias contradiction** — 0.93 ms of the 1.32 ms reserve depends on it. | (a) Trigger the bias on **composite instance count** as well as chunk quad density — one extra per-chunk counter. (b) Drop the bias and book 1.28 ms. (c) State C1's outer end explicitly instead of deriving it from the chunk's tier-0 range | **(a) + (c).** The bias needs the composite signal to work at all, and C1's extent should be a stated range rather than a side effect of a governor that now moves when players fell trees — otherwise adjacent chunks cross at different distances and the boundary walks through a forest as it is logged | No | The whole frame reserve, including everything in §9.2 |
| **UD-13** | **The abundance neighbourhood radius.** | R = 4 m (0.73 µs, a 9 m patch reads as a full meadow) / **R = 8 m** (2.6 µs, 1/289 sensitivity, a 20 m meadow saturates at its centre) / R = 16 m (9.8 µs, 49% of the typical edit budget) | **R = 8 m.** Per-row field | No | It decides how a meadow reads |
| **UD-14** | **Do surface marks persist across a realm spin-down?** | (a) No — they expire within the recovery window anyway. (b) Yes, in a per-chunk table written inside the WAL's own transaction, 5 B/entry, self-pruning | **Reserve the table name now, populate at P6/P7.** Reserving is free; adding a persisted record type after the WAL and the compactor are written is not, and the phase's DoD already requires surviving a hard kill and a re-shard | No if reserved | A relog inside the window costs a player his tracks |
| **UD-15** | **Does per-cell friction change the recommended Cartesian cube lane?** | (a) Split into one voxel collider per friction class (~8, sparse, plus 8 BVH roots), keeping the recommendation. (b) Take the greedy-box fallback on all four lanes, which carries per-box material naturally | **(a)** — but this is a **new argument that was not in that decision's table when it was written**, and it should be re-put to the owner alongside the cross-lane classification spike that already gates P5 | No | The spike is already owed; this decides what it must cover |
| **UD-16** | **The locomotion class of the first vehicle — decide BEFORE the terrain golden is pinned.** | (a) **Hover** — needs nothing new; four casts under a hull smooth a 1 m staircase to ~0.25 m. (b) Tracked. (c) Wheeled at speed — requires V-R6 from the first generated world | **(a) first, (b) second, reserve for (c).** (a) is the only option with zero terrain consequences, and choosing it does not foreclose (c) provided V-R6 is reserved | The terrain half is **YES** | Deciding after the terrain golden is pinned means changing generated terrain under existing player builds |

---

## 14. Adjudicated objections

Every adversary finding, ruled. No deferrals.

| # | Finding | Ruling |
|---|---|---|
| **A1** | **The tree has two published sizes, 19.4× apart; the composite figure fails its own content gate by 3.07× and therefore cannot fall** | **UPHELD, and it is the most consequential finding in the run.** Resolved to **≈ 998 cells** by a derivation that reconciles the 121 m² footprint, the 1,024-cell gate and the 8,192 B blob simultaneously (§7.4). Four downstream numbers corrected, all in the safe direction. The economy's yield figure must be re-issued |
| **A2** | **Felling destroys the composite** — a detached group's cells leave the block field, so the predicate cannot see them and the falling tree draws as cubes, on the *intended* harvest path | **UPHELD.** Resolved by the rule's second arm: during a topple a composite is an **entity**, and the composite rule does not reach it. Carry a 4-byte recipe tag on the blob (V-R8); the collider still comes from the detached cells' tier-0 occupancy, so no second authority appears |
| **A3** | **"Colliders: zero" is an accounting error** | **UPHELD.** Re-derived in §9.4: **12–20 MB of the 27 MB per-cluster budget** in dense forest, and a forested surface chunk is **1.35–2.25×** the published per-chunk worst case that the whole budget derives from. His box-per-canopy estimate is the weakest link and is owed a spike (`bench-collider-forest`) |
| **A4** | **The frame reserve is already spent, because the forest rung bias triggers on a signal composites delete** | **UPHELD.** A genuine self-contradiction in a settled document. 0.93 ms of the 1.32 ms reserve depends on it, and everything in §9.2 is quoted against that reserve. **[UD-12]** |
| **A5** | **The generation budget has 19.3% of headroom and the denominator is the wrong machine** | **UPHELD**, and the sequencing consequence is adopted: `bench-gen-reference` runs **before** any generation spend. One thing he did not mention is added: the coarse-lattice step from 4 to 8 buys 0.205 ms, which more than covers the +0.121 ms if the reference CPU comes back badly |
| **A6** | **Composite ordering is not anchor-invariant, and the gate that would catch it does not assert what it needs to** | **UPHELD and adopted.** Order by the anchor-invariant global cell (the address decoration already uses), and extend the existing anywhere-fixture to assert **recipe identity** across the re-anchor it already forces. One line, free today |
| **A7** | **The rock half raises a registry-wide maximum that five derived quantities read** | **UPHELD and ruled.** `A_max` becomes a build-asserted ceiling at 0.20 m; the outcrop look moves to the composite lane at zero cost to the warp (§4.3, **[UD-5]**) |
| **A8** | **The client's decoration cost is gated per edit and never per second, and breaches the main-thread contract by 3–21× at the design's own density** | **UPHELD.** The per-edit gate and the per-frame contract are two different quantities used interchangeably. The missing tuning value — the per-session edit-rate cap — is *named* in the block-store tuning struct and never given a number, and that number is what decides whether the meadow case ships. Out of scope to set here; recorded as a blocking risk |
| **A9** | **Promoted objects GROW with distance, 2× per rung, and at pebble density 96% of tier-3 surface cells are affected** | **UPHELD**, and it *supports* the ruling rather than undermining it: it is a **third independent argument** for the representability floor, arriving from a direction none of the investigations took. Correct behaviour at boulder density (27%, and a boulder genuinely is a multi-cell feature). Owes `decor_promotion_coarse` |
| **A10** | **`Majority` fails the owner's literal ask; `Conservative` fails in the opposite direction; the band is ±0.5 m and a sign must be chosen** | **UPHELD, and decided for `Conservative`** — siding with the adversary against the overhang investigation. The owner's sentence chooses it, the standing law chooses it, and it makes `render ⊆ collide` hold with no art discipline. His corollary — that the 92% coverage ratio then fails every rounded archetype at 84.5% and must be replaced by a per-cell distance bound — is also adopted (§7.3) |
| **A11** | **Two separately-justified divergence budgets compose to 0.731 m of standable air, 1.38× the autostep height** | **UPHELD IN PRINCIPLE, DISSOLVED IN FACT.** Nobody had written the composition rule and it must exist. Under this ruling the two never apply to the same triangle — a composite's cells are excluded from the chunk mesh's emit, so a composite triangle is never rim-warped — so the stack cannot occur. **The rule is written anyway** (§10), because "it happens not to compose today" is how a shipped physics bug is born |
| **A12** | **The C1 and C2 rungs are two unreconciled derivations of the same cells, asserted perceptually where everything else is asserted exactly** | **UPHELD and adopted.** Bake the cube forms **from the coarsener**, not from the art mesh, so they agree by construction and the quad ceilings become statements about the coarsener (§10.2). Before the first archetype is baked |
| **A13** | **The composite materialisation path is an unbounded matter printer for inorganic recipes, and the obvious gate is banned by an invariant** | **UPHELD.** Adopted as `footprint == core` for every inorganic recipe (§4.4), one assertion. On the wall-becomes-boulder exploit: his proposed fix (read provenance) is **refused** — the provenance design's ban stands and this ruling does not relax it — and replaced by an **isolation clause** on the recipe, which is pure pattern, needs no second reader of a durable field, and is strictly cheaper |
| **A14** | **A forest is a solid maze at canopy height and nothing says how high the lowest canopy cell must be** | **UPHELD.** Mean free path 6.9 m for a 2 m rover, a contact every 0.23 s at 30 m/s, against 33 m for trunks alone. `canopy_clear_m` becomes a bake column (V-R5), and it is the same lever as the collider cost |
| **A15** | **The 27 MB budget is per connected component of the cluster overlap graph, and the merge/split rule has not been designed** | **UPHELD and recorded.** Merging is required (two colliders for the same cells in one physics world double every contact impulse), and 128 players 60 m apart form one component spanning 7.6 km. Not this ruling's to fix — but this ruling takes the per-chunk figure from ≤ 1 MB to 2.3 MB, so it gets 2.3× worse at once |
| **A16** | **A projectile drags a 64 m collider bubble; the radius was derived against swept motion and never against body size** | **UPHELD and recorded.** 116 chunk-colliders/s at 800 m/s, and the same rule caps a player-built ship at ~118 m before its nose leaves its own bubble. Same 2.3× multiplication |
| **A17** | **The gravity direction can be zero, and every up-derived rule has an undefined branch in a zero-g ship interior** | **UPHELD.** It reaches this ruling directly: the upward/non-upward triangle classification, the mark's downward-only sign, and snow's "up" all need a defined up. A **declared per-realm deck normal** is the answer and nothing declares one. Recorded as a blocking risk for the ship arm |
| **A18** | **"Expensive" and "feels bad" are being conflated, and the industry's reason for the first does not transfer to us** | **UPHELD and adopted as the framing of §8.2.** Their reason is that the derivation lives on the GPU and is unrepresentable to the CPU; ours is integer and CPU-derivable at O(1). So the cost argument does **not** bind and the correctness arguments do. Recorded as a correctness ruling for exactly the reason he gives: otherwise someone re-runs the arithmetic in a year, finds props affordable, and reopens it |
| **A19** | **He could not break the rule, and it should be said** | **UPHELD**, with his amendment adopted in full: the rule is stated with **both arms** — a real block **or** a real entity — because five things ride the entity arm and would otherwise be rebuilt badly (§3). Two of his secondary observations are folded in: sub-catalogue objects have no representation and correctly cannot collide, and water is not a counter-example because buoyancy is a volume integral over the block field rather than a contact |
| **A20** | **The 40 kB/s snapshot budget is inferred, not measured** | **UPHELD as the largest unmeasured denominator on this path.** Every wire percentage in §9.6 rides on it. If the real budget is 20 kB/s, marks are 15% and footstep sound is 20% at the co-located density, and the shed governor becomes load-bearing rather than belt-and-braces. Pin it before the mark path ships |

### 14.1 Where an investigation was overruled

Four places, recorded so the reasoning is traceable:

1. **The collision line's step-height threshold (0.53125 m) is replaced by representability**, because our
   character controller auto-steps and Valheim's does not — so the Valheim failure mode, which is the whole
   evidential basis for the threshold, cannot occur here regardless of collider height (§3).
2. **Its fallback for an overruled grass decision — "a promoted block with no collider" — is refused
   outright.** A cell that exists for storage but not physics is the forbidden object read backwards (§3).
3. **Its recommendation to source abundance from the edit pyramid is refused**, in favour of the density
   investigation's own box filter, because the pyramid's dominant-substance field is a **mode** and a mode
   has unbounded single-edit sensitivity — the threshold failure reintroduced through the back door (§5.2).
4. **The surface-response investigation's Minecraft snow-collider mismatch is refused** — a shape whose
   collider differs from its visual is the second occupancy authority — and the same *feel* is delivered by
   the mark instead, which puts the visual below the collider and stays inside the divergence contract
   (§6.3).

### 14.2 The one thing this ruling cannot answer

**Whether the world is drivable.** §8.5 is a geometry finding, not a collision finding, and it is the
largest consequence of the owner's vehicle sentence. The metre grid and the cube-only generator together
make a hillside a staircase of 1.000 m risers, which a person clears by design and a wheel cannot. Three
honest exits exist, one of them (**V-R6**) has to be reserved before the first planet is saved, and the
choice between them is **[UD-16]**.
