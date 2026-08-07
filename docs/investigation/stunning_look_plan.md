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

# Voxeldust — Making a one-metre block world look stunning, inside the budgets

**Status:** design, awaiting owner decisions. Not yet binding.
**Date:** 2026-08-03.
**Question this answers:** *"investigate what we can do to make it look STUNNING while having blocks,
considering all our performance optimisations of course."*

---

## Front matter

### What this document decides

It decides **what we build, in what order, to make a world made of metre cubes genuinely beautiful —
without spending a millisecond or a megabyte that the block design has not already measured.** Every
recommendation carries its cost against a line that already exists in the block design's frame and
memory tables. Where a line does not exist, this document says so, states where the time comes from,
and books it.

It also discharges owner ruling **R4** (trees and rocks made of several blocks, all of it collidable),
which the rulings table records as OPEN and which the note under it correctly refuses to fold into the
decoration layer.

### How it was produced

Four independent design studies were written — the composite subsystem, the look backlog, the frame and
memory budget, and the art-direction contract — then attacked by an independent adversary who produced
twenty-two findings, three of them rated critical. This document is the adjudication: it takes the
studies as source material, resolves every disagreement between them, rules on every adversarial
finding (§0 below), and re-derives the budget from the corrected numbers. **Where the adversary is
right, the correction is applied and credited. Where he is wrong, it is said in one line.**
Engine claims were re-verified against the vendored crate sources in the local registry, not against
documentation.

Two of the four studies proposed designs that this document **overturns rather than corrects**: the
composite subsystem's "draw the art asset on top of the cubes" model, and the water design's
camera-dependent wave range. Both overturns are argued in place.

### How it relates to the main design and its two addenda

`docs/investigation/block_system_design.md` and its two addenda remain the binding design. This document does not
replace any section of it. It:

- **Discharges** ruling R4 (§3 here), which the main design leaves open;
- **Corrects** three published numbers in the main design (§6.6 here lists them, with the arithmetic);
- **Spends** the 1.20 ms sky line and the 1.50 ms post line, which the main design allocates and never
  specifies, and part of the 1.82 ms slack;
- **Quantifies and closes** deferred entry D-DEC-12 (the dense-forest triangle load), which the main
  design registers as unsolved and routes to the rendering section without an owner;
- **Discharges** deferred entry D-DEC-14 (no shadows past 150 m against a 100 km view) with a
  mechanism, a millisecond and a phase;
- **Adds** twenty-one owner decisions, deduplicated against the main register in §8.

Numbering here is local (`§1`…`§10`, decisions `SL-n`). References to the main design use its own
numbering (`§5.9`, `[USER DECISION 5-C]`, `D-DEC-12`, `R4`).

---

## §0. The adversary, adjudicated

Stated first, because three of the studies' headline numbers do not survive it and everything below is
built on the corrected ones.

| # | Finding | Ruling |
|---|---|---|
| **C1** | "One denominator" is a 1.78× multiplier on *residency*, not just fill; the corrected totals are 4.07 GiB against a 3.0 GiB cap and 17.8 ms against 15.00 ms | **Upheld in arithmetic, overturned in conclusion.** The arithmetic is exactly right and both budget papers quoted 1080p residency against a 1440p frame. But the fix is neither "soften the far field" nor "blow the budget": the ladder's detail knob must be stated as an **angle**, not as pixels, and then it is resolution-independent by construction. See §6.1. **The published residency, terrain-memory and geometry numbers are all correct as published**; only the fill lines move with resolution, and the main design's frame table is already stated at 1440p. Cost of the fix: one restatement, zero milliseconds, zero quality |
| **C2** | The composite mesh is hidden inside the cubes it is meant to beautify; the silhouette never stops being blocky | **Upheld, and it is the most important finding in the run.** The composite study's core model is overturned. §3 replaces it with a model in which the composite's cells are never *drawn* at the near rung — while remaining fully real for collision, mass, raycast and persistence. His proposed fix (suppress faces inside the mesh range) is *also* rejected, on the design's own rule that a chunk mesh may not be a function of the camera; the correct suppression key is the chunk's **tier**, which is camera-free |
| **C3** | The composite frame budget omits fill, and the study's own model gives 4.7–6.4 ms | **Half upheld.** He is right that the study priced geometry only. He is wrong about the number: he applied the *impostor* denominator — four-tap **alpha-tested** fill — to geometry that is opaque. A bought low-poly canopy is solid, not a cutout card, so occluded fragments are rejected by the depth test at raster rate instead of being shaded. The correct reading inverts his conclusion: **composites are the only thing in this design that makes a dense forest affordable**, because they replace roughly twelve overlapping layers of alpha-cutout leaf cube with one opaque surface. §3.7 and §6.4 |
| **H1** | The rock half routes around the "generated terrain is cubes only" ruling and costs 21–131× the triangles | **Upheld, and his fix is adopted and improved.** A per-cell rounded template is rejected. The rock half is delivered by the existing rim-warp machinery with a **per-material amplitude and a per-corner hash direction** — already O(perimeter), already baked, already proven contained, already collapsing to nothing at coarse rungs. Cost: zero. §3.2 |
| **H2** | Composite forests multiply the unresolved dense-forest triangle load by 4× | **Upheld in direction, resolved in this document.** The load is real, it is not the composite lane's creation (it is registered as D-DEC-12 and is present with or without R4), and it was never budgeted. §6.4 quantifies it at ~1.3 ms unbudgeted, then closes it to ~0.35 ms with the fix the main design itself proposed and left unowned |
| **H3** | Three documents, three different composite budgets, none derived from the others | **Upheld.** Resolved here to one set of numbers, in §3.7 and §6.3 |
| **H4** | The anti-conjuration invariant is defeated by its own recommended tuning value ("minutes, not days") | **Upheld.** "Minutes" contradicts the project's own ruling R8, which frames growth as slow and memoryless across server downtime. The bound is restated as **area × time** and quantified in §3.6 |
| **H5** | Recognition is not deterministic under its own stated total order | **Upheld; both halves of his fix adopted** (identifier tie-break, and maturity as instance state rather than a second recipe row). §3.4 |
| **H6** | Server-side buoyancy on a wave form built from sine and cosine breaks the determinism rule | **Upheld.** §4.5 rules the server-side surface-height function integer/polynomial, with the vertex shader evaluating the same polynomial |
| **M1** | The Mie correction is right in kind and wrong in number | **Upheld and verified in the vendored source.** `bevy_pbr-0.18.1/src/medium.rs` sets Mie `absorption: 3.996e-6, scattering: 0.444e-6`. The standard set is scattering 3.996e-6, extinction 4.44e-6, therefore absorption 0.444e-6. **The two values are exactly transposed.** The fix is to swap them and nothing else; the look study's proposed 4.40e-6 absorption would have frozen a 2× error into a pinned test |
| **M2** | Aerial perspective is ranked first for a view this planet does not have from the ground | **Upheld in fact, overstated in consequence.** The horizon at eye height on a 161,671 m body is 741 m — confirmed, it is the main design's own number. But "six of eight rungs draw nothing" is wrong: a 4 km peak is visible from the ground at 36.7 km, and a 100 m ridge at 6.3 km. The far field from the ground is not empty, it is **sparse and dramatic** — and that is a look worth having, which aerial perspective is what sells. The re-rank is applied anyway: sky-visibility ambient and the shadow horizon move above aerial perspective, because they change the frame the owner will screenshot first |
| **M3** | Chunk-load recognition is ~12× understated and the mitigation does not exist on a shard | **Upheld, and dissolved by the §3 redesign.** Generated composites are planted by the generator as ordinary generated cells, so there is nothing to recognise on chunk load for the 99.99% of chunks nobody has edited |
| **M4** | The resolved-chunk accessor is claimed free and is not | **Upheld, and also dissolved by the §3 redesign.** There is no occupancy overlay in the corrected model, so there is nothing to splat and nothing to binary-search |
| **M5** | The water merge cap applies at every distance, not inside 256 m; and the wave amplitude has a visible ring at 256 m | **Upheld — a 150× miss and a pop on the largest smooth surface in the game.** Both are fixed by keying the cap and the amplitude ramp to the **chunk's tier** rather than to camera distance, which removes the ring entirely by reusing the crossfade band that already exists. §4.5 |
| **M6** | The art contract is all ceilings and reducers; nothing in it adds chroma | **Upheld; his fix adopted and extended.** A per-biome chroma **floor**, a coloured sky-ambient term so shadow contrast is a *hue* contrast, and the macro layer promoted to a hue authority. §7.3 |
| **M7** | The contract's own gates make raw greybox assets unshippable | **Upheld.** Resolved with a named, dated, expiring exemption list that warns until its date and then fails the build — the same shape as the coverage-exemption file the project already runs. §7.5 |
| **M8** | Composite shadows are box-shaped | **Upheld and dissolved.** In the corrected model the cubes are not drawn at the near rung, so they cannot be the caster; the composite casts from its own mesh. The 1.6× cascade cost comes back onto the bill and is paid for by shortening the authored shadow horizon. §6.3 |
| **M9** | Pinned image hashes will not hold across two GPU vendors | **Upheld.** Image gates use a tolerance metric (per-channel mean and 95th-percentile deviation plus a structural term); byte-identity gates are reserved for the integer CPU paths. §7.5 |
| **M10** | Three simultaneous canopy representations inside 59 m, with nothing to suppress two of them | **Upheld and dissolved.** In the corrected model a composite owns its cells' *rendering*, so the leaf cubes are not drawn and the decoration layer, which reads the same resolved view, emits no cluster cards on them. One canopy, always |
| **L1** | Realm and shard boundaries are never mentioned | **Dissolved.** With no occupancy overlay, a composite is blocks; a composite straddling a seam is blocks straddling a seam, which the block design already answers |
| **L2** | The composite cost estimate for the demand scheduler is circular | **Dissolved.** Composites are generated cells; a realm's cost is its cells |
| **L3** | The invisible-wall bound is understated | **Upheld.** The coverage gate tightens: the canopy footprint must be simply connected and its fill threshold rises from 0.35 to 0.50, which is also what makes the far form meet its quad ceiling. §3.5 |
| **L4** | The render-origin float claim is right in conclusion and wrong in two numbers | **Upheld.** The single-precision step at 161,671 m is 0.0156 m, not 0.0125; relative precision is 6e-8, not 8e-6. The conclusion — reserve the pose now — is the best single recommendation in the run and is ranked second here |
| **L5** | The pit-versus-ridge gate cannot pass at P4 on its stated mechanism | **Upheld.** The gate moves to P6 and a weaker P4 form (sky-tinted hemispherical ambient, no occlusion) is asserted instead |
| **L6** | Per-block value jitter has no distance ramp and shimmers in the far field | **Upheld.** The jitter amplitude fades across the same crossfade band as everything else — consumer number six on a list that already exists |

Two things the adversary got right that are worth restating as *positive* findings, because they are
the two cheapest wins in the document: the **render origin must become a pose**, and the **starfield
must stop being 2,800 depth-writing meshes** before any atmosphere is switched on. Both verified in
source. Both cost nothing.

---

## §1. The look, in one page

*This page has no engineering vocabulary in it on purpose. It is the page to read if you read nothing
else.*

Stand on the starter planet at the end of a clear afternoon. The ground is made of metre cubes and you
can count them at ten paces — that is deliberate, it is the identity of the game, and nothing here
tries to hide it. But the world does not look like a stack of boxes, for six reasons.

**The light is real, and it comes from the sky rather than from a switch.** Today the engine lights
every surface with the same flat glow, so the floor of a pit is exactly as bright as the top of a
ridge and the world reads as cardboard. We delete that. Every surface is instead lit by *how much sky
it can see*. A hollow you dug is dark. An overhang has a soft gradient under it. A cave is genuinely
black, so a torch becomes an event rather than a decoration. And because the sky is blue and the sun
is warm, a shaded cube face is *blue*, not merely dark — which is the difference between a shadow that
reads as beautiful and one that reads as missing information.

**There is air between you and the distance.** The sky is computed from the planet's own atmosphere
rather than painted, so each world gets its own colour of daylight and its own sunset out of its seed,
with no artist involved. A ridge far away is pale and low-contrast because there are kilometres of air
in front of it. On this planet the ground horizon is only about seven hundred metres away, so what you
see beyond it is not a landscape — it is mountain tops rising over the edge of the world, and it is the
haze between you and them that makes that read as *far* instead of as a painted backdrop. From the air
it is the whole view.

**Shadows reach the horizon.** Today's engine casts shadows for a hundred and fifty metres and gives up.
We keep proper shadows close in, where they carry contact and shape, and beyond that we compute the
shadow of the land directly from the same mathematics that produced the land — so a mountain twenty
kilometres away lays its shadow across the valley in front of it, for almost no cost, and it can never
disagree with the mountain because it *is* the mountain. Add clouds drifting their shadows across a
hillside and the single most damaging thing about a big blocky landscape — every part of it being the
same brightness — is gone.

**Trees are trees and boulders are boulders.** Put a few wood blocks down and a real tree grows there —
a proper trunk and canopy from a bought art pack, not a green cube. Put more down and you get a bigger
one. Put rock blocks down and they become a rounded outcrop rather than a staircase, and put two
together and a real boulder appears on them. All of it collides, exactly as you asked, because
underneath the drawn shape the blocks are still there and the blocks are still what you walk into.
The important part, which is not obvious: this is not just prettier, it is *cheaper*. A forest drawn as
individual leaf cubes is the single most expensive thing in this world; drawn as trees it costs a
quarter as much memory and less time. The feature pays for itself.

**Surfaces have a history.** Dirt does not have razor edges — every exposed corner in the world is
already nudged inward a little, irregularly, so a cut bank looks cut rather than machined. On top of
that, crevices go dark and damp, exposed corners go pale and worn, faces that never see the sun go
green with moss and faces that always do go bleached. None of that costs a single extra texture
lookup: the world already computes every one of those facts for other reasons, and we are simply
reading them. Rain darkens sand hard and steel barely at all, because how much water a material drinks
is already a property of the material. Snow lies on the mountains from ten kilometres away and can be
shovelled off your path.

**Everything moves together.** One wind function drives the grass, the rain's slant, the drift of the
clouds and the direction of the swell, so a storm never blows east while the grass leans north. The
sea rolls where it is wide and lies flat in a pond, from nothing but the size of the water. It costs
essentially nothing and it is the cheapest way to buy the feeling that the world is alive.

The whole of that fits in the time the design has already set aside and never allocated, plus about a
millisecond and a half of genuine new spend, and it leaves a reserve. The one thing it needs that
nobody has agreed yet is a decision about how the picture is *finished* — the film-response curve and
the exposure — because that choice changes what every material must be painted like, and taking it
after a hundred materials exist means painting them again.

**The three-word version, for use at a glance: PHYSICAL, BLOCKY, OLD.** Anything that fails one of the
three is out of contract.

---

## §2. The ranked backlog

Ranked by beauty per millisecond at the reference machine (RTX 3060 class, 2560×1440, 60 Hz, 15.00 ms
of owned frame). "Claim" says where the time comes from: *sky* and *post* are lines the main design has
already allocated and never specified; *in-line* is added to the terrain shading line inside its
existing allowance; *slack* is a genuinely new claim on the 1.75 ms of corrected average slack.

| # | Item | What the player sees change | ms | MiB | Claim | Phase | Depends on | One-way | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **Sky-visibility ambient replaces the flat ambient light** | Every cube face in every frame stops being uniformly lit. Pits, overhangs and north faces gain real depth; caves go genuinely black | **0.10** | 0 | in-line | P4 partial, P6 full | occupancy slab (P6) | No | **TAKE FIRST** |
| **2** | **Render origin becomes a rigid pose, server-authored** | Nothing on its own. It is what makes items 3 and 5 *correct on a sphere* instead of correct at one point | 0.00 | 0 | — | plant P4 | — | **YES — wire ABI** | **TAKE NOW.** Only item that gets more expensive every week |
| **3** | **Starfield becomes a sky-pass term, not 2,800 meshes** | Stars stop being fogged by two kilometres of air; they fade in physically as the sky thins on the way up | **−0.05** | 0–11 | sky | P4, **before** item 5 | — | No | **TAKE.** Prerequisite, not a follow-up |
| **4** | **Long-range shadow horizon from the terrain's own function** | Mountains cast shadows into valleys at 20 km. Terrain past 150 m stops being a flat swatch | **0.05** | 4 | slack | P4 | coarse height field | No | **TAKE.** Discharges D-DEC-14 |
| **5** | **Atmosphere and aerial perspective, per body from its seed** | Distance reads as distance; every planet gets its own daylight and sunset with no art | **0.70** | 1.9 | sky | P4 | items 2, 3 | No | **TAKE** |
| **6** | **Composite trees and boulders (ruling R4), near mesh rungs** | A cluster of wood blocks is a tree, not a green column. Rock clusters are boulders. All collidable | **0.43** | 8 | slack | P6 | §3 in full | Yes — recipe table | **TAKE.** Also saves 420 MiB — see item 7 |
| **7** | **Forest drawn as instanced composite forms rather than loose leaf cubes** | Nothing visible. It is what makes a dense forest affordable at all | **+0.35** *(replaces ~1.3 ms that was never budgeted)* | **−420** | slack | P6 | item 6, per-chunk rung bias | No | **TAKE.** Closes D-DEC-12 |
| **8** | **Tonemapper and manual exposure, chosen and pinned** | The difference between a render and a photograph, on every pixel of a world made of large flat colour areas | **0.10** | 0.44 | post | P4 | — | **YES in practice** | **DECIDE with tile period** |
| **9** | **Cloud-shadow mask** | Large-scale brightness variation crawling across a hillside — the cure for "everything the same brightness" | **0.05** | 0.26 | in-line | P4 | one wind function | No | **TAKE** |
| **10** | **Albedo band + per-biome chroma floor, enforced at bake** | The difference between "believable" and "saturated toy", made structural instead of a taste review | 0.00 | 0.002 | — | P4 | — | **YES — authoring** | **TAKE with item 8** |
| **11** | **Anti-aliasing family: temporal resolve plus adaptive sharpening** | Clean resolve of the tier dither, the impostor dither, metre silhouettes against sky, specular sparkle | **0.45** | *saves* | post | P4 | — | **YES — excludes multisampling permanently** | **TAKE.** Already foreclosed in engine code |
| **12** | **One wind function, four consumers, exhaustive list** | Grass, rain, clouds and waves all move the same way | 0.00 | 0 | — | P4 | — | No | **TAKE.** Cheapest "alive" available |
| **13** | **Rock rim-warp amplitude per material, hashed direction** | Dirt and rock stop having machined edges; a rock face reads as an outcrop | 0.00 | 0 | — | P4 | existing warp | No | **TAKE.** The cheap 80% of "not blocky" |
| **14** | **Water: shoreline foam, depth absorption, sky reflection** | The metre staircase where a round ocean meets cubes reads as surf; shallows turquoise, deeps blue-black | **0.12** | 0 | slack | P4 | atmosphere | No | **TAKE** |
| **15** | **Emissive blocks as real coloured lights, plus bloom** | Ruling R3's payoff made visible: a torch, a forge, a thruster actually light the world | **0.25** | 0 | post | P4 | item 1 (to have dark to be bright against) | No | **TAKE** |
| **16** | **Crevice grime, corner wear, orientation moss and bleach** | The world reads as OLD. Nothing is factory-new except what you placed this minute | **0.02** | 0.002 | in-line | P6 | ambient occlusion, rim bit | No | **TAKE.** Zero extra texture taps |
| **17** | **Water: wave displacement at the near rung only** | The sea moves. Ponds stay flat, oceans roll, with no data telling them apart | **0.11** | 30 | slack | P4 (reserve in the mesher slice) | merge-cap field | **Yes-ish — merge rule** | **TAKE, at a 4-cell cap** |
| **18** | **Wet-surface response from the material's own porosity** | Rain visibly changes the world; sand and soil darken hard, steel barely reacts | 0.00 | 0 | in-line | P6 | weather | No | **TAKE** |
| **19** | **Snow: derived cover everywhere, real depth where you stand** | Snow-capped mountains from 10 km AND a path you can shovel | **0.01** | 0 | in-line | P6 | climate field | **YES — the boundary** | **TAKE the split** |
| **20** | **Puddles from the occupancy slab** | Water gathers in exactly the depressions you dug | **0.03** | 0 | in-line | P6 | occupancy slab | No | **TAKE** |
| **21** | **Precipitation on the micro-decoration machinery** | Rain and snow fall, slant with the wind, splash, and do not fall through your roof | **0.05** | 0 | decoration reserve | P6 | items 12, 1 | No | **TAKE.** No particle system needed |
| **22** | **Persisted age bits per block** | A freshly mined face is visibly fresher than thousand-year-old rock | 0.00 | 0 | — | P6 | reserved state bits | **YES — state bits** | **TAKE** |
| **23** | **Parallax self-shadowing, near field only** | The parallax stops wobbling and becomes readable centimetre depth on gravel, brick and rock | **0.15** | 0 | in-line | P6 | explicit mip derivation | No | **TAKE** |
| **24** | **Move hash shading variants off the merge key** | Per-block variation with no repetition — and it *recovers* merge efficiency the current scheme spends | **negative** | 0 | — | P4 slice 2 | — | **YES — vertex layout** | **TAKE.** One line, before the layout freezes |
| **25** | **Cloud deck as an analytic shell in the sky pass** | Clouds from the ground and a cloud-covered marble from orbit — the only form correct at both ends of a fly-by | **0.24** | 0 | sky | P7 | item 5 | No | **TAKE at P7** |
| **26** | **Ambient-occlusion stopgap until the real one lands** | Contact darkening in every corner from day one | **0.45** | 25 | slack | **P4 only**, retires P6 | — | No | **TAKE, and retire it** |
| **27** | **Water refraction through the engine's transmission path** | The seabed bends. The last big "this is real water" cue | **0.16** | 29.5 | slack | P8, on a trigger | — | No | **DEFER** |
| **28** | **Bounce light from an occupancy-traced probe clipmap** | A red canyon wall throws warm light on the opposite face; an overhang is dark but not black | **0.60–1.00** | 9 | measured slack | **P8 decision** | measured terrain line | No | **DEFER behind measurement** |
| **29** | **Local volumetric fog** | God rays through a cave mouth, mist in a dug valley, smoke with volume | **0.18** | small | slack | P8 | half-res custom pass | No | **DEFER** |

**Sum of items 1–26 (everything through P7): 3.61 ms**, of which 0.45 ms is the ambient-occlusion
stopgap that retires at P6, giving **3.16 ms steady state** — against 1.20 ms
of unspecified sky, 1.50 ms of unspecified post, and 1.75 ms of corrected average slack. §6 does the
full accounting; it closes with **1.32 ms in reserve**.

**The three-item critical path, if only three things happen next:** take decision **SL-2** (the render
origin becomes a pose — the only wire-format item, and the only one that gets more expensive every
week); migrate the **starfield**, which is free and which the file's own comment already asks for; and
land **item 1**, the sky-visibility ambient, which changes more pixels per millisecond than anything
else in the document and which is a prerequisite for a torch ever mattering.

---

## §3. The composite subsystem — trees, rocks, and everything made of several blocks

This discharges ruling **R4**. The composite study's model is **overturned**; what follows keeps its
pattern grammar, its recognition trigger discipline, its licence position and its slice plan, and
replaces its rendering and occupancy model with one that (a) actually makes a tree stop looking like a
column of cubes, (b) needs no second occupancy authority at all, and (c) pays for itself in memory.

### 3.1 R4 splits into two asks, and only one is a new subsystem

The owner's sentence contains two different requests.

**The rock half is not a composite.** *"I put a rock block and the decorational not-flat rock appears on
top of it"* is a single-cell statement. The rendering section already ships a rim-only inward
displacement of exposed convex corners — baked in the mesher, deterministic, provably contained,
costing time proportional to a quad's perimeter and collapsing to nothing at coarse rungs — authored
precisely so that *"dirt does not have razor edges"*. A rock block that reads as a rounded outcrop is
**that machinery with a larger per-material amplitude and a per-corner hash direction**. Amplitude is
already a per-material registry field (it is already zero on constructed materials). The change is a
number and a hash, and it inherits the existing containment proof, the existing merge-predicate
handling and the existing fade across the ladder.

The composite study proposed instead a per-cell rounded render template selected by the neighbour mask.
**Rejected**, and the adversary's arithmetic is why: a tier-0 chunk's exposed rock surface is about
3,844 cells; at even 24 triangles per template that is 92,000 triangles per chunk against a whole-chunk
measured figure of 4,400 — 21×, and 44–131× at believable template sizes. It also lands every exposed
rock cell in the per-cell instanced meshing lane, which the shape design bounds to *player builds* on
purpose.

**The tree half is genuinely new**, and it is the rest of this section.

### 3.2 The model: composites are *recognised*, never *overlaid*

> **The single rule. Every cell a composite occupies is a real cell in the block field. Recognition
> selects a RENDERING, and nothing else. There is no second occupancy authority, no overlay, no
> derived collider, and no cell that exists for physics but not for storage or vice versa.**

Three consequences, and each of them deletes a whole class of problem the composite study had to
design around:

1. **The collider question disappears.** The study spent four options and a full subsection on where a
   composite's collider comes from, given that the server has no meshes. The answer is: from the block
   field, because the tree *is* blocks. The block design's rule that a collider, a mass property, a
   raycast target, a structural-support query, a pressurisation query, a mount test, a block edit and a
   write-ahead-log record are all defined over tier-0 cells and over nothing else — eight readers — is
   untouched, with no narrowing type, no resolved-chunk accessor, and no risk of one reader forgetting
   the overlay.
2. **Generated composites are planted by the generator, as cells.** A forest costs zero stored bytes
   because *generated terrain* costs zero stored bytes, not because of any composite machinery. The
   generator writes the trunk and canopy cells while it writes the dirt. Cost: about 0.06 ms on top of
   the measured 0.885 ms per surface chunk in a dense forest — 7%. This is Minecraft's shipped model
   and it is why breaking one block out of a tree Just Works: the cells were already real, so the edit
   is an ordinary edit and there is nothing to "materialise".
3. **Degradation is free and instant.** Chop one block: recognition's predicate fails, the composite
   stops being drawn as a mesh, and the very next frame shows the cubes that were always there, minus
   one. No collapse, no flicker, no shrink-pop, no second fence.

**And the piece that makes the tree stop looking like a column of cubes:**

> **A cell owned by a recognised composite is EXCLUDED FROM THE CHUNK MESH'S EMIT STEP AT TIER 0 —
> but not from its OCCUPANCY step. The composite draws those cells itself, choosing one of five
> instanced forms by distance. At tier 1 and coarser the exclusion does not apply and the cells are
> drawn by the ordinary coarsened chunk mesh.**

The two halves of that rule matter separately. *Excluded from emit* means the tree's cubes are not in
the chunk's vertex buffer, so the drawn tree is the art mesh alone and its silhouette is the mesh's
silhouette — which is the entirety of what R4 asks for. *Not excluded from occupancy* means the mesher
still treats those cells as solid neighbours, so the grass face under the trunk stays culled and the
surrounding terrain's quad count is bit-for-bit what it was.

**Why this is legal where the decoration layer's version was not.** The decoration design considered
exactly this shape — a shell that suppresses the cube — and struck it, correctly, because a *cosmetic*
table must never reach back into the mesher. Here the suppression key is the recognition predicate,
which is an authoritative, integer-only, hash-gated pure function of the block field, in the same
category as the mesher itself. The chunk mesh remains a pure function of `(block field, block
definitions, composite recipe table, tier)`, with no camera in it — which is what the skirt ruling
requires, and it is why the adversary's own proposed fix (suppress faces *inside the mesh range*) is
rejected here: that makes the chunk mesh a function of camera distance, forbids server-side meshing,
and re-meshes about eighty chunk columns per second of pure waste every time the range boundary sweeps
across a chunk during flight.

### 3.3 The pattern grammar

Unchanged from the composite study, which got this right. Three masks and one derived integer; no
solver, no arbitration pass, no iteration to a fixed point.

```rust
pub struct CompositeRecipeId(u16);          // dense, append-only, never reused, retired rows kept

pub struct CompositeRecipe {
    pub id:          CompositeRecipeId,
    pub core:        CoreMask,              // <= 8^3 cells the player must have placed, plus their types
    pub core_types:  &'static [BlockTypeId],
    pub support:     SupportMask,           // <= 4 cells that must be solid under it
    pub soft_overwrite: SubstanceMask,      // substances the footprint may occupy; anything else BLOCKS
    pub rank:        u8,                    // = core.count_ones(). ASSERTED at table build, never authored
    pub feature_h_q8: u16,
    pub maturity_ticks: u32,                // the threshold; maturity itself is INSTANCE state (§3.4)
    pub yaw_mode:    YawMode,               // reuses the block record's 5-bit orientation
    pub variants:    &'static [CompositeVariant],
}

pub struct CompositeVariant {
    pub footprint: &'static [FootCell],     // the cells the generator plants / maturity writes
    pub mesh_c0a:  MeshId,                  // art mesh, near      <= composite_tris_near_max
    pub mesh_c0b:  MeshId,                  // art mesh, mid       <= composite_tris_far_max
    pub cubes_t0:  MeshId,                  // baked cube form, 1 m  <= composite_far_quads_max
    pub cubes_t1:  MeshId,                  // baked cube form, 2 m
    pub cubes_t2:  MeshId,                  // baked cube form, 4 m
}
```

**Supersession is arithmetic, not authoring.** Rank is the population count of the core mask, asserted
at table build. A 2×2×8 core always beats a 2×2×4 always beats a single block. *"Put more blocks, get a
bigger tree"* is therefore a property of the table rather than a number somebody can get wrong.

**Ordering is total, and it must be.** The ordering is `(rank descending, then anchor cell ascending,
then recipe identifier ascending)`, with totality asserted at table build. The adversary is right that
the first two do not order two rows with identical cores; the third clause closes it, and §3.4 removes
the case that produced two such rows in the first place.

**The table freeze rule is a build gate.** Rows are append-only and a shipped row's geometry is frozen.
Additionally, **a new row may not change the match set of any core pattern a shipped world can already
contain** — otherwise adding an "ancient oak" in a later patch silently converts every player's tree
into a different one. Enforced mechanically: a new row must either use a block type introduced in the
same patch, or sit behind a world-format epoch.

**The recipe table folds into the block-registry hash**, so a client with a different table is refused
at the door rather than silently seeing a different forest.

### 3.4 Recognition, maturity, and planting

**Recognition is trigger-driven, never swept.** It runs on the dirty set of an edit and, for edited
chunks only, on chunk load. Every shipped multiblock recogniser works this way; the only poller in
Minecraft is the beacon, and that is a gameplay effect where a late answer is acceptable. Geometry
cannot be late.

The predicate, first failure wins: every core cell holds the declared type; every support cell is
solid; every footprint cell outside the core holds a substance in the permitted set; no core cell is
owned by a strictly greater candidate.

**Cost, corrected.** The adversary is right that the third clause is proportional to the footprint on
every *successful* match and that a headless shard has no mesher to fold the work into. Both dissolve
here: **generated composites are planted by the generator, so an unedited chunk's composite list comes
from the generator's own placement pass at zero extra cost, and only edited chunks are scanned.** The
scan is bounded by the count of trigger-type cells in the chunk. Worst case — a fully forested,
heavily edited chunk — is ~0.24 ms once per chunk load on the task pool, which is stated rather than
hidden. Per *edit* the cost is unchanged from the study's measurement: 0.5 µs typical, 1.7 µs worst,
against a per-edit decoration budget of 0.02 ms typical.

**Maturity is instance state, not a recipe row.** The anchor block's growth byte — which already exists,
which ruling R8 already commits to, and which advances by a memoryless per-tick probability so that a
binomial draw over elapsed sleep gives the exact distribution with no planting timestamp — is the
maturity. A generator-planted composite is written with growth at maximum. A player-planted one starts
at zero. **One recipe row serves both**, which removes the ordering hazard entirely and removes the
adversary's two-identical-cores case at its source.

**Planting, end to end.** The player places the core blocks. Recognition fires immediately and the
composite is *drawn* at its full rank right away — the gesture is legible the instant the pattern
completes, which is the owner's literal ask. Its footprint cells do **not** exist yet: it is drawn as
its sapling variant, sized to the core, and it collides as the core. When the growth byte reaches the
recipe's threshold, the footprint is written **once**, as an ordinary edit batch under an ordinary
fence — 3,146 cells at 8 bytes is 25.2 KB, precisely one crater's worth, which the edit design already
treats as its worst single edit — and from that moment the planted tree is indistinguishable from a
generated one.

### 3.5 What the bake tool must guarantee

All offline, all producing checked-in, content-hashed data. The pipeline is necessarily source model →
Blender → glTF, because the engine ships no importer for the pack format.

| Gate | Rule | Why |
|---|---|---|
| **CONTAINMENT** | No solid part of the art mesh outside the footprint dilated by ≤ 1.00 m laterally and downward, and **nothing at all above the top plane of the topmost solid footprint cell** | A visual floating over nothing, and standing on air, are both worse than a blocky silhouette. The asymmetry is the design's own rule: a visual at or below the collision surface is the harmless side |
| **COVERAGE** | The art mesh's silhouette from eight directions must cover ≥ 92% of the footprint silhouette, and the canopy footprint must be **simply connected** | The adversary is right that 85% plus a 0.35 fill threshold permits ~13 m² of collider you cannot see. Raising both closes it |
| **FILL THRESHOLD** | A cell enters the footprint at occupancy ≥ **0.50** (was 0.35) | A solid canopy, not a ragged one. This is also what makes the next gate achievable |
| **FAR-FORM QUAD CEILING** | The baked cube form at 1 m must be ≤ **128 quads** after greedy merging; at 2 m ≤ 48; at 4 m ≤ 16 | This is the single number that decides whether a forest fits in the frame (§6.4). A ragged canopy cannot meet it; a solid one merges to a few dozen quads |
| **TRIANGLE CEILINGS** | Art mesh ≤ 1,200 triangles near, ≤ 400 mid | Budget-derived, asserted by the tool. Pack triangle counts are unverified by any primary source, so the budget names the ceiling |
| **MATERIAL** | Re-authored onto the one shared surface material; non-metallic; roughness in the natural band; texel density inside the art contract's band; **pipeline key OPAQUE, never alpha-mask, for every canopy form**; no opt-out of the depth prepass | §3.7 — the opaque ruling is what makes composites cheaper than the cubes they replace |

**The licence position is unchanged and is absolute.** Source files never leave the team, never enter a
public repository or a public build artifact, and **never go to any AI tool**; no bake stage may use a
learned upscaler. A baked asset inside the client is "incorporated into a Product" and ships normally.
One clause needs an email before a *player-plantable* pack-derived composite ships: the prohibition on
use for "Game Creation Software". Cost of asking: an email. Cost of being wrong: a shipped game with a
licence problem.

### 3.6 The anti-conjuration bound, restated so it actually binds

The exploit is real: place 32 wood blocks, get a 3,146-cell tree, harvest 3,146. The composite study
states it correctly, offers the right invariant, and then recommends a setting that reopens it
("minutes, not days"). The adversary is right.

**The invariant, which is not negotiable and lands in the first slice:**

> **An immature composite cannot be harvested for more than it cost. Breaking one before maturity
> removes it and drops only its core blocks. Materialising one resets its growth byte to zero.**

**And the bound that actually holds, which is not a rate limiter: area × time.** A composite's
footprint cells must be clear for it to exist at all (recognition clause three), so composites cannot
overlap while growing; and maturity is measured in **days of world time**, which is what ruling R8's
memoryless-growth-across-downtime framing already implies. Yield is therefore bounded by

```
blocks per second  <=  footprint_cells / (footprint_area_m2 * maturity_seconds)  *  cleared_area_m2
```

At a 3,146-cell tree over a 121 m² footprint with a seven-day maturity, that is **3.7 blocks per square
metre per day**. A hundred-metre-square plantation yields about 37,000 blocks a day and costs land,
time, 32 blocks and labour per tree. That is not an infinite printer; **it is forestry, which is
precisely the profession ruling R7 wants the player-driven economy to price.** The number belongs to
the economy design, not to this one; the *invariant* belongs here.

The rate-limit fields survive as belt-and-braces (materialisations per realm per second, fill writes per
player per second) and their job is the forest-fire case — a hundred trees materialising in one tick is
2.5 MB of log and delta traffic — not the economy case.

**One reserved bit is no longer needed.** The composite study reserved a placement-record bit to stop a
cleared-and-replanted stump re-materialising. With maturity as instance state and the growth byte reset
on materialisation, replanting simply takes another maturity period. That removes a persistence-format
one-way door at zero cost, which is the best kind of deletion.

### 3.7 Behaviour at every distance, and the number that inverts the adversary's verdict

| Rung | Extent | What is drawn | Where it comes from |
|---|---|---|---|
| **C0a** | 0 → 40 m | the art mesh, near chain, **alone** | the recipe's near mesh |
| **C0b** | 40 → 150 m | the art mesh, mid chain, **alone** | the recipe's mid mesh |
| **C1** | 150 m → the tier-0 range | the baked 1 m cube form, instanced | the recipe's cube form |
| **C2 / C3** | tier 1 / tier 2 chunks | the ordinary coarsened chunk mesh | the tier ladder + the edit pyramid |
| — | — | **no impostor rung, ever** | — |

**There is no impostor rung, and the composite study's arithmetic for that is correct.** Integrating the
fill-bound impostor model over a forest at one tree per 100 m², feature height 8 m, from 70 m to the
tier-0 range gives 4.27×10⁷ pixels — 7.97 ms at the impostor denominator, over half the owned frame,
at 11.5× overdraw. The rung underneath is real cubes, which is strictly better and already paid for.

**And here is where the adversary's third critical finding inverts.** He re-ran that same fill model on
the *mesh* rung and got 4.7–6.4 ms, concluding the composite lane cannot be scheduled. The model is
right; the denominator is wrong. That formula prices **alpha-tested** fill, where every fragment must
run the shader before it can be discarded, so occluded layers cost full shading. **A bought low-poly
canopy is solid geometry.** With opaque geometry and a depth prepass, occluded fragments are rejected
by the depth test at raster rate and never shaded at all.

Which reveals the real finding, and it is not about composites:

> **A dense forest of alpha-cutout leaf cubes has a depth complexity of about twelve. At one tree per
> 100 m² with a four-metre canopy radius the mean free path through the canopy is 12.5 m, so a
> horizontal look through 150 m of forest crosses about twelve canopies. Twelve layers × 3.686 Mpx ×
> alpha-tested shading is roughly 3 ms of pure fill that no line in the design's frame table
> accounts for.** That cost exists today, in the shipped decoration design, with or without R4.
>
> **Composites remove it**, by replacing those twelve alpha-tested layers with one opaque surface
> inside the mesh range and an opaque merged cube form beyond it. This is the strongest argument for
> the composite subsystem in the entire run, and neither the study nor the adversary found it.

**Composites cast shadows from their own mesh.** The study ruled them non-casting on the grounds that
the footprint's shadow already covers them; in the corrected model the footprint is not drawn, so that
argument evaporates and the adversary's box-shadow complaint is answered at the root. The 1.6× cascade
multiplier comes back onto the bill and §6.3 pays for it by shortening the authored shadow horizon.

**The handover to the far rungs is a fade to the cube form**, using the crossfade band field that
already exists, with an exhaustive consumer list, and never a second band field.

**One gap the detail-ladder addendum does not cover, and it must be named.** *"Tier L drops the L
highest-frequency octaves"* is the exact answer for a fractal height field and gives **nothing** for a
discrete object, because a tree is not an octave. The coarse generator must evaluate the *same integer
placement hash* at any tier and plant coarse composites, or every forest on the planet appears in a
ring at the tier-0 boundary — the one pop the seamless law forbids. Two lines and one extra hash
evaluation per coarse cell, at the measured 12.19 ns.

### 3.8 What this costs and what it saves

**Frame**, at the reference machine, one tree per 100 m²:

```
near mesh rungs:  pi * 0.01 * [40^2 * 1200 + (150^2 - 40^2) * 400]  =  322,800 triangles
                  322,800 * 2 passes / 1.5e9                        =  0.43 ms
```

with the shadow pass folded into the terrain shadow line (§6.3). Draw calls ride the existing
instanced indirect loop as extra archetype rows: about six extra indirect calls where multi-draw is
available, about forty loop draws on the fallback path.

**Memory — this is the headline.** A dense forest inside the tier-0 disc is about 19,400 trees. Drawn
as loose leaf and wood cubes in the chunk mesh, at roughly 250 exposed quads per tree, that is 4.85 M
quads of unique vertex data:

| | Loose cubes in the chunk mesh | Instanced composite forms | Saving |
|---|---|---|---|
| At the current 88-byte quad | **427 MiB** | 7.7 MiB catalogue + 0.6 MiB instances | **−419 MiB** |
| At the packed 8-byte quad | 39 MiB | 7.7 + 0.6 MiB | −31 MiB |

**419 MiB is more than twice the entire remaining video-memory margin at the current vertex format
(§6.5).** The feature that answers R4 is also the feature that makes a forest fit.

**Catalogue:** 32 archetypes × 4 variants × 5 rungs ≈ 7.7 MiB. **Colliders:** zero — the collider is the
block field. **Wire bytes:** zero. **New persisted record types:** zero. **New inter-shard message
arms:** zero. **New entity kinds:** zero.

### 3.9 Slices, gates, and the one-way doors

| Slice | Phase | Contents |
|---|---|---|
| **A1** | P4 | The recipe table, the recognition predicate, the total order, golden vectors. **Plus the coarse generator's placement arm**, or forests pop at the tier-0 boundary. No rendering |
| **A2** | P4 | The rock half: per-material rim-warp amplitude and hashed corner direction. Ships first and independently |
| **A3** | P4 | Generator planting of composites as cells; the tier-0 emit-exclusion rule in the mesher |
| **A4** | P6 | Maturity, planting, the one-shot footprint write, the yield invariant, the rate buckets |
| **A5** | P6/P7 | The five instanced rungs, the crossfade, shadow casting. Depends on the art source decision |

| Gate | The number it must hold |
|---|---|
| `composite_recognition_golden` | Byte-identical over a pinned corpus, on two target-cpu builds |
| `composite_bake_gates` | Containment, coverage, the quad ceiling and the triangle ceilings **fail the build** |
| `composite_anywhere` | The identical fixture — place core, recognise, collide, break — on a spherical planet realm **and** a Cartesian ship realm, with a forced re-anchor |
| `composite_no_dupe` | Break an immature composite: only the core drops. Materialise, clear, replant: a full maturity period is required |
| `composite_no_pop` | No discontinuity at any of the five rung boundaries, at walking and at 100 m/s |
| `composite_occupancy_identity` | The chunk mesh's **occupancy** result is bit-identical with and without the emit exclusion — the rule may change what is drawn, never what is culled |
| `composite_forest_continuity` | Generated forest coverage is continuous across every tier boundary on a 5 km traverse |
| `composite_forest_frame` | A dense-forest fixture holds the §6.4 line, at the reference machine, or the build fails |
| `composite_registry_hash` | A client with a modified recipe table is **refused**, not tolerated |

**One-way doors, and there are now three rather than five:** the recipe identifier numbering and the
table-freeze rule; the footprint record's layout (it must carry a shape, not a boolean, so a later
precision upgrade is a re-bake and not a re-author); and the overhang convention, which is an *art*
door — the tables are re-bakeable at any time, the convention every archetype is authored to is not,
and it must be fixed before the first archetype is authored. **Explicitly not one-way:** every range,
every triangle and quad ceiling, the fill threshold, every rate bucket, the maturity threshold, and
recipe-table additions that obey the freeze rule. The placement-record bit the study reserved is **no
longer needed** (§3.6).

---

## §4. Lighting, atmosphere, water and weather

### 4.1 Why the whole prize is here

Six shipped titles agree and the pattern is not subtle. Valheim ships deliberately low-resolution
textures and buys everything back with fog, weather and time of day. Tiny Glade's untextured
flat-shaded geometry reads as physically believable because of real-time bounce light plus a serious
display transform, at 60 FPS on a GTX 1060. Vintage Story kept Minecraft-identical cube geometry and
added god rays, contact darkening, sky scattering and cloud shadows. Teardown has no textures at all
and its entire look is ray-marched occlusion and volumetric fog.

This design has already spent very well on geometry: the shape catalogue, the rim warp, greedy quads,
the detail ladder. **It has spent nothing on light and air**, and it has 2.70 ms allocated to a sky
line and a post line that nothing in seventeen thousand lines ever chooses the contents of. That is
the size of the prize and it is not a request for more budget.

### 4.2 Delete the flat ambient — item 1, and the first thing to build

The engine's default ambient light is white at 80 units, documented in its own source as *"lights the
entire scene equally"*. **In a world made of metre cubes that is the most damaging default in the
engine**, because an unshadowed face at the bottom of a pit receives exactly the ambient of one on a
ridge, and the world reads as a stack of uniformly lit cardboard boxes with a directional term painted
on top.

Two forms, and the first is nearly free today:

- **Near-term form (0.02 ms):** replace the constant with a hemispherical term — sky radiance on
  upward-facing normals, ground-albedo-tinted radiance on downward-facing ones, sampled from the
  atmosphere's own environment map. No new resource, no new pass, about six arithmetic operations.
- **Full form (0.10 ms):** make the existing per-fragment occlusion evaluator return a **sky-visibility
  scalar and a bent normal** alongside the occlusion scalar, and shade with sky radiance along the bent
  normal scaled by sky visibility. This is what makes a cave genuinely dark, an overhang genuinely
  shaded and a hilltop genuinely bright, and **it re-reads a slab that is already built, already
  uploaded and already bound.** No new bound resource, no new memory, no new pass.

Two inherited constraints: the sky term must share the occlusion slab's cell-size cutoff (near rungs
only) or slab residency silently triples; and it must ramp to zero across the *same* crossfade band as
everything else, becoming the fifth entry on a consumer list that already exists rather than a second
field.

**The adversary is right that the gate for this cannot pass at P4 on the near-term mechanism** — a
hemispherical term with no occlusion gives a pit floor the same sky radiance as a ridge. The
pit-versus-ridge gate therefore lands at P6 with the full form; at P4 the assertion is the weaker and
honest one, that the ambient is sky-*coloured* and normal-dependent rather than a white constant.

**Why this matters beyond looks:** a genuinely black cave is the precondition for a torch being an
event. Deep Rock Galactic has no ambient sky light whatsoever, and that mechanic only reads as dramatic
because the base state is black. Ruling R3 already gives coloured emissive blocks as real lights; this
item is what gives them something to be bright against.

### 4.3 The atmosphere, and the three traps in the shipped one

The engine ships a physically-based sky in-tree — four lookup tables plus a fullscreen composite — and
the scattering medium is an asset, so a per-body atmosphere is **data**, derived from the body's radius,
scale height and composition. A carbon-dioxide or methane world gets a different sky from the
composition field the world design already returns. No art.

**Three traps, all verified in the vendored source, all data fixes, all invisible until someone looks
at a sunset and says "it's wrong" with no idea why:**

1. **The reference atmosphere hardcodes an Earth radius pair** (6,360 km / 6,460 km). Our reference body
   is 161,671 m. If the radii are not authored per body, the sky gradient and the horizon curvature
   visibly disagree with the planet drawn under them.
2. **The Mie terms are transposed.** Verified: `absorption: 3.996e-6, scattering: 0.444e-6`. The
   standard set is scattering 3.996e-6, extinction 4.44e-6, therefore absorption 0.444e-6. The Rayleigh
   and ozone triples in the same function match the reference exactly, which is what makes this a
   transposition rather than an artistic choice. As shipped, the aerosol single-scattering albedo is
   0.100 — that is soot, not haze: too clean a sky, no forward glow around the sun, weak low-altitude
   haze, distant terrain too contrasty. **The fix is to swap the two values and nothing else.** The look
   study proposed an absorption of 4.40e-6, which is the *extinction*, and would have frozen a 2× error
   into a pinned golden test.
3. **The aerial-perspective table's default range covers 32% of our view distance, and the clamp is
   silent.** Past 32 km the in-scattered radiance pins to the last slice while transmittance keeps
   attenuating correctly, so the top two rungs come out **correctly extinguished but under-hazed — too
   dark and too contrasty**, which is exactly the artefact this item exists to remove. The fix is
   derived rather than authored: set the range to the top rung's outer range and the slice count so
   that one slice equals the tier-0 range exactly. Both become render-tuning fields derived from the
   ladder, so they move correctly when the residency governor moves.

**Cost.** All four tables are recomputed every frame with no caching and are **resolution-independent**,
which is why this fits: about 5.2 M ray-steps per frame plus one fullscreen composite, **0.70 ms**
against the 1.20 ms sky line, 1.87 MiB of video memory, zero wire bytes, zero per-chunk cost. The
multi-scattering pass dispatches at one workgroup of 64 — two warps, poor occupancy — and is the line
most likely to exceed the estimate. **Measure it first**, because it is what decides which end of §6's
bracket we are in.

**Mandatory riders.** The atmosphere requires a high-dynamic-range camera and *silently warns and does
not load* without compute support and a floating-point storage format — so a startup assertion is
mandatory, or every sky gate screenshots a black sky and passes for the wrong reason.

**One thing the engine does not give us, stated before anyone builds on it.** The atmosphere is one
component per camera with one radius pair and one medium. **Two planets' atmospheres cannot be rendered
in one frame.** A warp fly-by past three worlds gets exactly one. The compliant answer has two halves:
hand off between bodies by ramping the medium's density multiplier to zero across a band, re-anchoring
the frame and radii while both ends render nothing, then ramping the destination in; and give *other*
bodies an analytic limb-glow shell on their realm proxy, at about 0.05 ms per body. **Register the
second half as owed before warp**, because the seamless law promises a physical fly-by and the shipped
subsystem does not deliver it.

### 4.4 The render origin must become a pose — the urgent reserve

The shipped sky shader computes the view's altitude by adding the planet radius along +Y and comments,
in the vendored source: *"We assume the `up` vector at the view position is the y axis, since the world
is locally flat/level. NOTE: this means that if your world is actually spherical, this will be wrong."*
The planet centre is hardwired to render-space (0, −R, 0) in four places.

**The fix costs nothing and requires no fork.** If the client's render frame is the surface-tangent
frame beneath the camera — origin at the sub-camera point on the reference sphere, +Y along local up —
then for a rigid transform taking planet frame to render space, every fragment's reconstructed radius
is its exact radius, in all three shader sites, and the CPU-side up vector is exactly right too. The
alternative is maintaining a fork of about 500 lines of Rust plus seven shader files, on a subsystem
that changed materially in each of the last three engine releases.

**Two corrections to the look study's numbers, both the adversary's and both accepted.** The anchor must
lie on the reference sphere, not on the terrain surface — under a 4 km peak the implied altitude would
be wrong by 4 km. And the single-precision step at 161,671 m is 0.0156 m, not 0.0125; relative
precision is 6e-8, not 8e-6. The conclusion is unchanged and is the best single recommendation in the
run: **reserve the rotation now.** Per the standing law that the server does all the maths, the
**server authors the orientation** — the client may not derive it. Two riders: re-anchor on a distance
threshold with hysteresis rather than every frame, and assert that the previous-frame transform updates
in lockstep, or the temporal resolve ghosts on every re-anchor.

### 4.5 Water

Water is undesigned in the sense that matters — the word appears thirty times in the block design and
every one is incidental. The material half is committed (a translucent material with a scrolling offset
and a depth-based absorption term), so three things are missing: **foam, waves, and a reflection.** In
a cube world water is the only large smooth surface, and it is where the eye goes.

**Foam is the highest-value pixel in the topic and is nearly free.** A shoreline where a spherical ocean
meets metre cubes is a staircase — the single ugliest artefact water can produce here, and the one most
likely to make a reviewer say *"it looks like Minecraft"*. Depth-based foam hides exactly that, and the
depth sample is already fetched for the absorption term, so the incremental cost is one scrolling-noise
tap and a clamp: **0.02 ms**. The rest of the item is blending the reflection to the atmosphere's own
sky table by Fresnel, so the reflected sky is by construction the sky that was just drawn and no colour
mismatch is possible: **0.10 ms**, no new binding.

**Waves need a merge cap, and the look study put it in the wrong place.** There is no tessellation stage
in the graphics API at all (verified: zero occurrences in the API's type crate), and a flat sea is
exactly what greedy merging is best at — which is why a 62 m merged water quad has four vertices and
cannot be wave-displaced. The study's answer was a per-material merge-extent cap plus a 256 m camera
radius. **The adversary is right that this fails twice**: the merge predicate lives in the mesher, which
has no camera, so *every* water quad inside the near rung becomes 2 m — 340,000 quads at a sea view
instead of the claimed 2,779, a 150× miss — and nothing ramps the amplitude to zero at 256 m, so there
is a visible ring where the sea goes flat, on the largest smooth surface in the game.

> **Both are fixed by keying to the chunk's TIER rather than to camera distance.** The merge cap applies
> at tier 0 and nowhere else; the wave amplitude ramps to zero across the **same crossfade band the tier
> ladder already uses**. The mesher stays camera-free, there is no ring because the tier boundary
> already dithers, and no new tuning field is declared.

Ship the cap at **4 cells, not 2**: 85,000 quads at a full sea view instead of 340,000, **0.11 ms**, and
a 4 m wavelength floor against a 0.25 m amplitude ceiling that is inside the rim-warp precedent and
keeps the surface inside its own cell. Amplitude derives from the quad's own extent, so a three-block
pond gets near-zero swell and an ocean gets full swell, from data the mesher already emits and with
zero new state.

**Why the closed-form wave and not a spectral one, and the determinism correction.** The spectral
approach only pays above about thirty waves, which a 4 m minimum quad at 0.25 m amplitude cannot show;
decisively, the mechanics catalogue already lists buoyancy, so the **server** must evaluate the water
height at a point, and a spectral model has no closed form at a point. **But the study then broke the
project's own determinism rule**, and the adversary is right: a wave built from sine and cosine,
phase-driven by a field that is explicitly exempt from byte-identity as client-local cosmetic, cannot be
a simulation input — two shards on different maths libraries float a boat at different heights, and the
exemption silently erodes. **The server-side surface-height function must be the integer-plus-polynomial
idiom the design already prescribes for weather, with the vertex shader evaluating the same
polynomial.** That is cheap, and it is the only version in which boats float where the water is.

### 4.6 Weather splits in two, and the split is the design

| Half | Contents | Format | Gate |
|---|---|---|---|
| **Gameplay** | precipitation kind and intensity, temperature, wind, per 64 m cell per 64-tick step | integer / fixed-point, **transcendental-free** | **inside** the byte-identity gate, pinned across two target-cpu builds |
| **Presentation** | droplet positions, gust phase, cloud shape, puddle ripple, lightning brightness | float | outside, on the design's own existing argument for wind |

Steady-state weather crosses the wire as **zero bytes**.

**Wetness costs zero new per-material data**, because porosity is already a material field and wetness
is already ruled a derived two-bit quantity with zero persisted bytes. The physically-based response is
two arithmetic operations, no new taps, no new texture, no new registry row: sand and soil darken hard,
structural steel does not, and the difference is physically motivated rather than a global multiply.
Drying must be a **linear ramp**, not an exponential, because the exponential is non-deterministic and
wetness is read by the fire-spread gate.

**Snow is the one place authority genuinely splits, and it is the highest-value weather effect a block
world has** — precisely because the grid quantises it. An eighth-metre layer on a metre cube reads
instantly, where the same twelve centimetres on a smooth mesh reads as noise. **Cover** is a rendering
term from a closed-form climate field that octave-drops exactly like the height field, available at
every rung, about six operations on up-facing quads, zero bytes stored and zero on the wire — that is
what paints a mountain cap from 10 km. **Depth** is the three persisted bits that already exist, placed
by the random-tick scheduler that is already priced, diggable and collidable on the ordinary edit path.
Shovelling clears both.

**Puddles are never water blocks.** A cell whose up-face is exposed and whose lateral neighbours are
solid is exactly an occupancy-slab neighbourhood query on a slab that is already bound; pool water in
the bottom eight centimetres, faded by wetness. **0.03 ms, zero memory.** One rainstorm over one
continent as real blocks would be millions of durable records for something nobody can dig.

**Precipitation needs no particle system.** Rain, snow, dust, embers and spray are all *stateless* —
position is a hash of the instance identifier plus a function of wind and tick — which is exactly the
shape of the micro-decoration tier: already built, already budgeted, already gated. Reuse it: one
instanced world-space pass and one archetype table. A 16,000-instance storm is 0.05 ms. **Roof
occlusion is free**: the client already holds the block field for resident near chunks because it meshes
them, so it derives the per-column sky height locally and discards instances below it with one fetch in
the vertex shader.

---

## §5. Surface craft and aging

### 5.1 Aging is a re-read, not a subsystem

The strongest "this world has a history" cues are **already computed by this design for other reasons**.
Every one of them costs **zero extra texture taps**.

| Signal | Already computed by | Aging term | Cost |
|---|---|---|---|
| Concavity | the per-fragment occlusion term | grime: dark, damp, mossy crevices | 0 taps, ~4 ops |
| Convex edge | the rim predicate's 256-entry table, already handed to the decoration layer | edge wear, lerped to the base material | 0 taps, ~3 ops |
| Orientation | the up-face, taken through the gravity direction | moss on shaded damp faces, bleach on sun-beaten ones | 0 taps, ~8 ops |
| Per-block identity | the per-block integer hash | value, hue and roughness jitter | 0 taps |
| Porosity | the material's fluid parameter group | the wet response | 0 taps, ~2 ops |

**The whole item is 0.02 ms and 2 KiB** — every age term is a colour-and-roughness *pair* on the
material registry row, about 16 bytes per row across the 128-material shipping set, **never a fourth
texture array**. A grime texture would be three more taps and would push the worst shippable material
past the per-pixel ceiling.

**Three traps.** If the up-face is taken as the raw grid face rather than through the gravity direction,
the run-anywhere fixture passes while meaning "+Y" on a planet and "up" on a ship. At coarse rungs the
per-block state bits do not exist, so the orientation term must fall back to the macro octave stack or
moss switches off at the tier-0 boundary. And a **global** grime strength reads as a filter over the
world rather than as history — it must be per-material, under the albedo gate.

**The adversary's per-block jitter finding is upheld**: an ±8% value perturbation on a metre grid is
two-pixel luminance noise at the tier-0 range and, because it is keyed to the cell rather than to an
authored feature size, it does not coarsen with distance. It must fade across the same crossfade band as
everything else — consumer number six on a list that already exists — or it is a shimmer source in
exactly the far field the atmosphere exists to calm.

### 5.2 Parallax self-shadowing beats silhouette parallax

Silhouette-refined parallax in the literature needs extra geometry or extra baked per-texel data, and
the design already concedes that parallax cannot alter a rasterised triangle's edge against the sky. In
a metre world **the silhouette IS the block edge**, and the design already buys a real geometric one:
baked, ≤0.20 m inward, provably contained, proportional to a quad's perimeter, and collapsing to nothing
at coarse rungs so most resident chunks pay zero.

Spend the taps on a **half-length second march along the light direction** inside the height field
instead: four more taps, near field only, **0.15 ms**. It delivers the contact darkening that turns an
eight-step march from a wobbling texture into readable depth.

**One correctness rider, and it is not optional.** The engine's own parallax shader samples at mip zero
always, with a comment explaining that one backend's compiler panics on gradient instructions inside a
loop. Ours must do the same, which means it **must derive one explicit mip level outside the loop** from
the pre-parallax derivatives, and the shadow march must reuse it. Otherwise parallax becomes a
per-pixel aliasing generator at every distance and the sparkle gate passes on baked content while the
shipped shader sparkles.

**A second, unclosed hole in the same family, stated so it does not ship.** The bake-time roughness
correction fixes sub-*texel* normal variance and the geometric anti-aliasing term fixes sub-*pixel*
geometric variance, but **neither covers normal variance added at runtime** by a detail normal or a
per-block wear term, because that variance is created after the fetch and the mip chain cannot know
about it. The rule to write down: any runtime normal perturbation contributes to roughness by the same
relation evaluated in the shader from its own known amplitude and screen footprint — about six
operations — and it may **not** be implemented by measuring the fetched normal's length, because the
locked two-channel normal format reconstructs the third component and every fetched normal is unit
length.

### 5.3 Move hash shading variants off the merge key — a negative-cost item

The vertex layout currently puts a three-bit cosmetic variant tag in the per-**vertex** packed word. For
cube faces that makes it a **merge key**, and the design already prices exactly this failure when
rejecting merge-key occlusion: the pattern changes cell to cell, so the merge predicate fragments quads
back toward the unculled count — made worse by the ladder, because the fragmentation then happens at
every rung where the saving lives.

> **Rule: a hash variant that changes SHADING is a fragment-shader read of the block hash (zero taps,
> already there). A hash variant that changes GEOMETRY must be a peel predicate on the quad perimeter
> like the rim warp (already there).**

Keep the three bits — the per-cell instanced lane genuinely needs them and its quads are already
unit-sized, so nothing is lost there. **This is a one-line change before the vertex layout freezes**;
cheap now, a format migration later.

### 5.4 The per-pixel tap budget does not include the lighting stack

The design states 24 soft / 32 hard texture taps per pixel and accounts only for **material** taps.
Counting from the vendored shaders, a lit fragment additionally pays one cube fetch for the filtered
diffuse environment, one cube plus one lookup fetch for the specular lobe, one atmosphere transmittance
fetch, one occlusion fetch, and at least one shadow comparison per cascade touched — **six to eight taps
before a single albedo sample.** The worst shippable material configuration is 26 taps. **26 + 7 = 33,
which exceeds the hard cap.** That is a real break, not a rounding error.

**Restate the budget as two independently asserted quantities** — material taps ≤ 24 soft / 26 hard, and
lighting taps ≤ 8 — with the shader-variant compile check asserting both. Do **not** simply raise the
single number to 40: the two halves have different owners and different failure modes. The material half
is a cache-locality and array-layer-thrash budget (a 26-tap pixel touches up to eighteen independent
array layers); the lighting half is a small fixed set of cache-resident tables.

---

## §6. The frame and memory budget

### 6.1 The denominator — the correction everything else rests on

The design states its frame budget twice: a 1080p / RTX 4060 sentence in the rendering section header,
and a fully derived 1440p / RTX 3060 reference tier with a 15.00 ms owned frame in the decoration
section. Those are 2.07 versus 3.686 megapixels — **1.78× apart** — and every fill-bound cost differs by
that factor depending on which line the reader believes.

Both budget studies ruled "the reference tier wins, restate the other header, it costs a paragraph."
**The adversary is right that this is not a paragraph.** The ladder's rung range is
`2^L / (px_per_cell × per_pixel_angle)`, and the per-pixel angle is the field of view divided by the
**render width**. Adopting the 1440p reference at the shipped two-pixels-per-cell target multiplies the
per-pixel angle by 0.75, which multiplies **residency** — not just fill — by **1.78×**:

| Quantity | As published (1080p ladder) | At 1440p, holding 2 px/cell | Factor |
|---|---|---|---|
| Tier-0 range | 786 m | 1,048 m | 1.333× |
| Resident chunks, 8 rungs | 6,821 | **12,113** | 1.776× |
| Terrain vertices + indices | 1,595 MiB | **2,835 MiB** | 1.776× |
| Committed geometry lines | 6.00 ms | **10.66 ms** | 1.776× |

Which puts total video memory at **4.07 GiB against a 3.0 GiB cap** and the committed frame at
**17.84 ms against 15.00 ms** — before a single beauty item is bought. His arithmetic is exact and both
budget papers quoted the 1080p residency against the 1440p frame.

**But his fix — set the pixel target to 2.67 and accept "a visible softening of the far field" — is
half wrong, and the right fix costs nothing.** Two pixels per cell at 1080p and 2.67 pixels per cell at
1440p are **the same angular detail**: 1.273 milliradians per cell, the same world, drawn with more
pixels. Nothing softens. What *would* be an upgrade is holding two pixels per cell at 1440p — and that
upgrade was never budgeted by anybody.

> **RULING: the ladder's detail knob is an ANGLE, not a pixel count.** One render-tuning field,
> `cell_angle_max = 1.273e-3 rad`, is the ladder's single control variable; the rung range becomes
> `2^L / cell_angle_max`; and pixels-per-cell becomes a **derived, reported** quantity —
> 2.00 at 1080p, 2.67 at 1440p, 4.00 at 2160p.
>
> Consequences, all of them good. **Every published residency, terrain-memory and geometry number in the
> block design is correct as published, and becomes resolution-invariant.** Only fill-bound lines move
> with resolution, and the reference table is already stated at 1440p. The residency governor's control
> variable becomes resolution-independent too, which is what it always should have been. And the whole
> class of bug the adversary found — a ladder number quoted at one resolution and spent at another —
> becomes unrepresentable.
>
> **The one thing it gives up is stated plainly:** a 4K screen gets a sharper picture of the *same*
> world rather than a more detailed one. Making detail scale with resolution is then a quality setting
> — lowering the angle — and it is exactly what the governor already does. That is decision **SL-1**.

**A second defect in the governor, found by the budget study and worth keeping.** Residency is
`C₀·(3 + 1.5(N−1))` where `C₀` falls as the square of the detail angle but `N`, the rung count, is a
**step function** of it. So every time the control variable crosses a rung boundary, residency jumps
*up* before the quadratic term wins it back — and at the shipped default the ladder sits *exactly on* a
boundary, so the first infinitesimal increase adds a ninth rung. **The governor must quantise its output
to the rung lattice and assert that resident bytes are monotone in the control variable**, or coarsening
to save memory can spend it instead. Cheap to fix now; nobody else caught it.

### 6.2 The four denominators, all of them named

The design publishes two throughput denominators and applies one of them outside the case it was derived
for.

| # | Denominator | Value at the reference machine | Status |
|---|---|---|---|
| D1 | Tiny alpha-masked triangle rate | 1.5 G tris/s | Published; correct **for decoration only** |
| D2 | Shaded fill, four-tap alpha-tested | 15 G px/s | Published |
| D3 | **Large opaque quad rate** | **3.0 G tris/s** attribute-fetched, **4.0** position-only | **NEW — unmeasured, and `bench-frame`'s first job** |
| D4 | Memory bandwidth | 360 GB/s (the AMD reference part is 256 GB/s — bandwidth-bound lines are 1.4× worse there) | NEW |
| D5 | Depth-only / raster-rejected fill | ~30 G px/s | **NEW — decides the forest line (§6.4)** |

**D3's absence is a live defect in the published budget.** Applying D1 to terrain predicts a 4.00 ms
depth prepass against a 1.10 ms budget — a 3.6× miss. Greedy terrain quads are large, opaque and
index-reused; they pay neither quad overshading nor alpha-test discard, which is exactly what derates
D1. Two denominators are required and the published one must be scoped to decoration in its own text.

**D5 is new here and it is what inverts the composite verdict** (§3.7): opaque geometry that fails the
depth test costs rasterisation and a depth compare, not shading. Every fill estimate in this document
states which of D2 and D5 it uses.

### 6.3 The 60 Hz frame, and it closes

Committed work is 13.18 ms with 1.82 ms of slack. **One correction to the published table, and it is
the one that makes the package affordable.** The decoration section returned 0.90 ms to the frame and
named the tier ladder — "skirts, the double residency of crossfade bands, and coarse chunk meshes" — as
the claimant. Priced: skirts are ≤1% of chunk quads and back-face culled from above, **+0.07 ms
average**; the crossfade's double residency is +7.8% on the geometry lines, **+0.28 ms and it is
TRANSIENT**, present only while rung boundaries are being crossed; coarse chunk meshes *are* the 6,821
and are already counted.

> **The ladder's average claim is 0.07 ms, not 0.90 ms. The other 0.28 ms is a 1%-low claim, not an
> average one, and belongs in the hitch budget. That frees 0.83 ms of average slack**, and it is the
> single change that makes the light-and-air package fit at the pessimistic end of the terrain line.

**The shipping frame, at the reference machine, steady state after P7:**

| Line | Published | After this plan | Δ |
|---|---|---|---|
| Depth + motion prepass, terrain | 1.10 | 1.10 | — |
| Shadow cascades, terrain | 1.30 | 1.30 | — (authored horizon drops 668 → 200 m; the terrain triangles that frees pay for composite and decoration shadow casting inside it) |
| Terrain opaque | 4.60 | **4.95** | +0.35 in-line: sky ambient 0.10, cloud shadow 0.05, wetness+puddles 0.03, aging 0.02, parallax shadow 0.15 |
| Decoration (its contract) | 1.90 | 1.90 | — (precipitation inside its own 0.15 reserve) |
| Entities | 1.60 | 1.60 | — |
| **Composites, near mesh rungs** | — | **0.43** | new |
| **Forest, far instanced cube form** | — | **0.35** | new — and it replaces ~1.3 ms that was never budgeted (§6.4) |
| **Water: surface geometry + material** | — | **0.23** | new |
| **Long-range shadow horizon** | — | **0.05** | new |
| **Environment map, one cube face per frame** | — | **0.05** | new |
| **Sky / atmosphere** | 1.20 *(unspecified)* | **0.85** | atmosphere 0.70, starfield 0.05, limb shells 0.10 — **0.35 returned** |
| **Post** | 1.50 *(unspecified)* | **0.80** | temporal resolve 0.40, tonemap 0.10, bloom 0.25, sharpening 0.05 — **0.70 returned** |
| Skirts (the ladder's real average claim) | *(inside slack)* | 0.07 | |
| **Slack** | 1.82 | **1.32** | |
| **Total** | 15.00 | **15.00** | |

> **VERDICT: the budget CLOSES, with 1.32 ms — 8.8% of the owned frame — in reserve.**
>
> It closes on two conditions, both stated rather than assumed. **First**, the detail-angle ruling of
> §6.1 must be taken, or the committed geometry lines are 10.66 ms and nothing fits. **Second**, the
> 0.35 ms of in-line shading is spent inside the terrain-opaque *allowance*, which carries about 2.09 ms
> of unmeasured conservatism (the bottom-up estimate is 2.51 ms against a 4.60 ms allowance). If
> `bench-frame` reports the line already needs its full allowance, the 0.35 ms moves onto the slack and
> the reserve becomes **0.97 ms**. It still closes. **This is the largest unmeasured number in the
> design and nothing in this package may be spent before it is pinned.**

**At P4** the composite lane, the parallax shadow, aging and puddles are absent and the 0.45 ms
occlusion stopgap is present: the frame lands at 13.15 ms with 1.85 ms of reserve. The stopgap **retires
at P6**, in the same phase composites arrive. That is a schedule, not a conflict.

**On the minimum machine** the reference detail angle would cost 2.4× the geometry time. The governor's
answer is arithmetic: raise the angle by the square root of 2.4 to 1.97 mrad — 3.1 pixels per cell at
1080p — and residency falls 2.4× to 2,840 chunks with terrain memory at 665 MiB. That is what the
governor is for, and it is the first time its authority has been written down.

### 6.4 The forest line — deferred entry D-DEC-12, quantified and closed

The decoration design hands one number to the rendering section and no owner: *"a dense forest at one
tree per 100 m² over the tier-0 disc is 19,400 trees × 300 triangles = 5.8 M triangles of leaf cube
inside tier 0."* It is registered as D-DEC-12 and it is the largest unowned number in the design. This
document owns it.

**What it actually costs, before any fix.** 19,400 trees at a conservative 256 triangles of far form is
5.0 M resident triangles; at a 20% frustum fraction that is 1.0 M submitted across prepass and main —
**1.33 ms**, against a whole-terrain submitted cap of 6 M triangles. On top of that sits the fill term
of §3.7: twelve overlapping layers of alpha-cutout leaf cube is roughly 44 megapixels of shaded fill,
about **3 ms**. Neither is in any published line. **A forest planet, as currently designed, does not
fit in the frame.**

**Three fixes, all of them already in the design, none of them new spend:**

1. **Composites make the forest opaque and instanced** (§3). The alpha-tested fill term collapses to a
   depth-tested one at D5, and the per-tree vertex data collapses to a shared baked form.
2. **The far form has a quad ceiling as a build gate** (§3.5): ≤128 quads at 1 m. A ragged canopy
   cannot meet it; a solid one merges to a few dozen quads. This is what keeps the triangle line
   bounded by a number somebody asserted rather than by whatever the mesher happened to emit.
3. **Per-chunk rung bias from measured quad density** — which is D-DEC-12's *own* proposed answer,
   never owned: a chunk producing many times the median quad count coarsens **one rung earlier**,
   bounded to ±1 rung. It is still one rule (screen-space error against a budget), it is per-chunk data
   the mesher already produces, and it is the same shape as the residency governor. For forest it halves
   the tier-0 range and therefore quarters the tree count inside it.

**Result: 4,850 trees at 256 triangles = 1.24 M resident, 0.25 M submitted, 0.35 ms** — the line booked
in §6.3. Beyond that, forest is drawn by the coarsened chunk mesh at 2 m cells, where a canopy at 500 m
subtends about eight pixels per cell and is indistinguishable from the fine version.

**Two things must be gates, not hopes:** the far-form quad ceiling (build gate, fails the build), and
**forest density as an authored per-biome field with a stated ceiling** — because the whole line is
linear in it, and one over-enthusiastic biome row can spend the reserve.

### 6.5 Video memory, and the ruling that makes it comfortable

At the reference resolution, with residency held by §6.1:

| Item | V1 (88-byte quad) | V2 (8-byte quad) |
|---|---|---|
| Terrain vertices + indices, full ladder, measured 2,200 quads/chunk | **1,595** | 147 |
| Material arrays, 128 × 4.00 MiB — tier-invariant | 512 | 512 |
| Macro-variation layers | 1.4 | 1.4 |
| Occupancy slabs, near rungs, sparse | 71 | 71 |
| Decoration base | 7.9 | 7.9 |
| Decoration impostor array *(optional — see the main register)* | 151.5 | 151.5 |
| **Render targets at 1440p** (corrected from ~200 at 1080p; ten targets itemised) | **320** | 320 |
| Engine residency, entity meshes, glyph atlas, misc | 200 | 200 |
| **Lighting package (new)** | **13.4** | 13.4 |
| **Composite catalogue (new)** | **7.7** | 7.7 |
| **Total** | **2,880 MiB = 2.81 GiB** | **1,432 MiB = 1.40 GiB** |
| **Spare against the 3.0 GiB cap** | **0.19 GiB** | **1.60 GiB** |

Two published numbers are corrected here and both are the budget study's finds: **render targets are
320 MiB at 1440p, not ~200 at 1080p**, and the **decoration row is 7.9 MiB base plus 151.5 optional, not
39** — the published row predates the decoration section's own re-sum, in both directions.

> **Finding: at the reference resolution, with the impostor array and the lighting package, the current
> vertex format has 0.19 GiB of spare against a 3.0 GiB cap — not the published 0.44 GiB.** That is
> inside any plausible measurement error and is a breach in practice. The published conclusion is
> correct **only** at 1080p, without the impostor array, and without lighting.

**And a saving nobody has counted: composites remove ~419 MiB of forest vertex data at the current
format** (§3.8), which is more than twice the remaining margin. So a *forested* planet at V1 is
comfortable and a *bare* one is tight — which is the wrong way round for a budget to behave.

> **RULING (decision SL-11): pull the packed vertex format from P8 to P6.** Three independent arguments
> now point the same way: video memory has no usable margin at the current format; the **upload** budget
> is already breached at the current format under motion (flight is upload-bound at about 240 m/s with
> the CPU at 21% of the task pool); and the composite instance path wants a packed instance record
> anyway. At the packed format the total is 1.40 GiB with 1.60 GiB spare, and every option in this
> document — including bounce light, the hero material array and 4K — becomes discussable rather than
> foreclosed.
>
> Until it lands, the residency governor holds the cap, and its enforced byte cap **must be derived from
> the corrected total above** rather than the published one, or it enforces a limit that is 120 MiB too
> generous and permits the breach it exists to prevent.

### 6.6 The 1% low — the hard rule that has no test

"No hitching" is a standing rule with **no gate behind it anywhere in the design.** An average-only
budget cannot detect the failure the seamless law actually cares about, because every hitch source is
invisible in a mean. And at 60 Hz with vertical sync, a frame that misses the deadline does not cost
17 ms — it costs 33, because it waits for the next scan-out. So "no hitching" means **99% of frames hit
the deadline**, plus a bound on frame-to-frame pacing, because a 12/20/12/20 ms sawtooth reads as judder
even though its mean is fine.

| Hitch source | Magnitude | Mitigation | Residual |
|---|---|---|---|
| **Pipeline specialisation on first draw** | **1–20 ms, unbounded** | Warm every pipeline key at realm spin-up by drawing it once into a 1×1 target; enable the on-disk pipeline cache | ≈0 if warmed; **the dominant hitch source if not** |
| Mesh slab growth | 1.25 ms for a 300→450 MiB copy | Pre-size the allocator to the resident byte cap at startup so it never grows in play | ≈0 |
| **The per-frame upload limit is a SOFT limit** — verified verbatim: *"uploading stops after the first asset that exceeds the limit"* | peak = the limit **plus one asset** | **Split at the producer**: the chunk-geometry encoder emits sub-meshes of ≤1 MiB. It is already one function | ≈0.1 ms |
| Crossfade double residency | +0.28 ms | none needed — inside the percentile budget | 0.28 ms |
| Edit storm: a 10 m explosion is 27 dirty chunks | 2 frames of upload | the three-frame remesh floor; coalescing | 0.15 ms main thread |
| **Realm spin-up / warp arrival** | the seamless law's own worst case | pipeline warm + medium precompute + the coarse-first residency rule | **Must be measured — no number exists** |

Summing the residuals onto the 13.68 ms average frame gives a 99th percentile of about **14.2 ms**
against a 16.67 ms deadline. That passes **provided pipeline warm-up and the producer-side upload split
both ship.** Without them the percentile is unbounded and no amount of average-case budgeting recovers
it: a 20 ms shader compile in the middle of a warp arrival is the loading screen the seamless law
forbids, arriving one frame at a time.

**The gate, and it is the single best new test proposed in this run.** Five scripted traverses, GPU
timestamp plus CPU wall clock, at least 3,000 frames each: a forward walk across a meadow with props at
5 m/s; a 5 km flight at 100 m/s crossing every rung boundary; a walk-away-and-return over 300 m; an edit
storm; and **a warp arrival with a realm spin-up and a descent to the surface**, which no current gate
covers and which the seamless law most directly demands. Assert the median, the 99th and the 99.9th
percentiles, the maximum frame-to-frame delta, and that no single stall exceeds 8 ms. Report the
histogram and the ten worst frames with their dominant pass into the run manifest.

### 6.7 Gates this section owes

| Gate | What it pins | First runs |
|---|---|---|
| **`bench-frame`** | The whole-frame GPU budget with per-pass timestamps. **Pins the large-opaque-quad rate and the terrain-opaque line — the largest unmeasured number in the design** | **P4.4 — before a single millisecond of this package is spent** |
| **`bench-frame-pacing`** | The percentiles above, including the warp-arrival traverse | P4.10 |
| **`bench-forest-frame`** | The §6.4 line on a dense-forest fixture, with and without composites | P6 |
| **`assert_vram_reference`** | Total video memory ≤ 3.0 GiB asserted at startup **at the reference resolution**, itemised into the run manifest so a regression names the row that grew | P4.3 |
| **`assert_pipeline_warm`** | Every pipeline key drawn once before the first player-visible frame of a realm | P4.10 |
| **`bench-gen-reference`** | The terrain-noise benchmark re-run on the reference CPU. The published 12.19 ns and 0.885 ms were **measured on an Apple M4 Pro**, and the maximum flight speed is derived from them | P4.1 |

---

## §7. The art-direction contract

Machinery does not produce a look. A look is produced by a small number of decisions taken once,
written down, and then enforced by tools that refuse the non-conforming asset rather than by a person
remembering.

### 7.1 The look statement, and the ceiling on stylisation

> **A real planet, built out of metre cubes, by people who have been there a long time.**

Four clauses, each a rule with teeth. **A real planet:** light, air and distance are physical; nothing
is lit by a constant; a shadowed face is blue because the sky is blue, and black in a cave because
there is no sky. **Built out of metre cubes:** the grid is the identity and is never hidden, apologised
for or smoothed away; edges are *worn*, never chamfered into a bevelled product; an observer must be
able to count blocks at ten metres. **By people:** constructed reads as constructed at 200 m without a
UI cue — natural materials are stochastic, unoriented, worn and anti-tiled; constructed materials are
periodic, oriented, panel-lined and clean. **A long time:** nothing is factory-new except what a player
placed this minute.

**The negative statement, which is the half that gets used in review.** Not a toy (uniform
high-saturation palettes with no dark values); not a cartoon (no light painted into albedo, no outline
pass, no cel ramp); not a photograph (we are not hiding the grid, and sub-metre smooth extraction is the
wrong target); and not Minecraft-with-shaders (per-block textures and per-block silhouette repetition
are already excluded).

> **The one-line ceiling: you may stylise FORM and COLOUR as far as you like; you may not stylise
> LIGHT.**

| Axis | Budget | Enforced by |
|---|---|---|
| Silhouette and form | **Unbounded** — any shape, any proportion. Blocky is already the identity | nothing needed |
| Colour | **Bounded** — the band and the floors of §7.3, applied to bought and authored art identically | per-texel bake gate |
| Shading and light response | **Zero** — one material family, one BRDF, one normal-map requirement, one roughness band, one tonemapper, one exposure, no light in albedo | pipeline-key and material-row assertions |

Every failure mode the decoration section names for bought assets — texel density, missing normal maps,
proportion — is a symptom of stylising *light* rather than form, because each of them changes how the
surface responds to the sun rather than what shape it is. Tiny Glade is the proof in one direction:
flat-shaded untextured geometry reads as believable when the light is real. A realistic-intent texture
set under weak lighting is the proof in the other.

### 7.2 The enforced parameter envelope

| Rule | Value | Enforced by |
|---|---|---|
| Base material type | the one surface material; the stock physically-based material alone is banned | CI: every registered material's type is one of one |
| Pipeline keys in existence | exactly **two** — opaque and alpha-mask (translucent is reserved for water, glass and panels, never for a prop or a composite) | CI |
| Metallic | 0.0 for every natural material; non-zero only on an explicitly flagged constructed class | bake assertion |
| Roughness | natural 0.70–0.95, constructed 0.25–0.90, never below 0.20 without a named exemption | bake assertion |
| Normal map | **mandatory on every material and every bought asset**; the normal layer may not be constant | bake assertion |
| Ambient | the engine's global ambient brightness is **0.0**, asserted at startup | startup assertion |
| Prepass and shadow phase | no asset may opt out, except translucent from the shadow phase | CI |
| Emissive | player-programmable emission is clamped to a ceiling, **a new required field and free today** | material-row clamp at decode |
| Per-primitive tint | ignored on every voxel stream | existing assertion |

Without the emissive ceiling, one player's wall of maximum-white blocks sets the exposure for everyone
in the realm.

**Cost of this entire section: zero milliseconds, about 2 KiB of material-row fields.**

### 7.3 The palette, with the correction that keeps it from going grey

**Rule 1 — the physically plausible albedo band, per texel, at bake time.** The industry's own
validators use sRGB 50–240 for dielectric base colour, which is linear luminance 0.032–0.871, with
0.013 as a tolerant floor for a short, reviewed, named list. That band is not a style choice; it is
where measured materials live, and violating it is what produces the toy look. Enforcement is the
mechanism the roughness chain already uses — a metadata flag written by the bake tool and checked at
load, refusing the material **by name**.

**Rule 2 — variation lives in VALUE and HUE, never in SATURATION.** Value ±8%, hue ±3°, roughness ±0.05,
wear 0..1, saturation not at all. Saturation jitter across neighbouring metre blocks does not read as
material variation; it reads as a palette-index error, which is precisely the artefact that makes a
voxel world look like a bug.

**Rule 3 — cross-biome colour is per-pixel DATA; mood is per-camera GRADING; the two must never swap
jobs.** The engine's colour-grading component is per *camera*, so it cannot vary across a frame showing
two biomes. Per-pixel colour identity therefore lives in the macro-variation array — at most 32 biome
layers at 256² for 1.4 MiB total — and per-camera mood lives in the grade, lerped by time of day and
weather at zero passes and zero memory. **Macro layer 0 is the IDENTITY layer** — constant, no variation
— and every constructed material references it; without it a ship hull acquires a 64 m mottle that reads
as grime the player cannot clean.

> **Rule 4 — NEW, and the adversary is right that without it this contract produces a competent grey
> world.** Every instrument in the contract as drafted is a *reducer*: a saturation ceiling, an albedo
> ceiling, a tonemapper that desaturates brights by its own documentation, aerial perspective washing
> the far field, cloud shadows multiplying the sun term down, and four subtractive aging terms. Nothing
> anywhere *adds* chroma. Three cheap counter-levers, all free:
>
> - **A per-biome chroma FLOOR beside the ceiling**, in the same tuning struct. A conifer band of
>   0.05–0.12 luminance with no saturation floor is a black-green smear.
> - **The sky-visibility ambient is explicitly COLOURED** — blue sky fill against warm sun — so that
>   local contrast is a *hue* contrast and not only a value contrast. That is the mechanism that makes
>   a shadowed cube face read as beautiful rather than as dark, and the look statement gestures at it
>   without ever making it a rule with a number.
> - **The macro layer is promoted to a hue authority**, not just a value one, at both octaves.

**Eleven archetype layers ship of 32 slots** — one identity plus ten natural, each a 256² layer;
twenty-one slots stay free because unused ones cost nothing. Everything that makes a *planet* out of an archetype is derived — the atmosphere's
scattering triples and scale height from the body's composition, the ground albedo from the dominant
surface substance, the biome assignment from the temperature and humidity fields, the per-block hash
variation. **That is a thousand recognisably different worlds from ten authored layers**, with no art
per world.

### 7.4 Texel density, and what it does to bought assets

The locked pair is four cells at 1024², which is 256 texels per cell nominal — and 256–361 across a cube
face, because a cube-sphere cell is 1.000 m only at the face centres and 0.709 m at the corners. That
1.41× range is the tolerance the eye is already being asked to accept.

> **Every asset is authored at 256 texels per metre of its own surface, and no asset may sit outside
> [128, 512] — one stop either side.**

| Surface | Texels/m today | In band? |
|---|---|---|
| Terrain material at the locked pair | 256–361 | yes |
| A large rock face at an 8-cell tile period | 128 | yes, at the floor |
| **A bought pack prop as shipped** — flat palette atlas | **1–10** | **no, by 25–250×** |
| A photogrammetric rock at 4096² over a 3 m boulder | ~1,365 | no, high by 2.7× — re-bake at 1024² gives 341 |

The band is one stop rather than half a stop for one honest reason: a metre cube face is viewed from
0.5 m to 100 km and a tree trunk from 2 m to 300 m, so their useful densities genuinely differ. Two
stops is where the mismatch becomes visible as a material discontinuity at the contact point — the trunk
meeting the dirt.

### 7.5 Bought assets: the real cost, and the exemption that makes greybox legal

**The re-authoring checklist is ten steps and 2.5–5.2 hours per archetype**: convert; re-UV to 256
texels per metre; author a real normal map and an occlusion-roughness-metallic-height set with roughness
in band; re-target base colour into the albedo band and **strip any painted light**; import into the
shared array and assert one of two pipeline keys; fix the pivot convention; author the feature height;
map the pack's own detail chain onto our rungs; check proportion against a metre cube; and — new for
composites — author the blocky far form the asset becomes past its mesh range.

**At 64 archetypes that is 160–333 hours, or €8,000–16,600 at a competent contractor's rate** — against
€15–40 k for 60–80 *original* props. So the cheap option is not "€200 and a few days of pipeline work";
it is **roughly a third to a half of commissioning originals, and it buys someone else's silhouettes**
in a style used by thousands of games. That reframing is the most useful number in the art study and
the owner should see it before choosing.

**The adversary's finding here is upheld and it is a real blocker:** the contract's own gates require
128–512 texels per metre for *every* mesh, and a raw pack prop is at 1–10. So either the gates are
bypassed for the entire window in which the owner forms his judgement of the look, or greybox is not
available at all.

> **Resolution: a named, dated, expiring exemption file.** Each entry names one asset, one gate and one
> expiry date. Before the date the build **warns**; after it the build **fails**. It is the same shape
> as the coverage-exemption file the project already runs, and it makes "gates off for greybox" a
> visible, ageing debt instead of a silent bypass.

### 7.6 How the contract is enforced rather than hoped for

| Gate | Asserts | Runs at | Fails on |
|---|---|---|---|
| `art_albedo_band` | every texel of every non-emissive albedo layer inside the band, or the material is on the named tolerant list | bake → metadata flag → load | a toy-bright or crushed-black material, by name |
| `art_chroma_floor` | per-biome saturation **floor** as well as ceiling | bake | the grey world |
| `art_texel_density` | measured density inside the band for every mesh and material | bake | a trunk at 4 texels per metre |
| `art_normal_present` | the normal layer is non-constant for every material | bake | plastic props beside normal-mapped terrain |
| `art_pipeline_keys` | exactly two keys exist across every prop, composite and decoration material | CI | an asset that cannot batch |
| `art_ambient_zero` | the engine's global ambient is 0.0 | startup | the flat-cardboard default silently returning |
| `art_exposure_pinned` | under the harness, auto-exposure is absent and exposure is the scenario's authored value | scenario start | history-dependent screenshots |
| `art_contract_asset` | the six-image acceptance test — the candidate beside a reference dirt cube and a painted-hull cube, at 2 m, 20 m and 200 m, under clear noon, overcast and night-with-one-light | per archetype | an asset that reads as bought |
| `art_contract_grid` | the canonical look grid — four times of day × three weathers at one fixed spot | per build | any change to the look arriving in review as a diff |

**And the adversary's finding on image hashes is upheld.** The content being compared includes a
four-table atmosphere chain recomputed every frame in compute shaders, a temporal history buffer and a
tonemapping lookup — none of which is bit-reproducible across three GPU vendors, and the reference
matrix guarantees at least two. **Image gates use a tolerance metric** — per-channel mean and
95th-percentile deviation against a stored reference, plus a structural term — not a hash. A hash gate
would be red on day two and disabled on day three, which is worse than not having it. The *integer*
gates (weather, recognition, decoration rules) stay byte-identity, because they are CPU integer paths.

**Exposure is manual, derived from a server-authored per-realm illuminance, and never automatic.** Two
reasons, both hard-rule-shaped: an adapting exposure extinguishes the starfield on a warp arrival into
a sunlit planet, which is a direct violation of "stars always visible"; and it makes every pixel-readback
gate history-dependent, so the failure presents as a flaky test rather than as a nondeterminism bug. The
range of daylight in this game is **nine orders of magnitude of illuminance**, from a moonless night to
raw sunlight in vacuum, which is exactly why the single exposure has to be authored rather than guessed.

**The contract binds ENTITY art too — player models, ships, projectiles, in-world panels — with no
exemptions.** The entities line is 10.7% of the frame and is currently outside every rule in this
section. A player model authored at a different texel density with baked occlusion in its albedo is the
most visible possible violation, because the player looks at it constantly.

---

## §8. The decision register for this document

Ordered by **cost of lateness**, which is not the same as importance. Rows that duplicate or supersede a
question already in the main design's register point at it.

| # | Decision | Recommendation | One-way? | Cost of taking it late | Duplicates / relates to |
|---|---|---|---|---|---|
| **SL-1** | **The ladder's detail knob becomes an ANGLE; the reference tier is the one denominator; the rendering header's 1080p sentence is restated as a derived figure** | **Take it.** Every published residency, terrain-memory and geometry number then becomes correct and resolution-invariant | No, but it is a **measurement door** | Every gate re-baselined; every recorded number re-measured | main row 38 / 8-B (governor); supersedes study decisions B-1 and C-1 |
| **SL-2** | **The server-told render origin widens from a translation to a rigid POSE, server-authored** | **Take it now.** It is what makes the atmosphere correct on a sphere; the alternative is a permanent engine fork | **YES — wire ABI and protocol minor** | Re-authoring the origin contract and every client consumer after the absolute-position path is in production | new; the character controller wants it independently |
| **SL-3** | **The display transform and the exposure policy, as a pair** | A neutral, hue-preserving tonemapper plus **manual exposure from a server-authored per-realm illuminance**. Auto-exposure only as an accessibility option the harness forces off | **YES in practice** | It changes what every material's albedo should be; choosing after 128 materials exist is a re-paint | **lock in the same sitting as main [USER DECISION 5-C]** |
| **SL-4** | **The look statement itself** — physically-lit naturalism, "PHYSICAL, BLOCKY, OLD" | **Take it.** It is the root node; every other art decision is downstream | **YES in practice** | Same door as SL-3 | new |
| **SL-5** | **Texel-density band: ±1 stop, [128, 512] texels/m** | ±1 stop. ±½ stop forces a second 2048² array trio that does not fit | **YES** — same door as 5-C | Same door as SL-3 | main 5-C |
| **SL-6** | **Composites are RECOGNISED, not overlaid: every composite cell is a real cell, and a recognised composite's cells are excluded from the chunk mesh's EMIT step at tier 0 only** | **Take it.** It is the only model in which a tree stops looking like a column of cubes, and it deletes the overlay, the derived collider and the second occupancy authority | **Yes-ish** — a mesher rule plus a frozen recipe table | Retrofitting an emit rule into a 100%-covered mesher, and re-baking every recipe | discharges **R4**; supersedes study decisions A-1, A-2, A-4 |
| **SL-7** | **Per-material merge cap applied at tier 0 only; water ships 4 cells; wave amplitude ramps on the tier crossfade band** | **Adopt in the mesher slice**, or accept a permanently flat sea | **Yes-ish** — merge predicate | Re-covering a 100%-gated mesher a second time and re-pinning its merge-ratio regression | supersedes study decision B-4 |
| **SL-8** | **Pull the packed vertex format from P8 to P6** | **Take it.** Video memory has no usable margin without it, upload is already breached under motion, and the composite instance path wants a packed record anyway | No — the encoder is one function | The governor holds a cap that is 120 MiB too generous and permits a breach | main row 21; supersedes study decision C-2 |
| **SL-9** | **Anti-aliasing family: temporal resolve plus adaptive sharpening; multisampling excluded** | **Record it as settled with a startup assertion.** Multisampling is already foreclosed in engine code — the occlusion pass and the temporal resolve both require it off | **YES — excludes multisampling permanently** | Someone proposes multisampling for "crisper block edges" and silently deletes occlusion, the temporal resolve, the tier crossfade and the impostor band at once | supersedes study decision B-6 |
| **SL-10** | **Snow: derived cover everywhere plus authoritative depth at tier 0** | **The split.** All-derived means you cannot dig snow; all-authoritative means a continental snowfall is a millions-of-records edit storm | **YES** | A persisted-format decision | supersedes study decision B-5 |
| **SL-11** | **Per-chunk rung bias from measured quad density, bounded to ±1 rung** | **Take it.** It is D-DEC-12's own proposed answer and it is what makes a forest fit | No | The forest line stays 4× its budget | closes main **D-DEC-12** |
| **SL-12** | **Engine version: stay on the current release through P4; take the next as a dedicated slice BEFORE the first custom render pass is written** | As stated. The next release replaces the render-graph API every custom pass in this plan and the decoration plan would be written against | **Yes in cost** — a render-graph rewrite either way | Writing five custom passes against a deleted API | supersedes study decision B-8 |
| **SL-13** | **Composite maturity is measured in DAYS; the immature-yield invariant lands in the first composite slice** | Days, per ruling R8's own framing. The number itself belongs to the economy design | No — a tuning field | The economy ships with a wood printer in it | relates to **R7**, **R8**; corrects study decision A-3 |
| **SL-14** | **Composite art source, and written licence confirmation before any player-plantable pack-derived composite ships** | Trees: bought greybox with a dated re-authoring commitment. Rocks: permissively-licensed scan-based. Leaf clusters: procedural — a bought mesh structurally cannot serve a class whose shape depends on the neighbour mask | **One direction only** | Shipping a recognisable style establishes an identity that is expensive to change | main 6-2, 6-6; supersedes study decision A-5 |
| **SL-15** | **The art contract binds ENTITY art too — players, ships, projectiles, in-world panels** | **Yes, all of it, no exemptions** | **YES in practice** | The asset the player looks at constantly is the one outside the contract | new |
| **SL-16** | **Forest density is an authored per-biome field with a stated ceiling, and the far-form quad ceiling is a build gate** | Take both. The forest line is linear in density | No | One over-enthusiastic biome row spends the whole reserve | new; supports SL-11 |
| **SL-17** | **Greybox exemptions are a named, dated, expiring file that warns then fails** | Take it, or the gates are silently off for the whole judgement window | No | Silent bypass becomes permanent | new |
| **SL-18** | **Global illumination approach** | **None beyond sky-visibility ambient through P7.** Revisit the occupancy-traced probe clipmap at P8, **after** the terrain line is measured. Hardware ray tracing is rejected: the graphics API exposes **zero** acceleration structures on the development machine's backend | No — nothing in the recommendation forecloses any other option | none; it is deliberately deferred | supersedes study decision B-7 |
| **SL-19** | **Launch biome archetype count and the starter world's set** | 10 natural plus 1 identity layer, 21 slots left free | No, except the starter world's set | The first thing every player sees | supersedes study decision AD-5 |
| **SL-20** | **The tolerant-albedo exemption list** | Ship with six names; every addition reviewed | No | — | supersedes study decision AD-6 |
| **SL-21** | **Is 120 Hz a shipping target?** | **No at the reference tier; yes as a high-tier preset** that raises the detail angle and drops the composite range. Saying "60 Hz is the contract" stops the gate matrix doubling | No, but it is a **gate multiplier** | Every percentile asserted at two refresh rates | supersedes study decision C-4 |

**If only three decisions are taken this week: SL-1, SL-2, SL-6.** The first makes every other number in
this document mean something. The second is the only wire-format item and the only one that gets more
expensive every week. The third is the one that decides whether R4 delivers a tree or a column of cubes.

---

## §9. What we are deliberately not building

| Not this | Because |
|---|---|
| **Hardware-raytraced global illumination** | The graphics API reports **zero acceleration structures** on the development machine's backend — verified in the vendored hardware abstraction layer. It cannot be developed against. NVIDIA-only in practice; alpha masks unsupported, which deletes the foliage tier; 8–14 ms of whole frame on a card well above the reference. The only shipped metre-voxel game with path tracing defaults to a **128 m** render distance against our 100 km ladder |
| **Baked irradiance volumes through the engine's documented workflow** | The engine ships no baker and the path is an offline bake through a third-party bridge. Our world is player-mutable at metre granularity; a runtime re-bake per edit is exactly the per-edit cost the occlusion design already rejected an atlas for. **Reject the workflow, keep the container** |
| **Screen-space reflections** | Verified in source: *"currently only supported with deferred rendering."* Our terrain is a forward material carrying a material identifier, a chunk row, a detail rung and an array layer — none of which survives a fixed geometry buffer. Going deferred re-specialises every pipeline and re-covers every producer test. The atmosphere's environment map buys most of the outdoor payoff free |
| **Multisampling and alpha-to-coverage foliage** | Foreclosed in engine code (the occlusion pass and the temporal resolve both hard-require it off) *and* by the design's own dither crossfade, impostor band and temporally-resolved grass |
| **A raymarched sky as the shipping path** | Three to six milliseconds — a third of the owned frame — for a gain visible mainly very close to the ground on small bodies. And the two modes differ **visibly**, so a hard switch is a pop the seamless law forbids |
| **Cloud raymarching as the first cloud** | The reference implementation's own numbers: ~20 ms naive, under 2 ms on console **only** via a one-pixel-in-sixteen update plus reprojection. A half-resolution version here is plausibly 1.0–1.5 ms of a 1.32 ms reserve, needs a history buffer that collides with the temporal resolve, and the sparse-update scheme smears during the 300 m/s flyover gate we already run. Take the shadow mask now and the analytic shell at P7 |
| **Billboard or impostor clouds** | ~300 cards at 200² pixels is 0.77 ms at our own impostor denominator; wrong from inside and from orbit; and a *substitute-for* rather than a *rung-of*, which the ladder addendum forbids by name |
| **Impostors for composites** | 7.97 ms at the fill denominator over a forest, at 11.5× overdraw, and the coverage cap would clamp the range to nothing anyway. The rung underneath is real cubes, which is strictly better and already paid for |
| **Planar reflections** | A second full scene render, *and* geometrically wrong on a small planet: the sagitta on a 161,671 m body is 0.77 m at 500 m and 3.09 m at 1 km, against 0.08 m at 1 km on Earth. The mirror visibly breaks at exactly the distance that matters |
| **A spectral ocean** | Its advantage only pays above about thirty waves, which a 4 m minimum quad at 0.25 m amplitude cannot show; and decisively it has **no closed form at a point**, so the committed buoyancy mechanic would read back a GPU texture or run a second approximate model, and boats would float at a height the water is not at |
| **A particle library, now** | Every effect in scope is stateless and is exactly the shape of the micro-decoration tier — already built, already budgeted, already gated. Adding one stands a **second instancing system** beside it, writes no motion vectors so particles ghost under the temporal resolve, and the leading candidate self-describes as *"lacking both features and performance."* A costed decision at P8/P11 when genuinely stateful particles arrive — not before |
| **Clustered decals for grime, wear or edit stains** | Verified: *"currently disabled on macOS and iOS due to insufficient texture bindings"* — the development machine. Every aging term must be derived inside the one surface material from signals the mesher already produced |
| **Puddles and rain-laid snow as real blocks** | Millions of durable records, the fluid frontier dragged across a continent, cosmetic state in the permanent edit pyramid — for something no player can dig |
| **Automatic exposure as the shipped default** | Stars vanish on warp arrival (a hard-rule violation) and every pinned image gate becomes history-dependent |
| **Per-cell rounded rock templates** | 21–131× the triangle count of the whole-chunk measured figure, and it lands every exposed rock cell in the per-cell instanced lane the shape design bounds to player builds on purpose. The rim warp with a per-material amplitude delivers the same look for zero |
| **A second wind function, a second crossfade band, or a second scheduler for anything** | The crossfade field already publishes an exhaustive consumer list; a duplicate band field has already been deleted once; a second tier scheduler is already named as *the* defect. Two copies of a fade band is precisely how a crossfade and an occlusion ramp end up disagreeing inside the same band |
| **A third frame budget** | §6.1 |

---

## §10. The adversary's embarrassing screenshot, and what prevents it

The adversary's exercise was to describe the screenshot this plan would produce if it were wrong. His
was: *a forest at golden hour on the starter planet, standing on the ground.* Every tree a two-metre
square column of wood cubes topped by a ragged blob of cutout leaf cubes, with a low-poly canopy
ghosting through the gaps and z-fighting where the mesh is coincident with a cube face. The trunk mesh
100% invisible. Three canopy representations in one volume. A rectangular box shadow under every tree.
An empty far field. And a grey palette.

It is a fair description of what the studies as drafted would have produced. Here is what in this plan
prevents each part of it.

| The failure | What prevents it |
|---|---|
| **The trunk is a column of cubes with an invisible mesh inside it** | §3.2's ruling: a recognised composite's cells are **excluded from the chunk mesh's emit step** at the near rung. The trunk you see is the mesh, because it is the only thing drawn. The cubes remain fully real for collision, mass, raycast and persistence — they are simply not the thing on screen |
| **The canopy ghosts through the cube leaves and z-fights** | The same ruling. There is one canopy surface, ever. And §3.5's opaque-only pipeline rule means the canopy is solid geometry, so there is nothing to see through and nothing to blend |
| **Three canopy representations inside 59 m** | The decoration layer reads the same recognition result, so it emits no leaf-cluster cards on cells a composite owns. One canopy |
| **A rectangular box shadow under an organic canopy** | The cubes are not drawn, so they cannot be the caster. The composite casts from its own mesh, inside a shadow horizon authored at 200 m — and the terrain triangles that shortening frees are what pay for it (§6.3) |
| **The silhouette never stops being blocky** | The coverage gate rises to 92%, the fill threshold to 0.50, and the outermost drawn surface inside the mesh range is the mesh. Beyond it, the silhouette is the baked cube form — blocky on purpose, at a distance where a canopy subtends a few dozen pixels |
| **The far field is empty** | Half true, and it is a fact about a 161,671 m body rather than a defect: the ground horizon is 741 m. But "six of eight rungs draw nothing" is wrong — a 100 m ridge is visible at 6.3 km and a 4 km peak at 36.7 km. From the ground the far field is **sparse and dramatic**: mountain tops over a close horizon. That is a look worth having, and aerial perspective is what makes it read as distance rather than as a decal. The re-rank is applied anyway: sky-visibility ambient and the shadow horizon are ranked **above** aerial perspective precisely because they win the ground frame |
| **The palette is grey** | §7.3 rule 4, which is new: a per-biome chroma **floor** beside every ceiling, a **coloured** sky ambient so shadow contrast is a hue contrast, and the macro layer promoted to a hue authority. Plus the pinned look grid, so a drift toward grey arrives in review as a diff rather than as a shipped mood |
| **The frame does not fit** | §6.1's angle ruling restores the committed geometry lines to 6.00 ms; §6.3's corrected ladder earmark frees 0.83 ms; §6.4 closes the forest line from ~1.3 ms to 0.35 ms; the budget lands at 13.68 ms with 1.32 ms of reserve |
| **The memory does not fit** | §6.5: it does not, at the current vertex format, with 0.19 GiB of margin — so SL-8 pulls the packed format to P6 and the total becomes 1.40 GiB with 1.60 GiB spare. Composites themselves remove 419 MiB of forest vertex data on the way |

**And the honest residue, stated rather than hidden.** Three numbers in this plan are estimates and not
measurements, and each of them is a gate before it is a commitment: the large-opaque-quad throughput
rate, the terrain-opaque shading line (2.51 ms bottom-up against a 4.60 ms allowance — the largest
unmeasured number in the design), and the composite and forest fill lines at the depth-rejected
denominator. `bench-frame` and `bench-forest-frame` exist to pin all three, and **nothing in this
package may be spent before they run.** If the terrain line lands at its full allowance the reserve
falls from 1.32 ms to 0.97 ms and the plan still closes. If the large-opaque-quad rate is 20% optimistic
the reserve absorbs it. If both go wrong at once, the first thing to cut is the composite mesh range,
from 150 m to 110 m, which is 0.17 ms and is a tuning field rather than a decision.
