# THE ONE GALAXY, THE REAL SKY, AND THE SEAMLESS WARP — DESIGN OF RECORD

**Date** 2026-08-24 · **Tree** `worktree-warp` @ `1884858` · **Status** proposal, awaiting owner rulings
**Read-only survey.** No build, test, benchmark or binary was run. Every number below is either read
from a named file and line in this tree, or arithmetic whose working is shown. Anything I could not
check is labelled **UNMEASURED** or **UNVERIFIED**.

**THE SPINE I PICKED, in one sentence:**
> **The ruler changes step; the metre never changes meaning.** Every realm's frame declares the unit its
> position lattice counts — millimetres inside a star system, one metre across a galaxy, thirty-two
> kilometres across the universe — so one galaxy holds a hundred and fifty thousand star systems at their
> **true** separations with **no compression anywhere**, and the sky becomes a **static catalogue stated
> once** instead of a roster re-sent fifty times a second.

**WHAT I GRAFTED, AND FROM WHERE.** Three designs and three judges were supplied.

| Taken | From | Why |
|---|---|---|
| The coordinate mechanism (re-value the existing per-frame coordinate tier; a metre stays a metre) | Design B | The tree already carries a per-frame unit tag on the wire (`FrameRef::tier()`, `pose.rs:120-133`), and that file already justifies it as HR3-clean. Design A's alternative — silently redefining what a metre means inside the galaxy — would be a **second** unit mechanism whose unit is invisible in the source (both quantities are `f64`). Both judges converged on this graft. |
| "A star we DRAW and a star we RUN are two different populations" | Design B | It is the correct diagnosis and the direct answer to Q-B. |
| The four-part *not-a-loading-screen* test | Design B | The only runnable instrument either design produced for policing high-speed sequences. |
| The no-pop proof from a shared call site (`generate.rs:420-424`) | Design B | A structural proof beats a margin. |
| Inter-galaxy travel as **physical flight**, needing no seamlessness exception | Design A | Shorter to build, shorter to fly, and needs no amendment to a standing law. |
| Framing the per-tick roster fix as something SL9 **already ordered** | Design A | `CLAUDE.md:96-107` post-dates the wire contract sentence it contradicts. |
| Inverting the interest fold (observers are bounded; children are not) | Design A | Verified viable at `crates/sim/src/stub/aoi.rs:365-372`. |
| The jittered lattice with a one-light-year hard core | Design A | Guarantees minimum separation by construction, with no pairwise search. |
| Insisting every frame-unit ratio be a **power of two** | Design A | `pose.rs:178-185` already explains why: exact idempotent normalisation and an exact integer tier ratio. |
| The reconnect burst; the ~550-player port-band wall; the body lane has no player-squared term; the scale-free 9.61 s warm lead; the mean distance in a ball is 36R/35 not 0.75R | Judge 2 | All re-derived below. |
| Refuse the new catalogue wire arm under SL6; keep the extent tag in the sky | Judge 1 | The local formulation exists and both designs already proposed it in a different paragraph. |
| The client's draw path must subtract **in the lattice** before flattening | Judge 2 | Verified: `crates/client/src/realm_scene.rs:129,849` flatten from the realm origin, then `client-harness/src/camera.rs:196-204` subtracts the eye. At galaxy magnitudes that is a 256 m quantum on the absolute coordinate. |

**WHAT I DROPPED, AND WHY.** Design B's change to the speed law (deriving a realm's ceiling from the
distance between its children instead of its own size) is deleted: I re-derived the journey times from
`crates/core/src/flight.rs` and the shipped law already gives 28 s / 114 s / 198 s. Design B's premise
("one star gap is a 1.5-second blink") is wrong because the two acceleration ramps alone are longer than
one star gap. Design A's dropping of the extent tag from the sky is deleted: it contradicts Design A's
own seamlessness proof. Both designs' request for a new compact catalogue wire arm is refused for now
(§2, ask 4).

---

## 1. OWNER SUMMARY

### What we will build

Your proposal works, and almost none of it needs new machinery. Two things are in the way, and neither
is the one you were worried about.

**The first is a unit, not a size.** Positions in this world are counted in whole steps plus a small
remainder. Today one step is a millimetre. A millimetre-counting ruler runs out at about a quarter of a
light year, which is why the world is currently a quarter of a light year across and holds three stars.
The fix is not to squash a hundred and fifty thousand stars into a quarter of a light year. It is to let
each level of the world count in a step that suits it: millimetres inside a star system, **one whole
metre** across the galaxy, and about thirty-two kilometres across the space between galaxies. Nothing
else changes — the same store, the same whole numbers, the same three-minute crossing rule. At one metre
a step, the galaxy can be two hundred and forty-three light years across, and a hundred and fifty
thousand stars at their **real** spacing need two hundred and thirty-two. It fits, with five per cent to
spare.

This means **there is no compression at all**. Not between systems, not inside them. The gaps are real,
the sizes are real, the brightnesses are physically correct, and the parallax you see when you warp is
the parallax a real galaxy would give you. Today every star in your sky is about twenty-five times too
close, which makes it about twenty-five times too large and its parallax twenty-five times too strong.
This makes that error exactly one. And because nothing is scaled, nothing can change size when you cross
a boundary — there is no scale change available for a jump to happen in.

**The second is a habit, not a limit.** Today the galaxy sends every player the full position of every
one of its children, fifty times a second, whether that child is running or asleep. With three stars that
is invisible. With a hundred and fifty thousand it is about twenty-two megabytes per player per tick,
which two existing size limits refuse and which would need nearly a million network packets a second per
player. Worse, the message from the galaxy to the gateway is not split into packets at all, so it would
simply not arrive — and the failure would be a **silent empty sky**, not an error.

The cure is to stop confusing two different populations. The **catalogue** is every star in the galaxy:
where it is, how big, what colour, how bright. It never moves, so it is the same for every player
forever. It is stated **once**, and then costs nothing. The **live set** is what actually runs: the
system you are in and the one you are flying to. That is the machinery you already have, unchanged.

Everything else in your proposal — the shrinking star, the parallax cruise, the growing star at the far
end, no blink anywhere — falls out of laws that are already written. The only thing I am proposing to
change about the *feel* of it is the very start of the galaxy leg: you asked to start at zero speed;
zero would itself be a jump. Starting at exactly the speed you left the system with, and accelerating
from there, is the seamless version of the same intent.

### Answer to your first question — does one galaxy of 150,000 systems fit?

**Not today, and not by compressing. Yes, once each level counts in its own step.** A hundred and fifty
thousand stars at their real spacing need a ball two hundred and thirty-two light years across the
radius; today's whole universe is a quarter of a light year. Squashing them in would need a factor of
about nine hundred and seventy-four — and that is the trap: squashing does not make them dots, it makes
them **neighbours**. A star system wakes up and starts running when it grows past one and a half degrees
across. At the squashed spacing about eight hundred systems would demand to run around **every single
player**, and the whole cluster can currently start about five hundred processes in total. Compression is
precisely what destroys the picture you asked for. Counting in whole metres at galaxy level costs
nothing, keeps every distance true, and each star then sits about a hundred and seventy times further
away than the distance at which it would wake — so it stays a dot always, by construction, never by a
rule that hides it.

### Answer to your second question — many players in one galaxy at once?

**Yes, and the worry rests on a premise that turns out to be false.** You wrote that we will not be able
to include all the destination realms for all the players. We never needed to. **Drawing never required
running.** Every star is drawn, always, from a list sent once. Only the system you are in and the one you
are arriving at ever run. The number of running systems is bounded by the number of *players*, not by the
size of the galaxy — at most one destination each, and two players heading for the same star share one.

There is also a guarantee here rather than a hope: the arrival slow-down law gives you **9.6 seconds** of
warning before you touch a destination's edge, at *every* scale, whatever the realm's size — and the
machinery needs 1.1 seconds to wake a realm. That is an 8.7-fold margin that does not shrink as the world
grows.

What actually runs out first is none of these. It is the number of separate processes the cluster can
start: about five hundred today, which binds at roughly **five hundred and fifty concurrent players**.
That is a machine-count problem with a machine-count answer, and it is the same problem the game would
have with three stars.

---

## 2. OPEN QUESTIONS FOR THE OWNER — read first

Column *"really yours?"* is honest about which of these is taste and which is engineering wearing a
question mark.

| # | The question, plainly | Options | Default if you say nothing | Blocks | Really yours? |
|---|---|---|---|---|---|
| **1** | May the galaxy and the universe count their positions in bigger steps than a millimetre — one metre and thirty-two kilometres respectively? Nothing new crosses any boundary; the same whole numbers travel, and the step size is already carried with every position. It changes the shape of one item in the frozen message list (**SL6 ask**). | (a) yes; (b) no — stay at millimetres and accept either three stars or a 974-fold squash | **(a) yes** | Everything. Slice 1. | **Engineering.** There is no version of your proposal that works without it. |
| **2** | May a star's position be sent **when it changes** instead of fifty times a second? A star never moves, so it would be sent once. This strictly *reduces* what crosses, but it edits one sentence of the shipped message contract (**SL6 ask**). | (a) yes; (b) no | **(a) yes** | The sky. Slice 3. | **Engineering.** Your own law SL9 (24 Aug) already says "never a per-child row on a per-tick lane". This needs scheduling, not permission. |
| **3** | How far apart must the two closest star systems be allowed to get? This sets the heaviest star the world may contain. | (a) **one light year** → heaviest star about 6.1 suns (loses O stars and supergiants); (b) two light years → about 8.9 suns, but the star field becomes noticeably more regular (a tenth of the volume is forbidden instead of a eightieth); (c) more → the sky looks crystalline | **(a) one light year** — the real closest-neighbour distance between independent systems, and an exact number already in the tree | Slice 2. | **Yours.** It decides which stars exist and how random the sky looks. |
| **4** | Should the star catalogue get its own compact message (about 15–26 bytes a star, ~3 MB total), or ride the existing message list at ~185 bytes a star (~28 MB, sent once and cached)? (**SL6 ask**) | (a) no new message — use what exists; (b) add a compact one | **(a) no new message.** Default on a new crossing datum is no, the local formulation exists, and a second mechanism doing the first mechanism's job breaks "one tooling" | Nothing — reopenable later on a measurement | **Engineering.** The measurement that reopens it: total bytes when a thousand players reconnect at once (see §8). |
| **5** | Travel between galaxies: your scripted transition, or the same physical flight one level up? | (a) **physical flight** — you fly out of your galaxy exactly as you fly out of a star system; it shrinks to a point of light behind you; about 57 seconds to the nearest galaxy, with real parallax; needs **no** change to the seamlessness law; (b) your scripted version — relay station, star map blending, a wait sized by distance — which needs a written, named **exception to SL8** | **(a) physical flight** | Slice 6 only | **Yours.** (a) is cheaper and needs no exception, so I recommend it — but the feel is your call. |
| **6** | If (5b): may we write the exception down as "exactly one world-metric discontinuity is lawful — passage between galaxies — and nothing else may cite it"? | (a) yes, written; (b) argue it away case by case | **(a) written** if (5b) is chosen | Slice 6 | **Yours**, but an unwritten exception becomes a precedent. |
| **7** | Your ruling of 5 Aug (D-SCALE-1) says compression applies between realms, the galaxy is a chart rather than a scale model, and the galaxy is a **lattice of cell-realms**. This design makes compression exactly 1.000000 (the chart *becomes* the model) and drops the cell lattice, which your own 24 Aug proposal supersedes. Confirm the reversal? | (a) confirmed; (b) keep the cell lattice | **(a) confirmed** | Slices 1–2 | **Yours.** A design that quietly contradicts a written ruling is a defect; this needs a spoken yes. |
| **8** | How many galaxies should the universe hold? | 8, **61**, or 492 (these are the three neighbouring powers of two for the step size; nothing in between is representable exactly) | **61** — a Local-Group-scale neighbourhood, 8 million light years across | Slice 6 | **Yours.** |
| **9** | The heaviest star drops (see 3). Accept losing O stars and supergiants from the world? | (a) accept; (b) revisit by widening the hard core (question 3) | **(a) accept**, and re-solve the exact figure with the code's own solver before building | Slice 2 | **Yours.** |

**Every SL6 ask in one place** (default is always NO; three of these move strictly *less* data):

- **Ask 1 — the coordinate step.** *What:* each realm frame declares its lattice unit; the galaxy's is one
  metre, the universe's 2^15 m. *From which realm to which:* **none.** No datum crosses any boundary. The
  same whole-number position crosses; only the unit its steps count changes, and that unit is already
  carried with every position (`FrameRef::tier()`, `crates/core/src/pose.rs:120-133`) and is already
  described there as "a coordinate unit, nothing else". *Why the receiver cannot compute it:* it can, and
  does — it is a pure function of the frame kind already on the wire. *The wire change:* the existing
  `FrameRef::GalaxySpace` arm must gain a galaxy seed and become realm-bearing, and a universe arm must be
  appended. **Measured: this is free today.** `GalaxySpace.realm()` returns `None`
  (`pose.rs:88`), `frame_for_realm` has **no** arm that produces it (`pose.rs:146-162`), and no coarse-tier
  position exists anywhere — so changing that arm's shape changes zero bytes on every message the system
  currently produces. It will not be free after the first galaxy-scale position ships. *Cost of no:* the
  world stays at three stars, or accepts today's 24.568-fold distortion.
- **Ask 2 — placements become send-on-change.** *What:* a child's position is stated when its **value**
  differs from the last stated one (a pure comparison — never a test of whether the child "has motion",
  which SL4 forbids). *Direction:* strictly **less** data than today. A static star is stated once instead
  of twenty times a second. *The catch, stated:* the per-tick lane is a drop lane, and your law of 15 Aug
  says only full state may ride a drop lane. So the *unchanged* placements move to the existing reliable,
  re-driven lane the appearance bags already ride (one appended message arm, attested identically), and the
  drop lane keeps full state **of what changed** — which is still full state for its own set, and a thing
  that changes every tick self-heals next tick exactly as the law intends. *Cost of no:* 150,000 rows ×
  145 B = 21.75 MB in one un-split datagram, silently dropped and counted — an empty sky with no error.
- **Ask 3 — the keep-alive stops clearing the send-on-change memory.** *What:* an 8-byte generation counter
  replaces a full re-serve every half second (`crates/sim/src/stub/window.rs:83-89,124-133`). *Why not
  local:* the receiver cannot know the sender's set is unchanged without being told something; a counter is
  the smallest possible something and carries no world data. *Cost of no:* every open window re-ships the
  whole catalogue twice a second — 150,000 × 59 B = 8.85 MB per beat, 17.7 MB/s per window, forever.
- **Ask 4 — a compact catalogue message.** **RAISED AND RECOMMENDED FOR REFUSAL** (question 4 above).
- **Ask 5 — the universe frame becomes a running realm with children.** Only if question 5 is answered (a).
  *What crosses:* nothing new — a galaxy states its own appearance and one occupancy bit upward, exactly as
  a star system does today. *Cost of no:* no inter-galaxy travel at all, or the scripted version.

**Nothing else in this design puts a new datum across a realm boundary.**

---

## 3. THE NESTING AND THE FRAMES

The owner's ladder, made concrete. "Step" is the unit one whole number of the position lattice counts;
the remainder inside one step is a floating-point number and is exact to the precision shown.

| Level | Frame | Step (one lattice cell) | Largest representable radius | Remainder precision | What it may state about a child | What crosses its boundary |
|---|---|---|---|---|---|---|
| **Universe** | new `UniverseSpace` arm | **2^15 m = 32,768 m** | 2^61 × 2^15 = **2^76 m = 7.556e22 m = 7.99 Mly** | ≤ 2^-38 m | a placement + (extent, colour, brightness) | an occupant crossing; one occupancy bit up |
| **Galaxy** | `GalaxySpace`, gains a galaxy seed | **2^0 = 1 m** | 2^61 = **2.3058e18 m = 243.7 ly** | ≤ 2^-53 m ≈ 1.1e-16 m | a placement + (extent, colour, brightness) | as above |
| **Star system** | `SystemSpace` (unchanged) | 2^-10 m (0.9766 mm) | 2^51 = 2.2518e15 m | ≤ 2^-63 m | a placement + (extent, colour, brightness) | as above |
| **Planet / Star / Ship / Station** | `PlanetCentered` / `StarCentered` / `ShipLocal` / `StationLocal` (unchanged) | 2^-10 m | 2^51 m | ≤ 2^-63 m | a placement + (extent, …) | as above |
| **Area** (optional, and again) | `AreaLocal` (unchanged) | 2^-10 m | 2^51 m | ≤ 2^-63 m | a placement + (extent, …) | as above |

**Every frame at or below a star system keeps the millimetre step it has today, byte-for-byte.** Only two
frames get a new one, and both of them are frames nothing has ever been drawn from.

**The conversion at a boundary lives in the parent** — where SL1 already puts every conversion. Crossing
down (galaxy → system) multiplies the whole-number part by exactly 1024 and shifts the remainder: **exact,
no rounding at all**, because 1024 is a power of two and the remainder is scaled by a power of two.
Crossing up (system → galaxy) divides by 1024 and carries the remainder: exact in the whole-number part,
and the combined remainder rounds at **2^-63 m ≈ 0.1 attometres** — a tenth of a thousandth of a proton
diameter. *That* is the whole numerical price of the ladder, and its source is one floating-point step at
the finer tier's own quantum, which is SL8's own form of a tolerance.

**One simplification falls out.** Today the millimetre-to-light-year ratio exceeds one machine word and is
declared as a double-width integer for exactly that reason (`FINE_CELLS_PER_LY: i128`, `pose.rs:185`). At a
one-metre galaxy step the ratio is exactly **1024** — an ordinary bit shift. The light-year constant that
currently *is* the galaxy step (`COARSE_CELL_EDGE_M`, `pose.rs:196-201`, doc: "Planted, value revisable at
P10") keeps its value but moves to its own name, because a light year is still needed — as the hard core in
§4, not as a coordinate unit.

**A realm still never learns its own position.** It knows its own size in its own units, which it already
does. Nothing is folded from the root (SL1).

---

## 4. THE COORDINATE SOLVE

### 4.1 The wall, re-derived from the tree

The per-axis limit is a clamp applied at wire ingress: `CELL_DOMAIN_MAX = i64::MAX / 2 = 2^62 − 1`
(`crates/core/src/pose.rs:193`), chosen so a *difference* of two positions cannot overflow. The boot fence
`guard_root_representable` (`crates/physics/src/worldgen/guards.rs:66-89`) then requires

```
K_SPAN · R / cell_edge  ≤  CELL_DOMAIN_MAX + 1        with K_SPAN = 2 (scale.rs:68)
```

At the millimetre step (`FINE_CELL_EDGE_M = 2^-10`, `pose.rs:178`) this gives `R ≤ 2^61 × 2^-10 = 2^51 m`
— exactly the shipped universe radius `REAL_UNIVERSE_R_M = 2_251_799_813_685_248.0` (`scale.rs:63`),
passing with **exact equality** (2 × 2^61 = 2^62 = `CELL_DOMAIN_MAX` + 1). The world is exactly half the
domain, by construction, not by luck.

### 4.2 What 150,000 systems actually need

Real stellar density, from the tree's own census constants (`scale.rs:331-335`):

```
n  =  0.1 / PARSEC_M³         PARSEC_M = 3.085677581491367e16   (scale.rs)
PARSEC_M³ = 2.93800e49 m³   ⇒   n = 3.403676e-51 m⁻³
```

Radius of a ball holding N systems at that density:

```
R = (3N / 4πn)^(1/3)
N = 150,000 :  V = 150000 / 3.403676e-51 = 4.406916e55 m³
               R³ = V / 4.188790 = 1.052076e55
               R = 2.19123e18 m  =  231.62 light years
N = 100,000 :  R = 1.91327e18 m  =  202.2 light years
```

Against today's galaxy radius `REAL_GALAXY_R_M = 2.2487974139336678e15` (`scale.rs:84-88`), the 150,000
figure is **974.4× too large**. Q-A's premise is correct.

*Cross-check against a real census, independent of the formula:* the nearest 10 parsecs hold about 350
stars; 150,000 / 350 = 428; the cube root of 428 is 7.54; 7.54 × 10 pc = 75 pc = 246 ly. Within 6 % of
231.6 ly. The figure stands.

### 4.3 The solve — the galaxy step is ONE METRE, and it is forced

Rearranging the same fence for the step:

```
cell_edge  ≥  K_SPAN · R / (CELL_DOMAIN_MAX + 1)  =  2 × 2.19123e18 / 2^62  =  0.95032 m
```

The step must be a power of two (`pose.rs:178-185`: only a power-of-two edge makes the normalising
constructor exactly idempotent, and only a power-of-two ratio makes tier conversion exact integer
arithmetic). The smallest power of two above 0.95032 is **2^0 = 1 m**. And the neighbours are not close:

| step | largest galaxy radius | systems it holds at real density |
|---|---|---|
| 0.5 m | 2^60 = 1.15292e18 m = 121.9 ly | **21,849** — 7× too few |
| **1 m** | 2^61 = 2.30584e18 m = **243.7 ly** | **174,791** ceiling; **150,000 fits with 5.23 % of radius to spare** |
| 2 m | 2^62 = 4.61169e18 m = 487.4 ly | **1,398,328** — 9× too many |

**One metre is the unique power of two admitting the owner's stated census band of 100,000–150,000.**
It is solved from (the storage fence) × (the measured stellar density) × (the requested count). Nothing
here is chosen.

*The galaxy's own radius* is set to `2^61 m` exactly — the fence's equality, the same construction today's
world uses one tier down — and the **populated** radius is whatever the census fills: 2.19123e18 m at
150,000. The 5.23 % of empty rim is the clearance the nesting law needs, and it is far more than the
clearance the largest child demands (§4.6).

### 4.4 The universe step

The universe holds galaxies at the real large-galaxy separation (~1 Mpc = 3.085678e22 m), giving
`n_gal = 1 / (1 Mpc)³ = 3.403676e-68 m⁻³`. Same fence, same power-of-two rule:

| step | universe radius | galaxies |
|---|---|---|
| 2^14 = 16,384 m | 2^75 = 3.7779e22 m | **7.7** |
| **2^15 = 32,768 m** | **2^76 = 7.5558e22 m = 7.99 Mly** | **61.5** |
| 2^16 = 65,536 m | 2^77 = 1.5112e23 m | **492** |

Default **2^15**, giving a Local-Group-scale neighbourhood of about **61 galaxies** — roughly 10.6 million
star systems in the whole world. That is owner question 8. Note that at every tier the radius is `2^61 ×
step`, so **the storage fence passes with exact equality at every level**, by the same construction, and
`guard_root_representable` must simply be taught to read the root frame's own step instead of the hard-coded
millimetre (`guards.rs:69`: `let edge = vd_core::pose::FINE_CELL_EDGE_M;`). **This is a required change and
neither design named it** — today that fence would refuse the new world by a factor of 1,024. Its own error
text already names this design as the cure: *"the world has outgrown the millimetre tier; the cure is the
galaxy cell lattice (P10), never a widened clamp"* (`guards.rs:32-37`).

### 4.5 The compression factor is exactly one — so there is nothing to prove about the boundary

The brief asks whether a uniform compression is a similarity transform, and what it costs. **This design
does not compress at all**, so the question does not arise:

- `real_compression_chi()` (`scale.rs:321-330`) is the real mean nearest-neighbour separation over the
  world's own placement radius. Real mean nearest neighbour = `0.55396 × 0.1^(-1/3) × PARSEC_M =
  3.682666e16 m = 3.8926 ly`. Under this design the world's mean nearest neighbour **is** that number.
  **χ = 1.000000 exactly**, against 24.568 today.
- In-system compression is already exactly 1.000000 and the field that would hold it has been deleted
  (`crates/physics/src/worldgen/config.rs:70-73`: *"au_to_render_m is DELETED — in-system compression is
  1.000000 EXACTLY, and the field does not exist to be anything else"*).

**WHAT HAPPENS TO APPARENT SIZE AT A SYSTEM BOUNDARY, AND WHY IT CANNOT JUMP.** Nothing happens, and it is
structural rather than careful, for three independent reasons — any one of which would be sufficient:

1. **No length is scaled anywhere.** Crossing a boundary changes the frame a position is *expressed in*, not
   the metre it is measured in. Apparent size is one length divided by another; neither changed.
2. **The size a system draws at from outside and the size its star draws at from inside are the same number
   from the same function.** `crates/physics/src/worldgen/generate.rs:420-424`, verbatim: *"THE SYSTEM'S
   LOOK IS ITS STAR: `star_radius_m` of the drawn mass — ONE function, TWO call sites … so the marker→body
   handover is the same radius at every depth."*
3. **By the time you reach a system's edge, everything inside it except the star's disc has already faded at
   its own distance** (§6, step 1). The crossing line is placed where nothing is left to change.

For the record, since the brief asks: *if* one had compressed uniformly, dividing every length in the galaxy
frame — including the traveller's own path — by the same power of two **would** leave every angle and every
angular rate unchanged, so the parallax would be genuine and the boundary would not zoom. That argument is
correct. It is also unnecessary, and it costs three things a design should not pay: it makes a metre inside
the galaxy mean 1,024 real metres while still being called a metre (the sharpest possible foot-gun in a
language where both are `f64`); it makes the occupant's walking speed mean 512 km/s in that frame; and it
duplicates a unit mechanism the tree already carries on the wire. Hence the spine.

### 4.6 Placement, the hard core, and the mass cap

**The placement law must become volumetric before anything else.** Today every non-home system sits at
**exactly one radius** — a sphere shell, direction seeded, magnitude fixed
(`generate.rs:197-233`, `config.stellar.system_ring_r_m`). A shell's capacity grows as radius squared
while the count grows linearly, so **no radius rescues it**: at 150,000 systems on today's shell the mean
neighbour gap would be 6.86e12 m, already smaller than one home system's own diameter (1.0177e13 m).

**Replace it with a jittered lattice.** Cell edge `a = n^(-1/3) = 6.647705e16 m` (one system per cell,
which *is* the density), position drawn uniformly inside a concentric sub-cube of edge `(1−γ)·a`. Two
points in adjacent cells are then at least `γ·a` apart along the shared axis, so the **minimum separation
is guaranteed by construction with no pairwise search at all**. Set `γ·a` = **one light year** =
9,460,730,472,580,800 m (an exact constant already in this tree, `pose.rs:196-201`) — the real scale of
the closest separations between independent stellar systems. Then

```
γ = 9.4607305e15 / 6.647705e16 = 0.142315
excluded volume = (4π/3)·γ³ = 4.188790 × 0.0028828 = 1.21 %
```

The hard core barely perturbs the field, so the sky should not look regular. *(How regular it actually
looks is **UNMEASURED** — the mean nearest neighbour of an 85.8 %-jittered lattice lies somewhere between
the Poisson value 0.554a = 3.6827e16 m and the full pitch a = 6.6477e16 m, and I could not sample it.
Sample it and look at the sky before shipping.)*

**The mass cap must be re-solved downward, and this is the design's real price.** I read the two halves of
the fence:

- The **geometric** half is what the code actually tests: `seeded_systems_disjoint_3d`
  (`guards.rs:262-277`) refuses any pair whose centre distance is below the sum of their circumscribed
  extents. With a 1-ly hard core this allows a system shell of `9.4607e15 / 2 = 4.73e15 m` — no constraint
  at all.
- The **wake** half is what the doc beside it names ("every pair must be separated by more than one
  system's AoI spin-up reach, so a system is ASLEEP at departure from any sibling") and what the code does
  **not** yet test. The reach is `visibility_factor(θ) × the body's own finite extent` — I confirmed the
  band is built from the *shell*, not the star (`crates/physics/src/worldgen/body.rs:178-181`:
  `config.interest.build(b.shape.finite_extent(), v_child)`), and `visibility_factor(0.026180) =
  1/tan(0.013090) = 76.39017` (`crates/core/src/geometry.rs:835`, `scale.rs:40`). So

```
76.39017 · shell  ≤  9.4607305e15      ⇒  shell ≤ 1.238474e14 m
```

against today's reservation of 749,489,793,576,937.9 m — **6.05× smaller**. Fitting the code's own two
pinned points (0.16179875 M☉ → 1.582054e11 m; 16.360035 M☉ → 7.494898e14 m) gives shell ∝ M^1.833359, so

```
M_cap ≈ 16.360035 × 6.0517^(-0.545447) ≈ 6.13 solar masses          ← UNMEASURED
```

**UNMEASURED and it must not be believed as stated.** It is a two-point power-law fit; the authority is
`solve_mass_cap`'s own monotone bisection (`scale.rs:255-276`), re-run against the new inequality, and I
was forbidden to run it. The consequence if it holds: the world keeps every G, K, M, F, A and most B
stars, and loses O stars and supergiants. Owner questions 3 and 9.

*(Design B put this figure at about 12.8 solar masses by dividing the **mean** separation rather than the
**worst-case** one. The fence must hold for every pair, so Design A's ~6.1 is the one I believe; Judge 2
independently reached the same conclusion.)*

### 4.7 What clearance remains, and the smallest step at the rim

| Quantity | Value | How obtained |
|---|---|---|
| Galaxy radius vs the storage fence | **exact equality** (2 × 2^61 cells = 2^62) | by construction, §4.3 |
| Populated radius vs galaxy radius | 2.19123e18 / 2.30584e18 = **95.03 %** — 5.23 % of rim to spare | §4.2, §4.3 |
| Largest child's clearance demand vs that rim | `max(shell, look·77.39) + look·77.39` ≈ 1.24e14 m against 1.146e17 m of rim | §4.6 + `child_clearance_m`, `scale.rs:340-346` |
| Each star's distance vs its own wake radius | pitch 6.6477e16 / reach (76.39017 × 5.0886e12) 3.88718e14 = **171×** | §4.6; home shell from `crates/bins/tests/warp_pixels.rs:106` |
| Systems above the wake threshold per observer in open space | (4π/3)(1/171)³ = **8.4e-7** — effectively zero | same |
| Drawn-position step at the galaxy rim | 2.19123e18 lies in [2^60, 2^61), so the whole-number-to-float step is 2^8 = 256 cells = **256 m** | `Separation::metres`, `pose.rs:470-478` |
| That step, as an angle against the nearest star | 256 / 6.6477e16 = 3.85e-15 rad = **3.6e-12 of a pixel** | one pixel at 60° over 1080 rows = 1.06917e-3 rad |

### 4.8 A required change neither design named: subtract in the lattice, then flatten

**Verified defect in the shipped draw path.** Every client draw site flattens from the **realm origin** —
`center.delta_m(LatticePos::ORIGIN, tier)` (`crates/client/src/realm_scene.rs:129` and `:849`, and the same
shape at `render_snapshot.rs:190,481`, `interp.rs:392`, `view.rs:400,409`) — and only *then* subtracts the
eye in `f64` (`crates/client-harness/src/camera.rs:196-204`). So each position independently carries one
float step **of the absolute coordinate**: 0.5 m today at 2.25e15 m (the `Separation::metres` doc says so
in as many words), and **256 m** at 2.19e18 m.

For distant stars this is harmless (3.6e-12 pixel). For two ships flying in convoy in the galaxy frame it
is fatal: each rounds independently to 256 m, so a 100 m separation is noise. And at the slowest
commandable galaxy speed it is a staircase — geometric throttle at 0.001 stick gives
`0.001 × 500 × (2.562048e16/500)^0.001 = 0.516 m/s = 0.0103 m/tick`, one 24,806th of a 256 m step.

**The cure is already in the tree and is one call away**: `LatticePos::separation` differences the
whole-number parts as **integers** first, and `Separation::metres`' own doc states the result is exact
while the cell difference stays under 2^53. Move the eye subtraction into the lattice — separate, then
flatten — and the drawn error becomes relative to the *answer* instead of to the world. This is a
precondition for Slice 3 and it also repairs a latent defect that exists today at a thousandth of the
magnitude.

---

## 5. THE SKY — 150,000 points of light

### 5.1 The two populations

**THE CATALOGUE** — every system in the galaxy, as a point of light: its identity, the placement the
galaxy authored for it, its circumscribed extent, and its colour and brightness. Three facts make it
nearly free, and each is verified:

- **It never changes.** A star system is a *static* child; the placement path has exactly two arms, run
  the motion function or use the stored centre, and a system takes the second.
- **It is the same for every observer.** The body set is built **once per realm per tick**
  (`crates/sim/src/stub/window.rs:299-337, 349-400`) and is deliberately never filtered by who is looking
  (`crates/wire/src/session_flow.rs:284-286`: *"marker/placement rows always ship … stars stay in the sky
  by construction"*).
- **It is a pure consequence of the seed.**

So it is **stated once**, and cached by the client under (world seed, galaxy, catalogue generation).
After that: zero bytes per tick, forever.

**THE LIVE SET** — the realms the demand loop actually wakes: the system you are in, the one you are
arriving at, and their insides. That machinery is unchanged, and §4.7 proves it stays at about one system
per traveller.

### 5.2 What is sent, when, and what it costs

Row sizes are **hand-derived from the type definitions** (`crates/wire/src/channels.rs:297-370`,
`crates/core/src/pose.rs:242-246,544-552`) and carry roughly ±30 % — no conclusion below turns on that
margin. A composed scene row is ~185 B (identity 11 + parent 12 + position 123 + appearance bag 39); a
placement row is ~145 B; an appearance bag is 38 B glowing, 23 B not (`crates/core/src/tlv.rs:37,39` +
`crates/core/src/look.rs:86-110`).

| | Today (3 stars) | Naive at 150,000 | This design |
|---|---|---|---|
| Sky, at login | ~19 rows | 27.75 MB in one frame — **refused**, 26.5× over the 1,048,576 B cap (`crates/wire/src/framing.rs:17`) | 27.75 MB in **27 chunked reliable frames**, encoded **once per gateway**, cached client-side by generation; a returning client sends and receives nothing |
| Sky, per tick | full roster | 21.75 MB per observer per tick = **435 MB/s** at the 20 Hz realm-lane cadence; 18,750 datagrams per tick | **0 bytes.** A star's placement was already sent |
| Shard → gateway, per tick | one message | 21.75 MB in **one un-split datagram** (`window.rs:265-282`, `rows: realms.to_vec()`, class `RealmSnapshot` = Unreliable, `crates/sim/src/io/mod.rs:120-130`) — **dropped and counted, never raised as an error** | the changed set only; partitioned by the existing shared partitioner |
| Live rows, per tick | ~19 × 145 B | same | ~13–19 rows = 1.9–2.8 kB/tick = **38–56 kB/s** — the same order as today, and **independent of the galaxy's star count** |
| Keep-alive | full re-serve every 0.5 s | 8.85 MB per beat, 17.7 MB/s per window | one generation compare (ask 3) |
| Gateway memory | trivial | 51 retained stamps × 150,000 rows ≈ **1.6 GB per window** and **2.0 GB per session** (`crates/connection-plane/src/window.rs:70-80, 1010-1026`) | the ring retains the live/changed set only; the catalogue is **one shared read-only block per gateway** (~28 MB), not 51 copies per session |
| Client | one entity per drawn box, transform rewritten every frame (`crates/client-render/src/lib.rs:531-536,745-800`) | 150,000 entities — a tick hitch, i.e. an SL8 defect | **one instanced point cloud**, ~2.25 MB of GPU memory, one draw call — the owner's own instinct ("they even can be rendered by shaders, we don't need any objects there") |

**The most important operational fact in this document:** the failure mode of doing this half-way is
**silence**, not an error. The shard-to-gateway level message is unpartitioned and rides a drop lane whose
oversize datagrams are dropped into a counter. A galaxy with 150,000 children and today's emit produces an
**empty sky with nothing in any log**. Every one of the rewrites in §8 is load-bearing; none is an
optimisation.

### 5.3 How a point becomes a running realm with no pop

Four reasons, each structural rather than careful:

1. **The catalogue is stated in the galaxy's own frame and is not restated at a crossing.** It does not move.
2. **The drawn radius before and after the handover is the same number from the same function**
   (`generate.rs:420-424`, quoted in §4.5). The colour and brightness likewise come from one seeded pair —
   `crates/core/src/look.rs:57-79` exists precisely so *"a star KEEPS its colour through the marker→body wake
   handover instead of turning grey"*.
3. **Which of the two is drawn is decided by DATA PRESENCE, never by a flag.** A realm's own outline beats
   its parent's point of light because the outline is *there*; the two are separate wire arms and
   `look_of()` on a marker bag returns a missing-tag error, pinned by a test
   (`crates/wire/src/session_flow.rs:444-457`, `crates/core/src/look.rs:174-181`). A flag arriving late
   cannot mistime the swap, because there is no flag.
4. **The swap happens while the object is a two-hundred-and-ninety-ninth of a pixel wide.** A system wakes at
   76.39017 × its shell = 3.887e14 m; it is one pixel wide at `2 × 6.957e8 / 1.06917e-3 = 1.30e12 m`. The
   ratio is 299. Even a gross radius error would be invisible — and there is no error, because it is the same
   number.

**The extent tag stays in the catalogue.** Design A proposed dropping it and estimating a dot's size
photometrically; that replaces a bit-equal radius with an estimate at exactly the instant SL8 names as a
seam. Eight bytes a row is the wrong thing to save.

**No star is ever filtered out of anyone's sky.** Every one of the 150,000 is drawn, always. What changes
is the lane: stated once instead of twenty times a second. No membership test is applied to a point of
light — the shipped rule (`session_flow.rs:284-286`) is preserved in meaning and improved in cost.

---

## 6. TRAVEL INSIDE A GALAXY — the seamless warp, step by step

All distances are true metres. Constants: τ = **1.10 s** (the demand pipeline's own wake budget, 55 ticks
× 0.02 s — `FlightTuning::derive`, `crates/core/src/flight.rs:59-79`; a *measured latency*, never a chosen
number). Traverse policy T = **180 s** (`flight.rs:32`). Realm ceiling = `max(v_foot, 2·extent/T)`
(`flight.rs:93`). Home system shell = 5.0886e12 m (`crates/bins/tests/warp_pixels.rs:106`), so its ceiling
is 5.65400e10 m/s. Galaxy ceiling = 2 × 2.30584e18 / 180 = **2.562048e16 m/s**.

**STEP 0 — standing in a system.** The star and planets are running realms drawing themselves. Behind them,
150,000 catalogue points. While you move anywhere inside the system the whole sky shifts by at most
`5.0886e12 / 6.6477e16 = 7.65e-5 rad = 0.072 of a pixel`. **Physical quantity:** the ratio of the system's
own radius to the nearest star gap. The star map is rock steady inside a system for the real reason, not
because anything froze it.

**STEP 1 — departure. "The star shrinks and the planets are left behind."** Your ceiling is the system's
own, 5.654e10 m/s, reached from a standstill in `τ·ln(5.654e10/500) = 1.10 × 18.545 = 20.4 s`. Flying from
1 AU out to the shell, the star's angular size falls from `2 × 6.957e8/1.496e11 = 9.30e-3 rad` to
`2 × 6.957e8/5.0886e12 = 2.73e-4 rad` — it shrinks by a factor of **34**, continuously, purely because a
distance changed. Each planet's realm tears down when it passes 76.39017 × its own sphere of influence,
which for the outer planet is well inside the system shell. **Nothing tears down because you crossed a
line. Everything tears down because a distance grew.**

**STEP 2 — the hand-over up.** You cross the shell; the re-home saga runs unchanged (SL4: it consumes a
placement and cannot ask how you move). Your position and velocity are re-expressed in the galaxy's frame:
whole-number part divided by 1024 with the remainder carried, remainder rounding at 2^-63 m. **Nothing on
screen moves**, for the four reasons in §5.3, and the numerical error at the crossing is
0.1 attometres. **Tolerance source:** one float step at the millimetre tier's own quantum.

**STEP 3 — the starfield does not blink.** The sky is not re-sent. The catalogue is keyed to the galaxy,
not to your origin realm, and survives the origin change. What is re-sent is your own position, one row.
**A required guard:** a crossing bumps the scene epoch and re-sends the composed level; the catalogue must
be excluded from that re-send, or a warp leg (two crossings) re-transmits 55.5 MB per traveller. That is a
gate, listed in §10.

**STEP 4 — the ramp up.** The galaxy ramps you at a constant proportional rate `dv/dt = v/τ`
(`ramp_cap_mps`, `flight.rs:117-121`) from your arrival speed 5.654e10 to 2.562048e16 m/s:
`τ·ln(2.562048e16/5.654e10) = 1.10 × 13.0239 = **14.33 s**`, covering `τ·(v_cap − v_start) = 2.8182e16 m`.
**Tolerance source:** none is chosen — the ramp's only constant is τ, the pipeline's own wake budget, and
the starting speed is the departure realm's own ceiling. *Note on the owner's "start with the speed 0":
zero would itself be a jump. Starting at exactly the exit speed is the seamless version of the same
intent, and the visible acceleration is a full fourteen seconds.*

**STEP 5 — the parallax cruise.** You hold 2.562048e16 m/s and pass a star every `6.6477e16/2.562048e16 =
2.59 s`. Because no length anywhere was scaled, the angular sweep of every star is exactly what flying the
real galaxy at that speed would show — **the parallax is not simulated; it is the real thing**. Travelling
one gap swings a star two gaps abeam through `arctan(1/2) = 26.6°`; a star a hundred gaps away swings
0.57°. Near sky sweeps, far sky is nearly fixed. Every star is 171× beyond its own wake radius, so nothing
spins up and the whole sky is one instanced draw. **Per-tick sky cost during the cruise: zero bytes.**
**Tolerance source:** the drawn-position step at the galaxy rim is one float step of the answer once §4.8
lands — 3.6e-12 of a pixel measured against one pixel, which is SL8's own form.

**STEP 6 — the approach.** The shipped governor (`approach_ceiling_mps`, `flight.rs:86-89`) sets your
ceiling to `child_cap + distance/τ`. **It needs no change at all**, and here is the structural payoff:
because the ceiling function is homogeneous of degree one in length, the child cap it computes is the
destination's own ceiling, at any tier, automatically. The governor starts biting at
`τ·v_cap = 2.8182e16 m` — one ramp length out — and delivers you to the shell at exactly the destination's
own ceiling.

**THE WARM LEAD IS A CONSTANT, AND IT IS THE STRUCTURAL ANSWER TO Q-B.** From the wake radius inward the
governor's closed form is `τ·ln(1 + 76.39017·E/(τ·v_child))` with `v_child = 2E/T`. **The extent E cancels
completely:**

```
τ · ln(1 + 76.39017 × 180 / (2 × 1.10))  =  1.10 × ln(6251.10)  =  1.10 × 8.74045  =  9.61 s
```

**9.61 seconds of warning before you touch any destination's edge, at every scale, in any unit, for every
realm above the 45 km break-even extent** — against the 1.10 s the demand pipeline needs to wake one.
An 8.74-fold margin that does not shrink as the world grows. *(Below the break-even the ceiling is the foot
speed and the lead stays in the same band; the exact figure there is **UNMEASURED**.)*

**STEP 7 — the hand-over down and arrival.** Identical to step 2 with the conversion inverted — and this
direction is **exact, with no rounding at all**, because multiplying the whole-number part by 1024 and the
remainder by a power of two is exact in binary. The destination is already running and already drawing
itself (it woke 9.61 s ago). The governor decelerates you onto the star and planets, which grow from dots
exactly as the owner described: the star is one pixel wide at 1.30e12 m, which is 0.26 of the system shell
— **inside** the system, with the star already a running realm.

### Journey times, all derived from `flight.rs`

Using `leg_time_s`'s law: ramp in over `τ·(v_cap − v_start)`, cruise, ramp out.

| Leg | Distance | Time | Working |
|---|---|---|---|
| Nearest neighbour | 3.68e16 – 6.65e16 m | **27.7 – 29.1 s** | two ramps cover 5.6365e16 m; at the Poisson value it is a short leg (each half `τ·ln(1 + 1.8414e16/(1.10×5.654e10))` = 13.86 s); at the full pitch there is 0.39 s of cruise |
| **Typical destination** | mean distance between two uniform points in a ball = 36R/35 = 2.2538e18 m | **114.4 s = 1 min 54 s** | cruise `2.2538e18/2.562048e16 − 2.20 = 85.77 s`, plus 28.65 s of ramps |
| Far rim | 4.3825e18 m | **197.5 s = 3 min 18 s** | cruise 168.86 s + 28.65 s — the traverse promise, kept |

Today's measured warp leg is 66.06 s. The typical warp becomes 114 s and the nearest hop 28 s. **All are
"journeys in minutes", all fall out of the existing speed law, and `TRAVERSE_S` does not change.** The
owner's "keep smaller distances and speeds in the Galaxy realm, just show them as big" is achieved by the
frame's unit, not by a second speed law.

*(Design B proposed changing the speed law on the grounds that one star gap would be a 1.5-second blink.
I re-derived it: you cannot reach the galaxy ceiling inside one gap, because the two ramps alone are
5.64e16 m and the gap is 3.7–6.6e16 m. The change is unnecessary, it would leave the ceiling undefined for
a realm with zero or one children, and it would move the world's radii too — `vd-physics` reads the same
constant for its outset derivation, `scale.rs:84-88`. Deleted.)*

---

## 7. TRAVEL BETWEEN GALAXIES

**MY RECOMMENDATION: there is no declared transition, because none is needed.** This is owner question 5;
the default is (a).

**Why.** The owner's proposal — a relay start, star map blending, a simulated near-light-speed effect, an
artificial wait sized by distance, then appearing at the destination — is a toggle, a loading period and a
teleport, three of the things SL8 names as defects. It would need a written, named exception. It does not
have to, because the same law that makes intra-galaxy warp seamless makes inter-galaxy travel seamless one
level up, with **zero new machinery**.

**What starts it.** Nothing special. You fly outward and cross your galaxy's shell, exactly as you cross a
star system's shell — the same re-home saga, the same generic code that cannot tell a moon from a galaxy
(SL4).

**What the player sees.**
- *Leaving.* Your galaxy's systems fall asleep behind you one at a time as they drop past their own wake
  radii; then the galaxy itself falls below one and a half degrees and becomes what every dormant child
  becomes — a point of light carrying one circumscribed extent and its photometrics, the owner's own
  one-radius law of 17 Aug. It shrinks to a smudge. **This is "the system shrinks to a dot" at the next
  level, done by the existing look horizon, not by new code.**
- *Cruising.* Real parallax of the galaxies, for the same reason as §6 — better than the "no parallax" the
  proposal assumed, and free.
- *Arriving.* The destination galaxy grows from a dot, wakes when it subtends one and a half degrees, and
  draws itself. You cross its shell and its 150,000 stars appear as its interior.

**No stars from another galaxy, and it is already enforced.** A galaxy's stars are its interior, and a
realm's own picture may climb at most `LOOK_CARRIER_ARITY = 2` levels
(`crates/wire/src/session_flow.rs:520-529`), with a boot fence refusing any world whose measured visibility
climb exceeds it (`crates/bins/src/bin/shard.rs:337-341`). From the universe frame, another galaxy's stars
are three levels down. **They cannot reach you.** The owner's requirement is satisfied by an existing law
and an existing fence, with nothing added. *(Whether other galaxies are visible from **inside** your own —
Andromeda is, in reality — is decided by the same budget and is **UNVERIFIED**: I read the constant and the
fence's call site, not the climb measurement.)*

**Duration, derived.** Universe ceiling = 2 × 7.555786e22/180 = 8.395318e20 m/s, which is exactly 2^15 ×
the galaxy ceiling. Ramp = `τ·ln(32768) = 1.10 × 10.3972 = 11.44 s` each, covering 9.2346e20 m.

| Leg | Distance | Time |
|---|---|---|
| Nearest galaxy | 1 Mpc = 3.0857e22 m | cruise 34.55 s + ramps 22.87 s = **57.4 s** |
| Far rim of the universe | 1.5112e23 m | cruise 177.80 s + 22.87 s = **200.7 s = 3 min 21 s** |

**The artificial wait is replaced by a real flight of the same length.**

**What it costs, honestly.**
1. The universe frame must become a running realm with children and a picture. Today it is the ambient root
   and nothing has ever been drawn from it (`crates/core/src/worldgen.rs:23-27` calls Universe and Galaxy
   *"P3 placeholder realm ids … Galaxy get one at P4+"*).
2. There must be a lawful way to fly **out** of a galaxy — that shell has never been crossed outward.
3. The generator must produce galaxies lazily by the same lattice one level up (~61 of them).
4. Relay stations, if the owner still wants them, become flavour: a place to buy the trip, not machinery
   that performs it.

**If the owner prefers the scripted version (question 5b),** then question 6 applies and the exception must
be written down once, and Design B's four-part test becomes the definition of "not a loading screen": the
world keeps running (the tick never pauses, other ships near the relay keep moving); the player keeps
control (attitude, thrust and view live throughout, abortable up to a stated point of no return); the frame
never stops updating and never goes flat or shows a progress bar; and the frame-to-frame pixel difference
stays inside the band of ordinary high-speed flight. **Keep that test either way** — it is a runnable
screenshot gate that *can fail*, and it is the right instrument for policing any future high-speed
sequence. Under (5a) it is applied as a gate on ordinary flight, where it passes, instead of as a licence.

---

## 8. WHAT SATURATES FIRST

In order of when it binds. Every figure is arithmetic over the tree's own constants and type definitions;
none is a measurement.

**1. THE PROCESS BUDGET — binds at about 550 concurrent players. THE FIRST REAL WALL.**
The demand cluster mints shards from a 1,000-port band, two ports per spawn — about 500 shards
(`crates/bins/src/lib.rs:196-203`). A thousand players cluster into roughly 300–400 distinct occupied
systems, each needing one system shard plus about two live interior realms, so ~900–1,200 shards. The band
tops out at ~500, i.e. it binds at roughly **550 concurrent players**. **This has nothing to do with the
galaxy.** It is one operational number in one config and it scales by adding machines. State it as a
capacity plan.

**2. THE INTEREST FOLD — dead on arrival without the inversion.**
`crates/sim/src/stub/aoi.rs:531-560` loops direct children (outer) × observers (inner) with an allocating
`path.clone()` inside the inner loop. At 150,000 children and ~591 observers (see below) × 50 Hz that is
**4.4e9 allocating operations per second** — four orders past dead. **The inversion is provably viable:**
`aoi.rs:365-372` already filters child observers on `child_liveness.contains_key`, and a dormant child
cannot be occupied, so the observer set is bounded by (occupants held + live occupied children + one
interest proxy) while the child set is not. Invert to observers-outer × their own lattice cells.
After: 591 observers × at most 8 cells = 4,728 evaluations per tick = **236,400/s**, sub-millisecond.
*(Observer count: at a 114 s typical leg and one warp per ten minutes, ~19 % of 1,000 players are in
transit = 190 travellers, plus ~400 occupied-child proxies (SL7 — neither design counted these), plus one
interest proxy.)*

**3. THE PER-TICK ROSTER — 21.75 MB in one un-split datagram, dropped in silence.** §5.2. SL9 already
ordered the cure.

**4. THE LOGIN LEVEL — 27.75 MB against a 1,048,576 B cap, refused 26.5× over.** §5.2; must be chunked.

**5. THE RELAY FINGERPRINT — ~1.45 GB/s of pure change-detection encoding.**
`crates/sim/src/stub/window.rs:495-520` postcard-encodes all rows + all bodies + the held interior **every
tick** purely to compute a fingerprint: ~29 MB per tick at this width. Must become an incremental digest.

**6. THE KEEP-ALIVE RE-SERVE — 17.7 MB/s per window.** `window.rs:83-89,124-133` clears the
send-on-change memory every 25 ticks. Ask 3.

**7. MEMORY — 1.6 GB per window, 2.0 GB per session.** §5.2. Ten players on one gateway would be ~20 GB.

**8. CONTAINMENT — 1.2e9 tests per second if the region set is walked.** The 64-region cap is **gone**
(`crates/sim/src/stub/regions.rs:26-33`, SL9, 24 Aug; I confirmed `MAX_REGIONS` and `TooManyRegions` have
zero occurrences in the tree, and membership is already the short list `RegionMembership { realms:
BTreeSet<RealmId> }` at `containment.rs:94-130`). *Design B treated the cap as its headline wall; it was
deleted five days before the design was written, and the measured-ground brief carries the same stale fact.*
What remains is that the region set is a `Vec` walked linearly. It must become the lattice lookup, which
SL9 demands anyway: *"Finding which child holds a point is a LOOKUP, never a scan."* **The lattice makes
this structural:** the jitter keeps every system at least `γ·a/2 = 4.73e15 m` from its cell face, which is
38× its own maximum shell (1.238e14 m), so a system **never crosses its own cell's face** — the child
holding a point, if any, is the one in that point's own cell. Three integer divisions per axis, then one
distance test.

**9. THE BOOT FENCES — 1.1e10 pair comparisons per boot, ×284 across the seed sweep.**
`seeded_systems_disjoint_3d` is an explicit all-pairs loop (`guards.rs:262-277`) and
`guard_swept_seeds_nest` runs a full world per seed over a 284-seed sweep. Replace the global pairwise
fence with a **construction proof** (the jitter bound implies the 1-ly hard core, proved once), plus a
randomised pair sampler in the gate, plus the exhaustive fence over the ≤27 cells around any point the game
actually touches. **Stated as a risk:** the replacement is a weaker negative control than an exhaustive
loop, and this tree's own history warns about disarming a fence whose refusal is the world's negative
control. It must be replaced, never deleted.

**10. THE CLIENT — 150,000 ECS entities with per-frame transforms.** §5.2. Also the delta path clones the
whole box map and recomputes every box's nesting depth on every delta
(`crates/client/src/realm_scene.rs:207-233`). Both are SL8 tick-hitch defects at this width.

**11. THE GALAXY SHARD'S OCCUPANT LANE — genuinely fine, and worth saying.** Bodies are membership-gated,
and the visibility rule is one generic angular test: a 100 m ship is above one and a half degrees only
inside `76.39017 × 50 = 3,820 m`. Against a lattice pitch of 6.6477e16 m, two travellers in the galaxy
frame are mutually invisible with a margin of about **1.7e13**. **So a thousand simultaneous travellers
produce zero cross-traveller rows: the body lane is O(players), not O(players²).** Fleet-mates in convoy
are the only exception, and they are exactly the one-location density fixture already proven at N ≥ 128.

**12. THE RECONNECT BURST — the one genuine star-count × player-count term, and neither design budgeted it.**
A thousand players reconnecting after a restart costs 1,000 × 27.75 MB = **27.75 GB** of catalogue egress,
arriving exactly when the cluster is least healthy and competing with the live lane. Three cures, all
required: encode **once per gateway** (the content is observer-independent, proven above); page it
**nearest-first**; and **cache client-side by generation** so a returning client sends nothing. With the
cache, a cold start is a one-time cost per client per world change, not per session. *This is the
measurement that would reopen SL6 ask 4.*

**WHEN THE GALAXY SHARD ITSELF SATURATES.** Its tick is O(travellers) with no term in 150,000. I estimate
saturation in the low tens of thousands of simultaneous in-transit ships; **UNMEASURED** — no gate in this
tree runs a realm with thousands of occupants, and I was forbidden to run anything. When it binds, HR1 and
HR3 forbid splitting the realm (a realm's world is private and cannot be replicated), so the lawful escape
is **more galaxies** — the universe frame holds about 61, each an independent shard.

**WHICH COSTS GROW WITH WHAT, after the rewrites:** the per-tick cost grows only with **players**. The star
count survives in exactly three places, all one-shot or offline: the catalogue bytes at join, the GPU vertex
buffer, and the generator's boot fences.

---

## 9. THE LAW TABLE

| Law | Verdict | Structural or careful, and why |
|---|---|---|
| **HR1 sealed shards** | HELD | **Structural.** No shard reads another's world. The catalogue is the galaxy's own authored placements for its own direct children — exactly what a parent already holds. A running system's interior still rides the existing sealed relay unopened. |
| **HR2 generic transfer** | HELD | **Structural.** A star system, a moon and a rock cross by identical code. The unit is a property of the frame pair, not the entity kind; the registry is untouched. |
| **HR3 one tooling** | HELD | **Structural, and this is why the spine is what it is.** The unit rides `FrameRef::tier()`, which `pose.rs:120-133` already declares "a pure geometric property of the frame, NOT a feature fork on realm/shard KIND — it selects a coordinate UNIT, nothing else". One mechanism, already blessed, already on the wire. *(Design A's alternative would have been a second unit mechanism standing beside this one; that is the HR3 breach the ranking turned on.)* |
| **HR4 features once, run anywhere** | HELD-WITH-CARE | The tier is unchanged for every realm a fixture runs on today, so existing fixtures are byte-identical, and the galaxy exercises the new arm. **The care:** the catalogue lane's ≥2-shard-kind fixture needs a second kind. The natural one is a planet with many player-built areas, and **no such fixture exists today**. Owed work, named. |
| **HR5 100 % coverage** | HELD-WITH-CARE | The conversion must be a branchless shim (a straight-line multiply/divide by a unit read from the frame) with all branching in monomorphic helpers — the `crates/core/src/tlv.rs` shape. The lattice lookup is straight-line integer arithmetic. Nothing here is generic, so the per-monomorphisation trap does not apply. |
| **HR6 agent-operable end-to-end** | HELD | Six new gates in §10, every one able to fail. |
| **SL1 only the parent knows positions** | HELD | **Structural.** The unit conversion lives on the frame edge, **in the parent** — precisely where SL1 puts every conversion. A realm knows only its own size in its own units, which it already does. Nothing is folded from the root. |
| **SL2 no occupant pose crosses a boundary** | HELD | **Structural.** Nothing new crosses. Travel is still out into the shared parent and in again. |
| **SL3 a realm draws itself** | HELD | **Structural.** The marker keeps exactly its two lawful tags. The per-child message **shrinks** — from 145 B twenty times a second to 145 B once — which is SL3's own "must shrink toward a placement, never grow". A running realm's own picture still supersedes the point of light by data presence, not by a flag. |
| **SL4 physics and re-home separate, one-way** | HELD-WITH-CARE | The unit conversion reads a frame pair and a length; it cannot ask how anything moves. The lattice lookup reads a position and returns a realm. **The care:** send-on-change must be a **value comparison**, never a "does this child have motion?" test — that would be a specific on a lookup path, which SL4 forbids by name. Enforce with a crate dependency rule, not care. |
| **SL5 one world** | HELD | **Structural.** One universe from the seed. The step (1 m), the count (150,000 inside a 174,791 ceiling), the density (the measured census), the hard core (one light year) and the mass cap (re-solved) are all solved. No knob, no preset, no second generator, no test-only variant. *(Slice 0's independence gate measures a **function** at N children, not a second world.)* |
| **SL6 ask before new data crosses** | HELD | Five asks raised in §2. **Not one puts a new datum across a realm boundary.** Three move strictly less data; one changes the shape of an arm nothing currently produces; one is refused. |
| **SL7 an occupied realm is its own occupants' proxy** | HELD | **Structural.** The parent still decides interest for its direct children and still treats an occupied child at the placement it authored (`aoi.rs:365-372`). It simply finds them by lookup instead of by scan. One bit still crosses upward. *This design counts the occupied-child proxies in the cost model (§8) — neither source design did.* |
| **SL8 seamless is a main rule** | HELD | **Structural for the crossing** (no length is scaled, so there is no scale change available to jump; the swap is decided by data presence; the crossing line sits where nothing is left to change). **Careful for the render path** until §4.8 lands and the instanced point cloud replaces 150,000 entities — both are named SL8 defects today at this width. **No exception is requested** under the default answer to question 5. |
| **SL9 a parent's child count is unbounded** | HELD | Requires every rewrite in §8, all of which SL9 already orders in its own words (`CLAUDE.md:96-107`: *"never a per-tick walk of all of them, never a per-child row on a per-tick lane … Finding which child holds a point is a LOOKUP, never a scan … it must be measured on a realm with many, not argued"*). Structural after the rewrites: no fixed width (already retired), no per-tick walk (the fold inverts), no per-child row on a per-tick lane (send-on-change), and containment is one integer division per axis. **The independence must be MEASURED, not argued** — Slice 0's gate. |
| **Server does all the maths; client only renders** | HELD-WITH-CARE | The client subtracts one server-supplied origin from a server-supplied static list — the same subtraction it already performs per row. It composes nothing. **The care:** the catalogue must never be client-generated from the seed, however tempting; and §4.8's fix must move the subtraction into the lattice **server-supplied**, not invent a client-side rebase. |
| **No client-side prediction** | HELD | Untouched. The catalogue is static; live rows still ride the interpolation buffer. |
| **Players physically collide** | HELD | Untouched. Travellers are ordinary occupants with ordinary bodies; convoys collide under the density fixture already proven at N ≥ 128. |
| **No magic numbers** | HELD-WITH-CARE | Every new number is derived: the galaxy step from the requested census and the storage fence; the universe step from the galaxy count; the lattice pitch from the measured stellar density; the hard core from one light year; the mass cap re-solved. **The care:** the mass cap must be re-solved by the code's own bisection, not by my two-point fit. |
| **D-SCALE-1 (owner ruling, 5 Aug 2026)** | **NEEDS OWNER RULING** | Its *model* clause is satisfied more than exactly — compression becomes 1.000000 everywhere, so a star system keeps true size and true internal distances and the gaps become true too. Its *rationale* clause ("a uniform compression makes the boundary a zoom") is moot, because nothing is compressed. Its *"galaxy is a lattice of cell-realms"* ruling is superseded by the owner's own 24 Aug proposal. **Owner question 7.** |
| **Owner law 3(b), 15 Aug: only full state rides a drop lane** | HELD-WITH-CARE | Preserved by putting unchanged placements on the reliable re-driven lane and keeping full state on the drop lane for what changed. **This is the subtlest point in the design and it needs a careful reviewer**, which is why it is SL6 ask 2 rather than a silent edit. |

---

## 10. THE SLICE PLAN

Each slice names a gate that **could fail**. Slice 0 needs no owner ruling.

| # | Slice | Delivers | The gate (can fail) | Depends on |
|---|---|---|---|---|
| **0** | **Make a parent's per-tick cost independent of its child count** — invert the interest fold to observers-outer; make the containment region set a lookup instead of a linear walk; replace the relay fingerprint's whole-state encode with an incremental digest; replace the keep-alive's baseline clear with a generation compare; partition the shard→gateway level message. | The only thing standing between this tree and SL9 compliance. Valuable even if every other slice is cancelled. | **The child-count independence gate.** Measure the fold's wall time, allocation count and emitted bytes **as a function**, at 3 children and at 150,000, with one traveller; assert no growth beyond noise. SL9 says this must be measured, not argued. Fails today by four orders. | **Nothing. No owner ruling.** |
| **1** | **The coordinate ladder.** Re-value the galaxy step to 1 m; append a universe step of 2^15 m; give the galaxy frame a seed and a realm; teach `guard_root_representable` to read the root frame's own step; move the light-year constant to its own name; **move the client's eye subtraction into the lattice before flattening (§4.8)**. | Positions that fit a real galaxy, at exact conversions. | **The exact-crossing gate.** Round-trip a position across every tier boundary and assert: downward conversion **bit-identical**; upward conversion within 2^-63 m; the storage fence passes at **exact equality** at every tier; every existing in-system golden **byte-identical**. Plus a near-field gate: two objects 100 m apart at the galaxy rim draw 100 m apart, not 512 m. | Ask 1 (question 1) |
| **2** | **Volumetric placement and the census.** Replace the sphere shell with the jittered lattice; make the generator lazy and per-cell; replace the quadratic boot fences with a construction proof + sampler + exhaustive-over-27-cells; re-solve the mass cap with the code's own bisection; raise the census to 150,000. | A real galaxy with real spacing. | **The separation gate.** Assert the minimum separation over a large random sample is ≥ one light year and that the construction proof's bound is tight; assert the mass cap the bisection returns; assert boot time at 150,000 systems is bounded. Plus: **look at the sky** and sample the mean nearest neighbour (currently UNMEASURED). | 1; questions 3, 7, 9 |
| **3** | **The sky.** Placements send-on-change across the two cadences; chunk the login level; encode once per gateway; cache client-side by generation; retain the live set only in the rings; replace 150,000 entities with one instanced point cloud. | 150,000 points of light for 0 bytes a tick. | **The sky byte gate.** Assert: per-tick sky bytes = 0 after the first tick; login level chunked under the frame cap; the catalogue is **not re-sent on an epoch bump** (a warp leg is two crossings); a thousand-session gateway holds one catalogue, not a thousand; client frame time flat at 150,000 points. | 0, 1, 2; ask 2, ask 3 |
| **4** | **The warp.** The full leg, end to end. | The owner's picture, playable. | **The full-warp pixel gate.** Fly departure → ramp → cruise → approach → arrival and assert: no black frame; no brightness step at the handover; drawn centre, drawn radius and drawn colour differ by **exactly zero** across the handover frame; the live realm count never exceeds a handful; frame-to-frame pixel difference across the crossing tick is inside the band of an ordinary tick at the same speed. | 3 |
| **5** | **Capacity.** Widen the shard port band; measure the galaxy shard's tick against traveller count; measure the reconnect burst at a thousand simultaneous logins. | An honest capacity plan instead of an estimate. | **The concurrency gate.** Assert the shard count and reconnect egress at 1,000 players against a stated budget. This is the gate that turns §8's "UNMEASURED" into numbers. | 4 |
| **6** | **The universe level.** Make the universe a running realm with children; generate galaxies lazily one level up; make the galaxy shell crossable outward. | Inter-galaxy travel. | **The inter-galaxy gate.** Fly out of a galaxy and into another; assert the visibility-climb fence still refuses a world where another galaxy's stars could reach you; assert the same zero-delta handover at the galaxy shell; assert the 57 s leg. *(Or, under question 5b, the four-part not-a-loading-screen test.)* | 4; questions 5, 6, 8; ask 5 |

---

## 11. WHAT THIS DESIGN DOES NOT DO

- **It does not compress anything, anywhere.** Not between systems, not inside them. There is no scale
  knob and no factor to tune. The field that would hold an in-system factor stays deleted.
- **It does not change a single stored number inside a star system.** Every frame at or below a star system
  keeps its millimetre step, and every in-system golden must come out byte-identical (Slice 1's gate).
- **It does not change the speed law.** `TRAVERSE_S` stays 180 s, `realm_speed_cap_mps` stays one
  expression with no realm-kind test, the ramp and the governor are untouched. The journey times move
  because the world got bigger, not because the law did.
- **It does not measure anything.** Every number here is read from a named file and line or is arithmetic
  with the working shown. I was forbidden to run cargo, just, any build, test, benchmark or binary, and I
  did not.
- **It does not solve the mass cap.** It states the inequality and gives a fitted answer of about 6.1 solar
  masses, labelled UNMEASURED. The code's own bisection is the authority and must be run before anything is
  built.
- **It does not know how regular the star field will look.** The jittered lattice's mean nearest neighbour
  is UNMEASURED, between 0.554 and 1.0 of the pitch.
- **It does not make the galaxy rotate.** `Separation::rotated` refuses a non-identity rotation beyond
  `cell_edge / f64::EPSILON`; at a 1 m step that reach is 4.5036e15 m against a 6.6477e16 m star gap, so a
  rotating galaxy frame is **refused at any tier** — this is unchanged by the design and identical under
  both source designs, because they describe the same world. Proper motion, if wanted later, belongs in the
  catalogue as a per-star velocity (a star at 30 km/s crosses one drawn-position step in 2.4 hours), never
  as a rotating frame.
- **It does not filter any star out of anyone's sky.** All 150,000 are drawn, always. Only the lane changes.
- **It does not build the inter-galaxy level.** It shows that the same law one level up gives ~61 galaxies,
  a 57-second flight to the nearest, real parallax, and no foreign stars — and that an existing fence
  already enforces the last of those. Making the universe a running realm is separate, unbuilt work.
- **It does not claim the rewrites are small.** They touch the interest fold, the window emit, the
  keep-alive, the relay fingerprint, the region set, the storage fence, the generator, the client draw path
  and the renderer. It claims only that **SL9 already requires every one of them**, whatever the galaxy ends
  up looking like — which is why Slice 0 needs no ruling.
- **It does not add client-side prediction, client-side composition, or client-side world generation.**

---

## 12. THE JUDGE HOLES REGISTER

Every hole the judges raised, with its resolution. "Accepted" means the design carries the risk knowingly.

| # | Hole | Raised by | Resolution |
|---|---|---|---|
| 1 | **Design B's speed-law change solves a problem that does not exist**, and its galaxy-crossing figure is 2× wrong (it divided the radius, not the diameter). | Judges 1 & 2 | **RESOLVED — deleted.** §6 re-derives the journey times from the shipped law: 28 s / 114 s / 198 s. The two ramps alone exceed one star gap, so the ceiling is never reached on a short hop. `TRAVERSE_S` and `realm_speed_cap_mps` are untouched. |
| 2 | **The 64-region boot cap no longer exists** — it was deleted five days before Design B cited it as its first wall. | Judge 1 (and Judge 2) | **RESOLVED.** Confirmed by grep: zero occurrences of `MAX_REGIONS` / `TooManyRegions`; `regions.rs:26-33` records the deletion naming 150,000 systems by hand; membership is already the short list. §8 item 8 states the *live* constraint (the region `Vec` is walked linearly) with the same cure. |
| 3 | **`guard_root_representable` is hard-coded to the millimetre step and will refuse the new world by 1,024×.** | Judge 1 | **RESOLVED.** Confirmed at `guards.rs:69`. Made the **first line** of Slice 1's work, not a discovery. Its own error text already names this design as the cure. |
| 4 | **Design A's catalogue row drops the extent tag, contradicting its own no-pop proof.** | Judge 1 | **RESOLVED — the extent stays.** §5.3. The guarantee it invoked is real and structural (`generate.rs:420-424`); replacing a bit-equal radius with a photometric estimate at the instant of the swap is exactly the seam SL8 names. |
| 5 | **The new catalogue wire arm should be refused under SL6** — the local formulation exists and both designs already proposed it elsewhere. | Judge 1 | **RESOLVED — refused.** Owner question 4, ask 4, default (a). The measurement that reopens it is named in §8 item 12. |
| 6 | **Design A's scale exponent is a second coordinate-unit mechanism standing beside `FrameRef::tier()` (HR3), and its unit is invisible in the source.** | Judge 1 (Judge 2's strongest graft) | **RESOLVED — the spine.** The design uses the existing per-frame tier. A metre never stops meaning a metre. And it is **free today**: `GalaxySpace.realm()` returns `None`, `frame_for_realm` has no arm producing it, no coarse-tier pose exists — verified. |
| 7 | **Design A's zero-step proof is carried out in floating-point metres; the store is integer cells and the integer path is never shown.** | Judge 1 | **RESOLVED.** §3 and §6 state the conversion on the whole-number part: downward exact, upward rounding at 2^-63 m. Slice 1's gate asserts on the **cell rebase**, not only on the drawn vector. |
| 8 | **Design A's headline zero-step gate would fail on the shipped render path**, which flattens from the realm origin and only then subtracts the eye. | Judge 2 | **RESOLVED, and promoted.** §4.8 — verified at six call sites. The cure (subtract in the lattice, then flatten) is in Slice 1 and repairs a latent defect that exists today at 0.5 m. The gate is stated as "exact after the fix", with the physical error before the fix quantified (256 m = 3.6e-12 pixel for stars, fatal for near-field). |
| 9 | **Design B's coordinate answer leaves no room for a second galaxy**, contradicting its own inter-galaxy section. | Judge 2 | **RESOLVED.** §4.4 adds the universe tier at 2^15 m, giving 7.99 Mly and ~61 galaxies inside the identical unchanged fence. |
| 10 | **Design B's mass cap (~12.8 M☉) divides the *mean* separation by a fence that must hold for every pair.** | Judge 2 | **RESOLVED.** §4.6 uses the worst-case hard core and I confirmed against `body.rs:178-181` that the wake band is built from the **shell**. Answer ≈ 6.1 M☉, labelled UNMEASURED, with the code's own bisection named as the authority. |
| 11 | **Design B specifies no catalogue wire format and no shader precision discipline** — a naive f32 star buffer would make the sky swim. | Judge 2 | **RESOLVED.** §4.8: the subtraction happens in integer cells before any narrowing. The quantised-integer discipline is the design's, whatever lane the catalogue rides. |
| 12 | **Design A's catalogue row carries no identity, so the dot cannot be matched to the realm that supersedes it.** | Judge 2 | **RESOLVED by refusing the compact arm.** The existing rows carry identity by construction. If ask 4 is ever granted, identity is mandatory. |
| 13 | **Design A's warm lead is miscomputed** (0.4 s of cruise is really 0.015 s; the ramp cannot be added whole). | Judge 2 | **RESOLVED, and strengthened.** §6 states the governor's closed form and shows the extent **cancels**: the lead is **9.61 s at every scale**, not an instance. Design B's 9.60 s was the correct arithmetic; the scale-free form is better than either. |
| 14 | **Design A uses 0.75R as the typical journey; the mean distance between two uniform points in a ball is 36R/35.** | Judge 2 | **RESOLVED.** §6 uses 36R/35, giving 114.4 s. |
| 15 | **Design A mixes the Poisson mean nearest neighbour with a jittered-lattice placement.** | Judge 2 | **ACCEPTED and bounded.** §4.6 and §6 quote both ends (0.554a to a) and mark the true value UNMEASURED. The error is in the safe direction on the number that matters (the dot margin rises from 95× to 171×); the only cost is how often you pass a star during cruise (2.6 s vs 1.4 s). |
| 16 | **Neither design converts the ~500-shard port band into a player number**, which is the actual answer to Q-B. | Judge 2 | **RESOLVED.** §8 item 1: it binds at roughly 550 concurrent players, and it is the **first** thing that breaks. Promoted to the lead of the saturation section. |
| 17 | **Neither design states that the galaxy's body lane has no player-squared term** — the most reassuring number available. | Judge 2 | **RESOLVED.** §8 item 11. |
| 18 | **Neither design budgets the reconnect burst** — the one genuine star-count × player-count term. | Judge 2 | **RESOLVED.** §8 item 12, with three required cures and Slice 5's gate. |
| 19 | **Design A's observer set omits the SL7 occupied-child proxies.** | Judge 2 | **RESOLVED.** §8 item 2 counts ~400 of them (591 observers, not 190). |
| 20 | **Design A's `s=10` yields 174,098 systems, 16 % above the owner's stated ceiling, presented as landing on the request.** | Judge 2 | **RESOLVED.** §4.3 states the ceiling (174,791) and the populated count (150,000) separately; the 5.23 % of empty rim is the clearance, and the census is the owner's number, not the fence's. |
| 21 | **Neither design solves galactic rotation, and only one names it.** | Judges 1 & 2 | **ACCEPTED.** §11: refused at any tier, identical under both readings, and proper motion belongs in the catalogue as a per-star velocity. No design here requires rotation. |
| 22 | **The quadratic boot fences become unrunnable and must be replaced, not deleted** — the replacement is a weaker negative control. | Judge 1 | **ACCEPTED with the risk stated.** §8 item 9. This tree's history warns about disarming a fence whose refusal is the world's negative control. |
| 23 | **Both designs treat "markers always ship" as a law they must ask to change, when SL9 already overrode it.** | Judge 1 | **RESOLVED.** Ask 2's row says so plainly: this needs scheduling, not permission — `CLAUDE.md:96-107` post-dates the wire contract sentence and wins. It is still raised as an ask because it edits a shipped contract sentence. |
| 24 | **Design B's fourth no-pop leg cites a constant the tree itself says is "documentation with a number attached", not a fact about the world.** | Judge 1 | **RESOLVED.** Confirmed at `scale.rs:308-318`. §5.3 rests on the three structural legs and states the fourth (the crossing line sits where nothing is left to change) as a **fence to be built**, not a measurement already taken. |
| 25 | **Design B asks for an SL8 amendment that Design A shows is unnecessary.** | Judge 1 | **RESOLVED.** §7 defaults to physical flight; the amendment is owner questions 5–6 and is taken only on an explicit yes. Design B's four-part test is kept regardless, as a gate rather than a licence. |
| 26 | **Design A's rotation-refusal margin is computed in the wrong units.** | Judge 1 | **RESOLVED.** §11 states it in true metres: reach 4.5036e15 m against a 6.6477e16 m gap — 14.8× short, identical under both readings, because they are the same world. |
| 27 | **Neither design measured anything; the per-row byte figures differ.** | Judges 1 & 2 | **ACCEPTED and labelled.** §5.2 states ±30 % and names the two structural facts that survive it (the 1 MiB refusal; the un-split drop-lane datagram). Both verified in code. |
| 28 | **Only two of three designs were delivered to Judge 1; only one and a half to Judge 2.** | Both judges | **NOTED as a process defect.** A third spine — an anti-strobe cruise ceiling derived from pixels-per-frame — was never judged. It carries an unjudged hazard: a ceiling that reads frame rate and field of view lets a **rendering** parameter into the world's physics, which collides with "the server does all the maths" unless both are fixed constants of the one world. **This design does not use it**, and does not need it (§6 shows the shipped law already gives journeys in minutes). If the owner wants it considered, it must be re-sent and judged. |

---

*End of design of record. Six owner questions block nothing until Slice 1; Slice 0 can start today.*
