# Owner decisions — 2026-08-27 — the galaxy's shape, its density, and how galaxies differ

★ **LATER THAN EVERY DESIGN DOC, AND LATER THAN THE MOVEMENT ANSWERS OF THE SAME DAY.** Where a plan
disagrees with this, this wins. Written BEFORE S12 starts, on the owner's instruction, so the placement
law is built against it rather than retrofitted to it.

This file completes the shape ruling of 2026-08-24 (*"Shape please. Also never use the same distances
between star systems. Of course placement should be part of the seed and do not change."*). That ruling
said WHAT the placement must be. This one says where the numbers behind it come from, and what may
differ between one galaxy and the next.

---

## G1. EVERYTHING THE PLACEMENT READS IS SEED-DERIVED

**Owner, asked whether density should be an input or an outcome:** *"Density also should come from seed,
otherwise any tiny change might change positions."*

That reasoning generalises, and it is the rule:

> **Any number the placement reads must be drawn from the seed. A value an operator can set is a value
> that re-rolls the galaxy.**

A density knob is not a tuning parameter. It is a lever that moves every star that any player has ever
seen. The same is true of the count, the radius, and the shape.

**What this forbids, concretely:**

- no density in a config file, an environment variable, or a `DevClusterParams` field;
- no "census" as a target number that a person picks;
- no scale preset, and no test-only variant (SL5 already says this; G1 is the same law reaching the
  placement).

**What it requires:** a galaxy's seed draws its own shape, its own density and its own size. Those three
are facts ABOUT that galaxy, fixed for as long as the world exists.

---

## G2. THE SHAPE IS BELIEVABLE, AND IT COMES FROM THE SEED

**Owner:** *"Believable, and all should come from the seed, and shape and position of the stars should
not change."*

The taxonomy already draws a galaxy's KIND from its seed — `GalaxyType::{Spiral, Elliptical, Irregular}`
(`crates/physics/src/taxonomy.rs:40-44`). The placement law ignores that draw today. It must consume it.

**Believable means the shape reads as the thing it is named after.** A spiral has a bulge, a disc with
real thickness, and arms. An elliptical is a smooth three-dimensional swell, denser toward the middle. An
irregular is neither, and looks it.

**What is refused, and each for a stated reason:**

| Refused | Why |
|---|---|
| A shell (today's law) | zero thickness; every star at ONE radius; nothing between or beyond |
| A ring | same failure on one axis |
| A plain filled ball | the owner declined it explicitly on 2026-08-24 — it is not a shape |
| A lattice AS A POSITION SOURCE | a grid produces repeated distances, which ruling 2 forbids |

★ **A LATTICE REMAINS LAWFUL AS A LOOKUP.** Finding what is near a point may use any structure it likes.
DECIDING where a star sits may not use a grid. The two are different jobs and only one is constrained.

---

## G3. NO TWO SYSTEMS AT THE SAME DISTANCE — INCLUDING THE RADIAL AXIS

Restated from 2026-08-24 because S12 is where it becomes real: no constant radius, no ring, no shell, no
lattice-quantised position. The radial axis is the one today's law violates absolutely.

---

## G4. PLACEMENT IS STABLE FOREVER

Where the seed puts a star, the star stays. Growing the world must not move what is already placed.

The gate S12 already carries says the same thing: *grow the census from N to N+1 and every existing
system is bit-identical, in identity AND position.* The ruling and the gate are one requirement stated
twice.

**This is what makes the world's size a reversible decision rather than a launch-freezing one.** Today
the radius is derived from the count, so changing either moves every star — including every player's
home.

---

## G5. HOW GALAXIES DIFFER — AND THE LINE THE SEED RULING DRAWS

The 2026-08-24 ruling requires galaxies to DIFFER, *"in density, star population, what is found there"*,
because otherwise galaxy 61 is 150,000 systems statistically identical to galaxy 1 and the whole feature
is a corridor to more of the same.

### ★ A CORRECTION, OWNER-MADE 2026-08-28

An earlier draft of this section argued that a seed-derived RICH galaxy would be a treasure map, and that
the third item therefore collided with the seed ruling. **The owner corrected it:**

> *"we will not derive the planets contents. So even if this is a map, players don't know what they will
> find there."*

That is right, and the concern was overstated. The block ruling already forbids a block's substance from
being a function of (position, seed), so **what is IN the ground is not seed-derived at all.** A galaxy's
seed can therefore say a great deal without saying anything about what a place pays.

**THE REAL LINE, restated:**

> A seed-derived fact may tell a player **WHERE TO LOOK**. It may never tell them **WHAT THEY WILL FIND**.

A wiki entry reading *"galaxy 61 is dense, young and metal-rich"* is a HYPOTHESIS, not an answer. Going to
check it IS the survey — the mechanic, performed. That is a better game than one where nothing about a
place can be known before reaching it, and it passes the seed ruling's own test unchanged: printed on a
wiki tomorrow, the game still works.

**So the difference splits — but the line falls further out than the earlier draft placed it.**

### MAY differ, drawn from the seed, safe on a public wiki

| Differs | What it changes for a player |
|---|---|
| **Shape** | how you navigate — arms to follow, or a formless cloud |
| **Density** | travel time between stars, and how crowded the sky looks |
| **Size** | how long a crossing takes, and how many systems exist |
| **Stellar population** | the COLOUR of the sky. A young galaxy carries bright massive stars; an old one is mostly dim red dwarfs |

Each of these changes what you SEE and how you TRAVEL. None of them is worth money.

A galaxy's seed may also shape what KINDS of place exist there — how many rocky worlds against gas
giants, how hostile the neighbourhood is. That is a reason to go, and it is publishable, because knowing
a galaxy holds many rocky worlds tells you nothing about what any one of them contains.

### MAY NOT be seed-derived

**What a place CONTAINS.** Not its ore, not its deposits, not what is worth taking. That reads world state
that moves — depletion, what NPC life consumed, what players built and took. This is the block ruling,
unchanged, and it is what makes everything above safe to publish.

The distinction in one line:

```text
   seed may say:  "this galaxy has many rocky worlds"      -> where to look
   seed may NOT:  "this world's crust holds what you want" -> what you find
```

---

## G6. WHY ANYONE PAYS FOR AN EXPENSIVE GATE — ROOM, NOT TREASURE

The owner's standing observation is *"keep gates expensive"*, because gates are the only thing keeping
the world's size matched to its population.

If a second galaxy may not be richer, what is it for?

**TWO reasons, and both are honest.**

**It is empty.** Nobody has taken it: no depletion, no built structures, no claims. That is LIVE world
state, not seed state, so it satisfies the seed ruling exactly — and it decays honestly. Once people live
there, the reason to go is gone, and the next galaxy becomes the frontier.

**And it is DIFFERENT, knowably.** Its shape, density and population are publishable, so a player can have
a REASON to choose one galaxy over another — a young dense spiral rather than an old sparse elliptical —
without knowing what waits in any particular place. A reason to go is not the same as a guarantee of
reward, and the difference between those two is exactly where the survey lives.

This is the same shape as the dormant-world pillar and the player-driven economy: what is valuable is
what the world has DONE, never what the seed says.

---

## G7. WHAT S12 BUILDS, AND WHAT IT DOES NOT

**IN S12 — one galaxy, done right:**

- shape drawn from the seed, and believable;
- density drawn from the seed;
- positions stable under growth;
- no repeated distances on any axis.

**NOT IN S12 — the differences BETWEEN galaxies.** They matter only when more than one is reachable, and
a second galaxy needs a player-built gate, which is far later (the 2026-08-24 ruling: *"a second galaxy
needs a player-built gate, so an unreached galaxy is not generated, not simulated, and costs nothing"*).

**Designed now, built then.** G5 exists so the mechanism is not invented under pressure later, and so the
seed ruling's line is drawn before anything valuable is written.

---

---

## G8. THE COUNT IS A RESULT, AND EVERY GALAXY HAS ITS OWN (owner, 2026-08-28)

**Owner:** *"Count is a result - I agree with that. 150K is the target number, not exact amount all
galaxies should have. The amount also should come from the seed."*

So `150,000` is a SCALE, not a census. It says roughly how big a galaxy of this world is. It is not a
number any galaxy must hit, and it is not a number a person sets.

Each galaxy draws its own size and its own density. Its population FOLLOWS:

```text
   seed -> shape, size, density
        -> the count is whatever that volume at that density holds
```

**THE GATE CHANGES WITH IT.** *"The world holds 150,000 systems"* is no longer the property to assert —
it names a number nobody chose. What must be asserted is that the population MATCHES the density the
seed drew, and that the mean nearest neighbour is what that density implies. The believability is the
measurement; the count is an outcome of it.

⚠ **`WORLD_SYSTEM_COUNT: u32 = 3` (`generate.rs:51`) is a stated count and must go.** So must any later
constant that names a population.

---

## G9. THE SHAPE IS AS FAITHFUL AS WE CAN MAKE IT (owner, 2026-08-28)

**Owner, asked how faithful the spiral should be:** *"Ideally very faihful!"*

A bulge, a disc with real thickness, arms with a pitch angle, and a density that falls with radius. Not
a gesture at a spiral — the thing itself, to the extent the seed and the arithmetic allow.

---

## G10. THE HOME IS FOUND, NOT PLACED — AND THE START BECOMES DYNAMIC (owner, 2026-08-28)

**Owner:** *"In the new galaxy we need to find a new home, with the earth-like planet. In the future all
players will start in one city, which will be the economy and communication hub, meaning we will need to
build cities and stations by hand on the planets of that system. So the starting position in the future
should be dynamic and should be decided by the algorithm (similar to star citizen, players can start
inside their apartments, and we will need to make sure that they do not overlap, but now as we want to
test the ship realm mechanics and warp, we can start in that home system)."*

**THE END STATE:**

1. The home system is **found** — searched for by the property that matters, an Earth-like planet. It is
   not a position the placement law hands out, and it is certainly not "the centre".
2. **Every player starts in ONE city**, which is the economy and communication hub. That city is BUILT,
   by hand, on a planet of that system.
3. The starting POSITION is then **dynamic, decided by an algorithm** — players begin inside their own
   apartments, and no two may overlap.

**FOR NOW, DELIBERATELY:** keep starting in the existing home system. The immediate goal is to fly a
ship realm and test warp, and moving the start would change the thing under test.

⚠ **THIS RETIRES "THE HOME SITS AT THE CENTRE".** Today the home is anchored at the galaxy's middle. In
a faithful spiral that is the BULGE — the densest, brightest, most crowded region, and the hardest place
to make believable. The home must be found by its own property, wherever the seed put it.

---

## G11. RAISE THE COUNT GRADUALLY — BUT NEVER WITH A SECOND WORLD (owner, 2026-08-28)

**Owner:** *"It's ok to raise the count gradually, but we should not build separate arms or configs, or
worlds for that - use the same seed and generation mechanisms and see how world reacts, so in the end we
have the ~150K of stars in the production ready world."*

This is SL5 reaching the placement, and it refuses the obvious shortcut. There may be:

- no small-world preset,
- no test-only census,
- no config knob for the star count,
- no second generator and no second code path.

**HOW TO LOOK AT A SMALL GALAXY WITHOUT BUILDING ONE.** G8 already provides it: galaxies DIFFER in size
and density, both drawn from their seeds, so some galaxies simply ARE small. Looking at a small galaxy is
looking at THE world through a different seed — the same generator, the same law, no variant anywhere.

**The end state is one production world holding roughly 150,000 systems**, reached by watching the real
mechanism react, never by a scale that gets removed later.

---

## G12. WHEN THE SHAPE AND THE GAP DISAGREE, PUSH — NEVER DROP (owner, 2026-08-28)

A faithful bulge asks for density. The per-pair gap sets a minimum separation. Where they meet, one must
yield.

**RULED: push the star out. Do not drop it.**

```text
   DROP  ->  thins the bulge exactly where the shape is trying to be densest
             a galaxy with a suspiciously hollow middle — the shape defeated by its own rule
   PUSH  ->  the star survives, moved, deterministically
```

**The push must itself be seed-derived**, or G4 breaks: a position that depends on the ORDER stars were
considered in is a position that moves when the count changes. Same input, same answer, every time.

### ★ THE HEADROOM, MEASURED 2026-08-28 — this is not a common case

| | |
|---|---|
| galaxy radius | 487.5 ly (`root_radius_at(Tier::Galaxy)`) |
| 150,000 systems spread evenly | **8.19 ly** mean nearest neighbour |
| the gap floor | 1.00 ly |
| headroom | **8.2x** |

The floor binds only where the bulge exceeds **~550x** the mean density. A realistic bulge is around
**60x**, which lands at 2.09 ly — comfortably clear.

⚠ **BUT THE CONTRAST IS ITSELF SEED-DERIVED (G1), so a galaxy MAY draw one above 550x.** The push rule
cannot assume a safe value; it must hold whatever the seed picks.

⚠ **AND THIS IS THE UNIFORM FIGURE.** Arms concentrate stars further, so spacing inside an arm is
tighter than 8.19 ly even outside the bulge. That is a SECOND measurement and it needs the arm model,
which does not exist yet.

---

## G13. THE ARM LAW — PROPOSED IN CODE, JUDGED BY LOOKING (owner-agreed, 2026-08-28)

*"Very faithful"* is a direction, not a formula. The assistant proposes the standard model and the owner
judges the picture, because a galaxy is easier to judge than to specify:

- a **logarithmic spiral** — arms winding outward at a fixed pitch angle;
- stars **scattered around** an arm, never sitting on it (a line is as unbelievable as a shell);
- a **bulge** whose density falls with radius;
- a **disc with real thickness**.

Every number of it drawn from the seed (G1): arm count, pitch angle, bulge fraction, disc thickness.

---

## G14. THE HOME SEARCH COMES AFTER S12 (owner, 2026-08-28)

G10 says the home is FOUND — by having an Earth-like planet — not placed at the centre. **That work
lands after S12.**

S12 replaces the placement law. Finding a home is a search built ON that law. Doing both at once means a
bad search and a bad shape look identical, and neither gets diagnosed.

Until then the existing home system stays, deliberately: the immediate goal is to fly a ship realm and
test warp, and moving the start would change the thing under test.
