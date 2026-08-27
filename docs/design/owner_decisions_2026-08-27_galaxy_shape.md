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

## STILL OPEN

- **How faithful is the spiral?** A bulge, a disc thickness and an arm pitch angle are real modelling
  numbers. They must be seed-derived (G1), but their FORM — the law they are drawn from — is not decided
  here.
- **Does the count follow from density and size, or is it still stated?** G1 says every number the
  placement reads is seed-derived, which points at the count being an OUTCOME. The plan still carries
  `census 150,000` as a target. These must be reconciled in S12, and the gate re-stated in terms of what
  is actually measured.
