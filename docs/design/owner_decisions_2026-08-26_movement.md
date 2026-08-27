# OWNER DECISIONS — 2026-08-26, the movement walk

**These are LATER than `owner_decisions_2026-08-24.md` and later than every design document. Where they
disagree with a plan, a design or a code comment, THESE WIN.** They extend the movement law (A4 of the
2026-08-24 file); they do not replace it. Read that file's A4 first — it is quoted where it binds.

---

## M1. THE WIDTH-OVER-THREE-MINUTES RULE IS RETIRED

The owner rejected it in as many words: *"the rule is quite terrible and non-scientific, let's not use
it."*

It is how the shipped code works today: a realm's top speed is twice its own extent divided by a fixed
180-second traverse constant, floored at a foot speed. That is an arbitrary number wearing a formula.

**Its replacement was already ruled on 2026-08-24 and needs no new decision:**

> A realm's maximum speed is DERIVED FROM SAFETY: it is the speed at which one tick of travel still fits
> inside the thinnest boundary band in that realm. Band width and top speed become two readings of one
> solve; a faster warp must be paid for with wider bands, explicitly.

So: **top speed = thinnest band in the realm ÷ one tick.** A 700 m band at a 0.05 s tick gives 14,000 m/s.
Faster is bought with a wider band, and the cost is visible in the same solve.

⚠ **This is not yet what the code does.** The traverse constant is live and load-bearing — see M4.

---

## M2. WHAT A CHILD REALM HANDS UP: ACCELERATION AND TORQUE, NOTHING ELSE

**RULED: option B.** The owner chose it directly: *"Go with option B, acceleration / torque only"*.

This settles a conflict between two binding passages that disagreed and neither of which was built: one
listed a force with an inertia tensor and no power state, the other a power state and no inertia tensor.

### The contract

| Direction | Data | Rate | Stated in |
|---|---|---|---|
| **UP** (child → parent) | linear acceleration (3) + angular acceleration/torque (3) | **per tick** | the **CHILD'S OWN** frame |
| **UP** (child → parent) | mass, cross-section, drag coefficient | **ON CHANGE ONLY** — when the hull is rebuilt, never per tick | — |
| **DOWN** (parent → child) | the placement the parent authored: position, velocity, facing, spin, and the instant | per tick | the parent's frame, as a stamped read-only reading (SL1 clause 2) |

### M2a. THE PARENT HOLDS THE MASS — ruled 2026-08-26, same walk

The owner ruled it directly: *"Go with way 2"*, after asking whether drag and gravity can be computed at
all without masses. The answer separates into three cases, and only the first was a mistaken premise:

- **GRAVITY NEVER NEEDED IT.** The child's mass CANCELS: the pull is `G·M·m/r²` and the acceleration is
  that divided by `m`, leaving `G·M/r²`. A 50,000 kg ship and a 1 kg drone both fall at 8.13 m/s² at
  7,000 km from an Earth-mass body. The parent needs only ITS OWN mass and the distance, and it holds
  both. *(Exception: a child heavy enough to pull on its SIBLINGS. Those are generated bodies — planets,
  moons — whose masses the world generator already gives the parent. Nobody has to tell it.)*
- **DRAG GENUINELY NEEDS IT.** Drag is a push from OUTSIDE, so dividing by mass does not cancel it. Two
  ships of identical shape at identical speed in identical air take the same 30,000 N: a 50,000 kg
  freighter slows at 0.6 m/s², a 5,000 kg racer at 6.0 m/s² — ten times apart.
- **COLLISIONS NEED IT** and cannot be done in accelerations at all; an impulse must be shared out by mass.

**THE REJECTED ALTERNATIVE, recorded so it is not re-proposed:** the parent could state its medium
DOWNWARD and let the child compute its own drag, keeping mass sealed. It was refused because the child
would use the velocity it was TOLD, which is one tick old, and lagged velocity in a strong-drag regime
oscillates or diverges. Exactness beat sealing.

**WHY MASS IS NOT A LEAK.** The parent already holds every child's size and shape — that is how it decides
who contains what. Mass is no more private than extent, so sealing it bought nothing.

**THE SPLIT THIS GIVES, and the reason engines stay an acceleration while drag uses a mass:**

| The child states | Which is | Rate |
|---|---|---|
| acceleration + torque | **what I am DOING** — the child knows best how hard its own engines push it, because it knows its own mass | every tick |
| mass, cross-section, drag coefficient | **what I AM** — a declared property, like extent | only when it changes |

The parent uses the second only for forces IT applies: drag now, impacts later. **The per-tick lane is
still six numbers and still acceleration and torque only.** Nothing about the hot path changed.

**VELOCITY STILL NEVER CROSSES UPWARD** — see below. Mass crossing does not reopen that.

**THE CHILD SPEAKS IN ITS OWN FRAME** — *"this hard, along my own nose"*. It does not know where its nose
points in the parent; the parent does, because the parent authored its facing, and the parent rotates.

**VELOCITY NEVER CROSSES UPWARD.** A velocity is half of a placement, and SL1 clause 3 forbids a child
stating its own placement. Velocity travels DOWN only. It is the answer, never the question. (The owner's
phrasing in the walk was *"ship generate velocity and other forces"*; the velocity half was raised as a
conflict with SL1 and the acceleration-only contract was chosen instead.)

**THE PARENT NEVER CHOOSES A SPEED.** It sums what it is handed with its own ambient accelerations —
gravity today, drag later — and integrates. The speed that results is the child's engines' business.

### The worked example that fixed the shape

A ship crosses a star system and enters a planet's sphere of influence.

1. The ship adds up its running thrusters, divides by its own mass, and sends *4 m/s², along my nose*.
2. The system rotates that into its own frame and adds the star's pull.
3. The system advances the ship one tick and writes its new position and velocity.
4. The system sends the ship its reading. The ship holds it read-only and **never passes it on** to its
   own crew (SL1 clause 4, one hop).
5. The ship's new position falls inside the planet's bound. **Only the system can see this** — it is the
   one party holding both placements.
6. The system SUBTRACTS: ship position minus planet position = the ship's position in the planet's frame,
   and hands the ship down. The ship does nothing; it is carried. Out to the shared parent and in again,
   never sibling-to-sibling (SL2).
7. From the next tick the ship sends **the identical six numbers** to the planet, which adds **its** own
   ambient instead. Nothing about the ship changed. It does not know it moved house.

**That last line is the acceptance test for this design.** One mechanism; any realm can be the parent.

### What this defers, stated so it is not a surprise

- **DRAG** is now the PARENT's arithmetic (M2a). The realm holds its own medium; nothing about the medium
  crosses in either direction. What is still owed is the medium itself — no realm states a density today.
- **COLLISIONS** have their mass (M2a) but no solver. An impulse exchange is a different lane from
  per-tick motion and its shape is not decided here.

---

## M6. WARP IS A DECLARED STATE — and the GATE that stops it becoming a back door

**RULED 2026-08-26.** The owner chose the declared-state form over the absurd-acceleration form, and
immediately asked for a gate: *"It should work and make sense for Warp and probably for the cross-galaxy
travel, but we should not use for, for example, autopilot."*

### Why warp cannot be an ordinary engine, measured

A warp drive is not a big thruster. Reaching a galaxy-crossing speed (5.124e15 m/s) by acceleration:

| At | Takes |
|---|---|
| 10 g | 1.66 million years |
| 1,000 g | 16,600 years |
| 60 seconds | needs 8.71e12 g — **and you travel 16.2 LIGHT YEARS during the ramp**, further than the star you aimed at |

So warp changes the MEDIUM's relationship to the ship, not the push. That is a property of the hull, which
is the slow lane we already have (M2a) — the same lane mass rides on.

### What it does

The declared state tells the parent's physics **which medium law applies to this ship**. The parent needs
it for exactly two things, both of which are the parent's own arithmetic:

- **DRAG.** At warp speed, normal-space drag would be ruinous. In warp it is not.
- **THE SAFETY SOLVE.** A warping ship is not caught by ordinary bands, and its wake radius must grow with
  its closing speed (M3.2).

**The ship applies its own warp multiplier before it speaks.** It still sends one acceleration on the
per-tick lane. Nothing about the hot path changes.

### ★ THE GATE — five tests, ALL must pass

A declared state is lawful ONLY if:

1. **It says what the ship IS, never what it WANTS.** A property, not an intention.
2. **It changes how the WORLD acts on the ship, never what the ship DOES.** What the ship does is an
   acceleration, and acceleration has exactly one lane.
3. **IT SURVIVES AN EMPTY SHIP.** Take every person off. Stop every program. Does the value still stand?
   Mass does. A stuck-on warp field does. A destination does not.
4. **It changes rarely.** It rides the slow lane. Anything that changes per tick is an intention wearing a
   property's clothes.
5. **IT CANNOT MOVE YOU BY ITSELF.** Declaring it must produce NO motion. Warp removes what was stopping
   you; you still fire engines to go anywhere. **If a state alone moves you, it is a velocity in disguise
   and it is refused** (SL1 clause 3).

Test 3 is the anti-autopilot test. Test 5 is the anti-velocity test.

| Candidate | 1 | 2 | 3 | 4 | 5 | Verdict |
|---|---|---|---|---|---|---|
| Mass | ✓ | ✓ | ✓ | ✓ | ✓ | **lawful** |
| Cross-section / drag coefficient | ✓ | ✓ | ✓ | ✓ | ✓ | **lawful** |
| Warp field running | ✓ | ✓ | ✓ | ✓ | ✓ | **lawful** |
| Landing gear out (changes drag) | ✓ | ✓ | ✓ | ✓ | ✓ | **lawful** |
| Autopilot destination | ✗ | ✗ | ✗ | ✗ | — | **REFUSED** |
| Desired cruise speed | ✗ | ✗ | ✗ | ✗ | ✗ | **REFUSED** |
| "Slow me for docking" | ✗ | ✗ | ✗ | ✗ | ✗ | **REFUSED** |

**An autopilot is software pressing the stick.** It gets no special path. It computes an acceleration and
sends it on the ordinary per-tick lane, exactly as a hand on a stick does. That is the whole of its
privilege.

### ★ THE STRUCTURAL FENCE, because five remembered rules are not a fence

**The set of declared states is CLOSED and lives in ONE reviewed place.** Adding one needs the owner's
word, exactly as adding a wire arm does (SL6). The five tests above are how a candidate is argued; the
closed set is what makes the gate hold when nobody remembers the tests.

### Warp cannot hold near mass

The drive fails; the parent does not clamp. The parent's own physics decides whether the medium here
sustains the state, and if it does not, the state lapses and the ship is TOLD — a stamped fact about
itself, which SL1 clause 2 already allows.

This is what makes the geometry work. **You reach top speed in the middle, where there is nothing.** You
are slow near the star you left and near the star you are arriving at, because a real flight speeds up and
slows down. Measured: to the next star (4.46 ly) in 30 minutes, the peak needs a band 15× the system — but
that peak happens in deep space, and near the systems you are slow enough for the bands they already have.

---

## M7. THE INTEREST SIGNAL CARRIES A DISTANCE — approved under SL6, 2026-08-26

**THE ASK, AND THE ANSWER.** SL6 requires the owner's word before new data crosses a realm boundary. The
interest signal ("someone outside may be looking in") today carries a YES/NO. It now carries **one number:
how far away the looker is.** The owner approved it: *"Agree with the distance signal, go ahead."*

### Why the signal had to change at all

A realm told "someone is looking in" today wakes **every** direct child. The down-proxy sits at the realm's
own origin with reach equal to the realm's own extent, so every child's distance clamps to zero and every
child is in band **by construction**. At the target census that is 150,000 realms woken, and 150,000
messages per tick. No change of loop shape can fix it: every child really is in range.

### The form that was TRIED FIRST AND REJECTED, recorded so it is not re-proposed

The owner's instinct was to use the looker's own area of interest. The direct form fails on geometry, and
it was checked rather than assumed. Given the looker's reach `R` and its distance `D` from the child, a
direction-free bound on what the looker can see inside the child is `R + D`, **not** `R − D` — the
triangle inequality runs the wrong way for an over-approximation. `R + D` exceeds the realm's own extent,
so it wakes everything again.

Doing better needs the looker's DIRECTION. Direction plus distance is the looker's position, and **SL2
forbids an occupant's pose entering another realm's simulation.** That road is closed.

### The form adopted: PROXIMITY needs a direction, SIZE DOES NOT

How big a thing looks depends only on how far away it is. So the parent sends the **distance alone** —
direction destroyed, and a position cannot be rebuilt from a scalar — and the child wakes only those of its
own children whose ANGULAR SIZE at that distance clears the visibility threshold. This is SL3's own logic
(a realm draws itself; a thing is worth drawing when it is big enough to see) applied to waking.

**MEASURED on THE world, at the shipped threshold of 0.026180 rad (1.5°):**

| Looking at | From | Angular size | Wakes? |
|---|---|---|---|
| An Earth-sized planet | 1 AU | 8.56e-5 rad | no |
| An Earth-sized planet | 1 light year | 1.35e-9 rad | no |
| A big planet (2.0e9 m) | 1 light year | 4.23e-7 rad | no |
| A star system (1.58e11 m) | 1 light year | 3.34e-5 rad | no |

From a light year away **nothing inside a star system is worth waking.** You see the star; you do not see
its planets. The count goes from 150,000 to approximately none, and that is not a trick — it is true.

### The SL6 answers, stated as the law requires

- **WHAT DATA:** one scalar, the distance from the nearest outside looker to this realm. No direction.
- **FROM WHICH REALM TO WHICH:** parent → direct child, on the lane that already carries the interest bit.
  It is not a new lane; the existing signal carries a number where it carried a flag.
- **WHY THE RECEIVER CANNOT COMPUTE IT:** a realm does not know where it is (SL1), so it cannot know how
  far anything is from it.
- **WHAT DOING WITHOUT COSTS:** waking all 150,000 children, every tick.

### ⚠ WHAT THIS DOES NOT DO

It does not invert the fold. The loop still walks every child, and that change is DEFERRED TO S12 (M8).

---

## M8. THE FOLD INVERSION IS DEFERRED TO S12 — the tests cannot prove it yet

**RULED 2026-08-26:** the owner refused a hand-planted wide-field fixture — *"Defer to 12 please, lets test
on the real one."* The generator cannot produce a galaxy with many systems until S12 (today: three).

**WHY THAT DEFERS THE CHANGE ITSELF, not just its test.** MEASURED against the 499 sim-tier tests: a
mutation that makes the fold silently SKIP A CHILD fails only **8 of 499**; building the lookup grid on the
wrong radius fails only **12 of 499**. The widest AoI fixture in the tree has **two** children, often
co-located. So the inversion could ship broken — tearing down a realm while an occupant still watches it —
and the suite would stay green. That is the worst place in the system to guess.

**WHAT LANDS NOW INSTEAD (S10):** the two pieces that are provable against the world we have —
the latch re-key (behaviour-identical, so todays tests DO prove it) and the angular-size wake (M7).

---

## M3. STILL OPEN — the owner has NOT ruled on these

1. **THE ROCK.** A ship has engines; a rock and a fired shell do not. The 2026-08-19 ruling — *"the
   realm's ceiling governs everything the realm contains, piloted or not"* — is recorded as landed, and
   the 2026-08-24 movement law neither names ungoverned things nor carries a reversal note (that same file
   DOES write an explicit reversal for the placement law, so its absence here is meaningful). If rocks
   stay governed, the old ceiling survives on the ballistic path for one class of thing. If they do not,
   an ungoverned rock crosses a star system's band in under one tick, which is the hole the 08-19 ruling
   closed. **Unresolved in writing.**
2. **WAKING UP IN TIME.** The safety solve names ONE test (one tick fits inside the thinnest band). There
   is a second: a realm needs time to start, and a realm that is not running cannot be drawn (SL3), so at
   high closing speed you arrive at a dark realm — a seam (SL8). Either the common-law top speed is the
   STRICTER of the two tests, or the interest radius grows with the observer's measured closing speed,
   which the parent legitimately holds since it authors the pose.

---

## M4. WHAT THIS COSTS TO BUILD — measured by reading, not by running

**Both halves of the law are missing from the tree.** Reported so the sequencing is an owner choice and
not a surprise:

- **No physics engine exists.** 708 dependencies, no rigid-body solver.
- **Nothing that moves carries a mass, an engine, a thrust figure or a hull size.** The only masses in the
  tree describe stars and feed closed-form orbit sums.
- **The parent side is missing too.** Every occupant advance passes zero acceleration explicitly.
- **The lane a child's numbers would ride is sealed** with no port in it.
- **The traverse constant is not a ceiling above a speed — it IS the speed.** A stick position is a
  fraction of the realm's number; velocity is read back OUT of the step just taken and never accumulates.
  Remove it with nothing in its place and every occupant falls to foot speed. MEASURED on THE world:
  crossing the galaxy (9.2234e18 m) at 15 m/s takes **1.95e10 years** — longer than the universe has
  existed. (A reader's figure of 280,000 years for the same trip was wrong by about 70,000× and is
  recorded here as retired.)

**Therefore this is a PHASE, not a slice.** An interim exists and was discussed but is NOT ruled: a ship
states its own rated cruise speed and acceleration as facts about itself, and the parent applies them.
That is lawful (an engine rating is not a placement) and stops the parent choosing immediately. It is not
a second mechanism — the parent's side is the same shape in both, and only the child's statement gets
richer later.

---

## M5. CONSEQUENCE FOR THE GALAXY PLAN'S SLICE 10

Slice 10 proposed to keep the approach governor and make its per-child walk cheap. **The 2026-08-24 law
removes that governor** — its own text says *"this law removes that guarantee"* about not being able to
fly through a moon. So that piece of S10 is deletion, not optimisation.

⚠ **But the bands were sized ON the governor.** Slice 6 sized every band in the world from the governor's
guarantee, and measured the alternative: sizing against the parent's ceiling instead *"would ask for bands
thousands of times larger than the bodies they wrap."* Removing the governor without settling M3.1 and the
band solve re-opens that measurement. **S6's numbers are not safe across this change.**
