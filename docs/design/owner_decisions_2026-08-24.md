# OWNER DECISIONS — 2026-08-24

The signed record of the rulings the owner made walking the nine open questions of
`scratchpad/galaxy_sky_design.md` (the Universe / Galaxy / star-field design of record), plus the three
law changes made during that walk.

**Status: BINDING.** Where this file and any design document disagree, this file is later and wins.
Where a ruling reverses an earlier one, the reversal is named and dated here — a design that quietly
contradicts a written ruling is a defect.

**Numbers in this file are the DESIGN's, not measurements.** Every figure marked ⚠ must be re-solved by
the code's own solver, on THE world, before anything is built against it.

---

## PART A — THE LAW CHANGES

### A1. SL1 REWRITTEN — a realm is TOLD where it is; it never DECIDES where it is

Full text in `CLAUDE.md` SL1. Six clauses: the parent is the only writer; a parent MAY state to a child
the placement it authored for it; the child holds it stamped and read-only; the child never derives,
adjusts or states its own placement; **one hop only** (what you are told about yourself you never pass
on, and you are never told your parent's placement, so no realm can chain hops into an absolute); a
**stale reading is refused**, never used.

**This deliberately reverses the 2026-08-05 ruling** that a realm may never be told its own placement.
The owner's reasoning, recorded so it is not re-litigated: the old law forced TWO mechanisms for one job
— "contents plus your own eye" to a renderer, "a view composed for you" to a realm — a fork by RECIPIENT
KIND. The composed-per-observer form costs contacts × observers, and the sky arc measured that product
fatal. Being told where you are turns the product into a SUM for every consumer. The clause given up —
"a field's PRESENCE is the leak" — was about people, not correctness; clauses 4 and 5 replace that taboo
with a fence the compiler and a test enforce.

**Owed before the datum exists: the three fences of [[D-SL1-2]].** Do not build the reading first.

### A2. SL2 CLARIFIED — it governs REALM-TO-REALM, and the connection plane is not a realm

A shard handing its OWN occupants to the gateway was never a crossing — it is how any player has ever
seen anything. **Seeing** another realm's people is therefore lawful with no ruling: each shard streams
its own occupants to its own subscribed clients, and the GATEWAY composes one ready-to-draw picture per
observer. That is the lane [[D-RLM-18]] already specifies and owes.

**The owner's line for the simulation side: CONTACTS ARE REALMS, PEOPLE ARE SEEN.** A ship's systems
track ships, stations and bodies — placements the parent authors and may state — never individual
people. Acting on a person is a projectile CROSSING a boundary, i.e. the transfer machinery.

SL2 is clarified in `CLAUDE.md`, not amended.

### A3. SL9 ADDED — a parent's child count is unbounded

Full text in `CLAUDE.md` SL9. No cap, no reserved width, never a per-tick walk of all children, never a
per-child row on a per-tick lane. A cost that grows with the child count is a defect, and it must be
**measured on a wide realm**, not argued.

### A4. THE MOVEMENT LAW — the parent never sets a speed

> A parent **never sets an occupant's speed**. A ship realm computes its **own forces** from the engines
> that are running, with their angles, and passes those forces to its parent. The parent runs **its own
> physics** and from the forces it receives computes **positions**. A **maximum speed per realm** exists,
> but as part of the common law, never as a control.

Consequences the owner ruled in the same walk:

- **Warp time is never a policy number.** There is a warp **speed** and a **distance**; the time is what
  falls out of them. Later, warp speed varies by **engine type**, so it becomes a property of the ship.
- **Travel between two star systems must never be shorter than travel between two planets.** The inverse
  was measured real at true scale in the other worktree — crossing to another star came out faster than
  crossing your own solar system, because time was a property of the box.
- **A realm's maximum speed is DERIVED FROM SAFETY** (owner, this walk): it is the speed at which one
  tick of travel still fits inside the thinnest boundary band in that realm. Band width and top speed
  become two readings of one solve; a faster warp must be paid for with wider bands, explicitly.

**What this makes load-bearing, stated plainly:** today you cannot fly through a moon because the realm
lowers your ceiling as you approach. This law removes that guarantee. Its replacement is a band sized
from the REAL closing speed — and every band shipped today is built at zero relative speed, the widening
inert (D-WORLD-4b). **This law cannot land before the bands are speed-sized.**

**Also owed with it:** arriving gently becomes the SHIP's job (an autopilot firing engines backwards),
not the room's — so **a basic flight computer belongs in every hull from the start, not as an upgrade**,
or a new player cannot arrive. And a flown path accumulates, so it is checkpoint-carried, never
re-simulated: a ship's trajectory is durable state in a way an orbit is not.

### A5. AUTOPILOT BLOCKS — possible under A1 + A4, and three things will bite

The owner intends autopilot blocks that compute trajectories and signal the engines. This is possible,
and the law makes it MORE possible: under the old model the room handed you a speed and an autopilot had
nothing to control. Recorded risks:

1. **The control loop crosses a process boundary**, so it can oscillate. The loop delay must be a known
   bounded number the autopilot reads, not an incidental property of the day's network. During a
   hand-over the loop's far end changes host and the autopilot flies open-loop for a bounded window.
2. **Planning needs a model, not just a feeling.** To fly a curve near a planet the autopilot must
   PREDICT gravity. Two lawful routes: infer the field from how it is pushed over time, or **a realm
   states its own physical character** (its mass, its field) as a statement about ITSELF. ⚠ **This will
   probably become a fresh SL6 ask. It is flagged here so it is not discovered late.**
3. **Player-written autopilots are untrusted code in the server tick** — hard per-ship time budget,
   sandbox, and overrun degrades the ship, never the tick. Tie the loop rate to the computer block so a
   performance limit becomes a game rule.

---

## PART B — THE NINE QUESTIONS

### Q1 — the coordinate step: **YES, and take the bigger step**

The galaxy and the universe count their positions in bigger steps; a star system keeps millimetres.
**RULED: a TWO-metre galaxy step, not the document's one metre** — one metre leaves 5 % headroom against
what 150,000 systems need, and content grows; two metres costs nothing a player can see at light-year
distances and buys eight times the room. ⚠ the universe step and the resulting galaxy count are
unchanged by this (they depend on the universe's own step).

**Three conditions, all accepted:**

1. **The saved-data format stamp lands FIRST.** A position written in millimetres and read as metres puts
   a player 1,024× further out. Nothing we write to disk carries a format version today (D-48); until it
   does, this change cannot land on a world anyone has played in.
2. **The unit is part of the version handshake**, and a mismatch REFUSES the connection rather than
   guessing. A mixed-version fleet otherwise teleports players by 1,024× and it looks like a physics bug.
3. **Every diagnostic that prints a position prints its unit.** Cheap now, painful to retrofit, and the
   ambiguity costs real time during an incident.

Also recorded: this is a **one-way door**, free only while no galaxy-scale position has shipped.
`guard_root_representable` (`crates/physics/src/worldgen/guards.rs:69`) is hard-coded to the millimetre
step — **verified by reading the line** — and would refuse the new world by 1,024×; it must learn to read
the root frame's own step, in the same slice.

### Q2 — send a star's position on change: **YES, with four conditions**

Today every child's position ships every tick, by design, which is what keeps stars in the sky. At
150,000 that is ⚠ ~21.75 MB per player per tick, a login message ~27× over the frame cap, and a
galaxy→gateway message that is not split into packets at all — **an empty sky with no error anywhere**.

**Conditions, all in the same slice:**

1. The catalogue rides a **re-driven, acknowledged** lane — never a drop lane. A lost row for a thing
   that never moves is otherwise lost for the whole session.
2. **The receiver states its generation; the sender answers from that**, never from its own memory of
   what it sent. A sender must never infer what a receiver holds.
3. The keep-alive **compares a counter** instead of clearing the send-on-change memory — otherwise the
   whole catalogue re-ships twice a second, forever, and the optimisation cancels itself.
4. A **liveness digest**, so silence is provably "nothing changed" rather than "nobody is working". A
   stuck producer must not look like a quiet one.

**Owed measurement before it ships: a thousand players reconnecting at once.**

### Q3 — the minimum gap between systems: **ONE LIGHT YEAR as the floor**, plus two conditions

⚠ At one light year about one eightieth of space is forbidden (the field stays natural); at two light
years about a tenth (ordering begins to show); beyond that the sky reads as a grid. The other worktree's
jittered-lattice answer was **measured 5.75× too even** — a visible lattice — with both cures colliding
with standing laws.

1. **A binary is ONE system realm containing two stars.** Written down now, or the minimum-gap rule
   silently deletes binaries — and the nearest real star system to us is one.
2. **Recorded: the star-mass cap exists ONLY because a system's boundary is its gravitational reach.**
   If heavy stars are ever wanted, that assumption is the thing to revisit — not the gap. (Superseded in
   part by Q9: the gap becomes per-pair.)

### Q4 — the catalogue's message shape: **REVERSED — add the compact message**

The design recommended reusing the existing shape. **The owner's argument overturned it and I agree:**
the design reasoned as if the client were the only consumer, and the shape it wanted to reuse is the
CLIENT'S PICTURE shape (composed per observer, per instant, ready to draw). Handing that to a realm is
the wrong tool, not one tooling. The existing lane is composed per observer and per tick; the catalogue
is neither, so reusing it means **bending** it — more work and more risk than a small purpose-built arm.

**But the ship needs no message at all.** Every shard already folds the world from the seed itself; a
ship is a shard and holds the same generator. A star-map block asks the generator. Zero bytes cross.

**The client genuinely needs it** — it may not derive (the server does the arithmetic, the client
renders), and putting the world generator in the client hands every player the ability to enumerate the
galaxy offline. That door does not close again.

**Four conditions:**

1. **The client is its only recipient.** Nothing is sent to realms.
2. **A test asserts the encoded catalogue and the folded one are IDENTICAL** — one truth, two producers,
   which will otherwise drift at the first patch and nobody will notice.
3. The generation is **derived from the content**, never hand-incremented.
4. The client **caches it on disk**, and **the catalogue is a drawing aid only** — the server validates
   every warp destination itself. Otherwise an edited cache becomes "warp anywhere".

### Q5 — travel between galaxies: **THE TUNNEL** (neither option the document offered)

**Physical flight is dead under A4.** Neighbouring galaxies are roughly half a million times further
apart than neighbouring stars; whatever engine makes a star hop take tens of minutes makes a galaxy hop
take tens of years. The document's recommendation was honest under the OLD law, where a bigger room made
you faster. That law is gone.

**The ruling:** you fly to a relay, you fly IN, and the corridor **is an ordinary realm**. Entering and
leaving are ordinary crossings. Its walls are its own appearance (SL3). A **current** carries you, set
from the distance between the galaxies. You see no stars because the corridor is what is around you.

This gives every part of the owner's scripted description — a relay start, no other galaxy's stars, a
blend, a wait sized by distance, arrival in the destination system — with **no exception to SL8 and no
new machinery**. And the artificial wait becomes the **warm-up budget** for the destination.

**PLUS, owner-added: the tunnel realm is BUILDABLE BY PLAYERS, the same as ships.** It rides work
already owed — player-built ships already need realms created while the game runs, and today the region
set is fixed at start-up with no mutator.

**CONDITION, accepted: the tunnel's topology gets its own design pass before any code.** It is the first
thing in this design that is **not a tree** — a tunnel has two ends in two galaxies, while containment,
the directory key, the boot fences and the lineage path all read a single parent. The shape I believe
saves it: **the corridor's parent is the UNIVERSE**, a sibling of the galaxies, so travel is four
ordinary crossings — out into the shared parent and in again, which is SL2's own sentence. Confident,
not confident enough to build on.

**Open questions for that pass, recorded now:** who builds the far end (a one-ended frontier corridor, or
a slow one-way pioneer voyage first); whether you can stop, fight or log out inside; what happens when
the far side is not ready (the exit refuses and you sit at a closed door — design for this, not the happy
case); the corridor's interior length against the real distance between galaxies (nobody sees both, but
it may show as a jump at the mouth); gate ownership, destruction and stranding; and the denial-of-service
surface — **a player who can create a realm can make the cluster start processes**, so a gate must cost
enough and be capped where realms are started, not where they are built.

### Q6 — the SL8 exception: **DISSOLVED**

It existed only if Q5 chose the scripted version. It did not. **No exception to the seamless law is
needed anywhere.**

### Q7 — the 2026-08-05 reversal: **CONFIRMED**, with four recorded consequences

Compression becomes exactly 1 (this worktree squeezes the star gap by ⚠ ~24.6× today); the chart becomes
the model; the cell lattice is dropped. **This is less of a reversal than it looks** — the 2026-08-05
ruling itself reserved the coarse coordinate type as "the cheaper alternative fix if subdivision proves
awkward". It proved awkward.

1. **The lattice is dropped; LAZY GENERATION IS KEPT.** They arrived together and only one is returned.
   A galaxy with 150,000 children must never enumerate them at boot.
2. **The occupancy reason the owner gave for the lattice is UNANSWERED.** Every traveller between stars
   now lands in ONE process. ⚠ Nobody has measured a galaxy shard's tick against traveller count; that
   measurement is owed. The reserve, if it saturates, is splitting the galaxy's AUTHORITY across
   processes without creating new realms — one galaxy, one frame, one identity, several hosts.
3. **The brightness-knob ruling loses its reason.** It was ruled a knob because a 24× squeeze made a
   neighbouring star ~10⁵× brighter than reality. With no compression the sky is physically correct. A
   brightness control survives as a PICTURE control, not a physics one, and the record must say so.
4. **Two rulings of that day stand unchanged and both get stronger:** a system's boundary is never drawn
   as an object, and all movement works in deep space and is slow by design.

Also recorded: compression at one **moves every distance in the world**. Fixed test values, recorded
pictures and flight budgets all shift — the largest single re-measurement in the plan, to be scheduled
as one.

### Q8 — how many galaxies: **61**, with two conditions

⚠ About 9 million star systems in the world, at a Local-Group-scale spacing. This is a **growth ceiling,
not a starting world**: a second galaxy needs a player-built gate, so an unreached galaxy is not
generated, not simulated, and costs nothing.

1. **Check that the generator places by POSITION, not by counting outward.** If it does, enlarging the
   universe later materialises more at the edge and touches nothing discovered — the decision becomes
   reversible and stops being a launch-freezing one. **Unverified; worth more than the choice itself.**
2. **Record that galaxies must DIFFER** — in density, star population, what is found there. Every galaxy
   is drawn from the same rules, so galaxy 61 is otherwise 150,000 systems statistically identical to
   galaxy 1, and the whole feature is a long corridor to more of the same. The taxonomy already knows
   galaxies have kinds.

Standing observation, not a condition: **keep gates expensive.** They are the only thing keeping the
world's size matched to its population. One well-known space game runs ~8,000 systems and feels vast;
one of our galaxies is ~18× that.

### Q9 — the heaviest star: **DO NOT accept the loss — the gap becomes PER-PAIR**

The document asked the owner to accept a world with no giant stars. ⚠ A single global gap of one light
year caps the heaviest star near six suns, which removes all O stars, the upper half of B, the
supergiants, and every black hole above the cap. Neutron stars survive (a remnant is light). **The visual
cost is larger than the count suggests** — brightness rises as roughly the 3.5 power of mass, so this
removes the brightest things in the sky, which are also the navigation landmarks in a field of 150,000.

**Measured, and it decided the question:** REDUCING THE SYSTEM COUNT DOES NOT HELP. Separation is set by
DENSITY, not by count — at real density the mean nearest neighbour is ⚠ ~3.9 ly whether the galaxy holds
120,000 systems or 150,000; fewer systems only makes a smaller galaxy (⚠ 232 ly → 215 ly) with neighbours
exactly as close. The only knob on that road is density, and it is a bad trade: **⚠ one eighth the
density to double the gap** — an eight-times emptier sky — and the cap still only reaches ~9 suns, so the
O stars are still gone.

**RULED:**

1. **The minimum gap is the two systems' OWN RADII plus a margin**, with one light year as the floor for
   ordinary pairs. A rare giant simply pushes its neighbours away, which is what a giant star should do.
   The binding limit then becomes the coordinate one — ⚠ around thirty suns by the design's own solve —
   so in practice **every type is kept**.
2. **⚠ Re-solve the real cap with OUR solver, on THE world, at our step, before building.** Thirty is a
   number from a neighbouring design and must not be built against.
3. **The star field must never be evenly spaced** (owner, explicit). The per-pair rule and a natural
   draw-and-retry placement give that; a jittered lattice does not, and was measured not to.
4. **The system count stays 150,000.** The count controls world size and content density, which is what
   it should be chosen for — not this.

Recorded consequences: the generator must place with a **varying** exclusion radius and stay lazy, so its
neighbour search must reach far enough to catch a distant giant's larger claim. Rare giants clear wide
**voids** — physically pleasing, to be looked at once generated rather than assumed.

---

## PART C — WHAT THESE RULINGS MADE URGENT

Not new work; work whose priority moved.

- **D-SL1-2** (new) — the three fences SL1's rewrite depends on. The law is a promise until they exist.
- **D-9** — snapshots are a whole-realm broadcast with no per-entity interest filter, double-encoded.
  A2 (seeing other realms' people) makes this the enabling defect for any crowded scene.
- **D-RLM-18** — the read-sub lane for remote avatars. A2 says it is lawful and owed; it also owes a
  range and line-of-sight bound on streaming a hull's INTERIOR to outsiders, which is an exploit surface
  rather than a rendering detail, and a mixed-tick join.
- **D-WORLD-4b** — bands built at zero relative speed. A4 cannot land before they are speed-sized.
- **D-48** — nothing written to disk carries a format version. Q1 cannot land before it does.
- **Runtime realm creation** — no mutator exists for a shard's region set. Q5's player-built tunnels and
  player-built ships both need it.
- **The universe level** — was the last, optional slice. Q5 makes it load-bearing.

---

## ADDENDUM — 2026-08-26: THE GALAXY HAS A SHAPE

Raised because I read the shipped placement law and it is not a shape at all. Ruled by the owner the same
day, in three sentences: **"Shape please. Also never use the same distances between star systems. Of course
placement should be part of the seed and do not change."**

### What the code does today, and why nobody had noticed

`system_center_at` (`crates/physics/src/worldgen/generate.rs:219-235`) draws a **direction** from the seed
and multiplies it by `system_ring_r_m`, which is **one constant**. The home system is anchored at the
centre; every other system sits on the surface of one sphere, at exactly the same distance from the middle.
The shell has **zero thickness** — not thin, zero.

It is invisible at the shipped THREE systems, because two points on a sphere and one at the centre look like
nothing in particular. At 150,000 it is the first thing a player would see: every star in the sky exactly as
far away as every other, one thin layer to fly through, and **nothing at all between them or beyond**.

It also contradicts the plan's own arithmetic, which describes 150,000 systems occupying *"47.5 % of the
radius and 10.7 % of the volume"* — a filled ball. A surface has no volume to fill. Only one of the two is a
galaxy.

### THE RULING, in three parts

1. **A GALAXY HAS ITS OWN SHAPE, and the placement follows it.** The taxonomy already names three
   (`GalaxyType::{Spiral, Elliptical, Irregular}`, `crates/physics/src/taxonomy.rs:40-44`) and already draws
   a galaxy's type from its seed. The placement law must consume that draw instead of ignoring it. A hollow
   shell is not one of the three, and a plain filled ball is not either — it was my own suggested first
   step, and the owner declined it. **Build the shape.**
2. **NO TWO SYSTEMS AT THE SAME DISTANCE.** This extends ruling 3 above (*"the star field must never be
   evenly spaced"*, already recorded as measured, with a jittered lattice explicitly failing it) to the
   RADIAL axis, which the shell violates absolutely: today every system shares one radius exactly. No
   constant radius, no ring, no shell, and no lattice-quantised position. ⚠ **This constrains the S9 design
   run's proposal directly:** it offered a density-threshold CELL LATTICE for placement. A lattice is
   lawful as a LOOKUP structure and is refused as a POSITION source.
3. **PLACEMENT IS SEED-DERIVED AND STABLE.** Where the seed puts a star, the star stays. Adding systems must
   not move the ones already placed — which is exactly the gate S12 already carries (*"grow the census N→N+1
   and every existing system is bit-identical"*), so the ruling and that gate are one requirement stated
   twice. Position-addressed placement, never an index into an ordered draw.

### What this changes, and where

- **The placement law is now S9's business, not a later slice's.** Moving the radii without fixing the shape
  would ship a bigger hollow shell.
- **The design run's headline number is NOT the reason.** It concluded the current law caps the world at
  ~12 systems and that 150,000 would sit 28× inside the one-light-year floor. **I checked and it is wrong:**
  150,000 systems on that sphere at the S9 radius are ≈ 4.46 ly apart, comfortably ABOVE the floor, and the
  cap at a 1-ly floor is ≈ 674,000. The spacing was never the problem. **The shape is.** Do not build
  against the 12.57 figure or the 28× figure; both come from comparing the spacing against the galaxy's own
  radius rather than against the floor.
- **Still owed a measurement:** what a shaped, non-uniform, seed-stable placement does to the per-pair
  minimum-gap fence and to the child lookup at 150,000. Neither is answered by this ruling, and neither may
  be argued — the standing rule is that a cost which grows with the child count must be measured on a realm
  with many, not reasoned about.
