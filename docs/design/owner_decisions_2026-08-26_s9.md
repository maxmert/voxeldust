# S9 OWNER DECISIONS — asked and answered one at a time (2026-08-26)

## Q1 — may the universe and the galaxy have names of their own? **YES, IDENTITY ONLY.**

**Why it was asked.** SL6: ask before adding a wire arm, default NO. S9 adds arms.

**The problem, in the code.** The universe and the galaxy have no `RealmId` of their own. They borrow
stand-ins — `UNIVERSE_STANDIN = RealmId::System(0)` and `GALAXY_STANDIN = RealmId::System(1)`
(`crates/core/src/realm_path.rs:25-27`) — and `RealmCoord::lowered()` (`realm_coord.rs:49-56`) resolves to
them. `lowered()` **is** the directory key. The borrowing works only while a galaxy owns nothing and nobody
asks. S9 makes a galaxy own star systems, so the question starts being asked and the answer is ambiguous.

**RULED:**
- `RealmId::Galaxy(u64)` — the galaxy's own seed. Derived from the world seed, never random, never moving.
- `RealmId::Universe` — **fieldless**. There is exactly one, and a field that always holds the same value
  is a field that will eventually hold a different one by accident.

**AND THE SPLIT THE OWNER APPROVED — identity now, NAMES LATER.** These two questions were being conflated
and they are not the same:
- **Identity** is what a machine files a record under. It must be a number, seed-derived, stable.
- **A LABEL is what a player reads, and today it is unusable.** `FrameRef::label()` (`pose.rs:103-116`)
  formats a raw seed: a player is told they are in **`System 10487570625701098367`**. There is NO name
  generation anywhere in the workspace — verified by search, not assumed. At 150,000 systems that is what
  every player sees, every time.
- **RULED: a generated-name slice of its own, BEFORE the census raise at S12** — which is when 150,000
  twenty-digit numbers first exist. A syllable table keyed on the same seed gives millions of stable,
  offline, free names. Doing it inside S9 would mean designing a name generator while moving every radius
  in the world: two unrelated risks in one commit.

**Consequential wire items that follow mechanically from Q1** (stated, not separately asked — each is a
notification rather than a decision, and each is refused loudly if a peer disagrees): the galaxy frame must
say WHICH galaxy; the universe needs a frame at all; and the protocol minor and its floor move.

## Q2 — may S9 add the far-field star message? **YES**, and it is ALSO the star map.

**Why it was asked.** A star system is a realm, and a machine is told about the realm it is in and the ones
near it. Stars are never "near" — the closest is four light years. So under the shipped rules **nothing
tells a client that any star exists**, and a player would fly through 150,000 of them and see a black sky.
No error, no warning, no missing message: the arithmetic is right, they genuinely are not near any of them.
**A correct black sky is worse than a broken one, because nothing distinguishes them.**

**RULED: add it in S9**, rather than waiting for S11's sky-lane rebuild. It carries a distance FROM THE
OBSERVER, never a position in the world, so SL1 holds — a realm still never learns where it sits.

### ★ AND IT IS THE STAR MAP TOO — one catalogue, two drawings

The owner asked whether the same message could drive a HUD star-map view. **It can, and this is not new
scope — it is the Q4 ruling of 2026-08-24 already recorded.** The sky and the map differ only in how the
client draws the same rows. Neither needs a mechanism of its own.

**The owner re-affirmed the four Q4 conditions, and they bind here:**
1. **The client is its ONLY recipient.** Nothing is sent to realms. A ship needs no message at all — every
   shard already folds the world from the seed, and a ship is a shard, so a star-map block asks the
   generator and **zero bytes cross**. That was the owner's own observation and it is what keeps this from
   becoming a second data path.
2. **A test asserts the encoded catalogue and the folded one are IDENTICAL** — one truth, two producers, or
   they drift at the first patch and nobody notices.
3. The generation is **derived from the content**, never hand-incremented.
4. The client caches it on disk and **it is a DRAWING AID ONLY** — the server validates every warp
   destination itself, or an edited cache becomes "warp anywhere".

**Why the client needs it at all, restated so it is not re-litigated:** the client may not derive (the
server does the arithmetic, the client renders), and shipping the world generator to clients would hand
every player the ability to enumerate the galaxy offline. That door does not close again.

## Q4 — how much of the galaxy is held, and how much is sent? **LIST THE STARS. DERIVE THE PLANETS. SEND NEITHER IN BULK.**

**Owner, 2026-08-26, two statements:** *"Just list all of them at once, don't do lazy smart."* and *"You
don't need to hold and pass planets / asteroids. You just need stars. You need info about the planets only
when you spin up the Star System when anybody arrives there... we can use a function to get this info from
the generator based on the seed. We don't need to pass all planets to the client to render."*

### THE MEASUREMENT THAT MADE THE FIRST RULING RIGHT

| | Cost at the 150,000 census |
|---|---|
| Every system held at once (`RealmRegion` = **336 bytes**, MEASURED) | **50.4 MB** |
| Every region including planets (11 per system, 1.65 M rows) | 554.4 MB |
| The shipped all-pairs separation fence | **11,249,925,000 pair tests** |

**Fifty megabytes is nothing.** The design run's position-addressed lazy generator — cells, hashes,
per-cell candidate draws, rejection machinery — was avoiding a cost that does not exist. **It solved the
wrong problem:** the expense was never holding the list, it was the ALL-PAIRS CHECK, and that is fixed by a
spatial bucket, not by a new generator.

### THE THREE-WAY SPLIT, which is the part I was about to conflate

1. **HELD:** the galaxy's stars. 150,000 rows, 50 MB, built at boot. Simple, flat, enumerated.
2. **DERIVED ON DEMAND:** a system's planets, from `f(seed, system)`. Available to ANY shard at ANY time —
   on arrival, and for a HUD that wants to say what a system contains. **Not stored, not a cache, not a
   lazy-generation scheme** — just a function nobody has to remember to call twice.
3. **SENT:** neither in bulk. The far-field catalogue carries STARS (Q2). Planets are never broadcast.

**★ THE SPLIT IS FREE, AND THIS IS WHY.** `generate_system_forest`'s own doc
(`crates/physics/src/worldgen/generate.rs:243-249`, READ) already states that *"each system draws its
planets from its OWN realm_stream, keyed on its own lineage, in a fixed order — so a shard hosting system 4
generates byte-identical planets to every other shard's view of system 4, without any shared state."* The
per-system draw is ALREADY independent. Not drawing planets until someone asks moves no planet by one
metre. **VERIFIED by reading the stream derivation, not assumed from the doc.**

### THE ONE CONSEQUENCE FOR THE CLIENT

A client may not derive (Q4 of 2026-08-24: shipping the generator to clients hands every player the ability
to enumerate the galaxy offline, and that door does not close again). So a HUD that wants a distant
system's planets ASKS, and the answer comes back for that one system — a request and a reply, not a
broadcast. That is a different shape from the star catalogue and must not be folded into it.

### WHAT THIS CANCELS

The entire lazy placement machinery. What survives from that design run is only what the owner's rulings
actually require: **the shape** and **the varied spacing**. The cleverness around them is deleted.

## Q5 — ★ EVERY GENERATOR TAKES TIME. A realm was ALIVE while nobody was there.

**Owner, 2026-08-26:** *"Please make sure that generator consider temporality. All generators should
consider temporality. Player should feel that system or any other realm were alive while they were not
there."*

### THE PATTERN IS ALREADY PROVEN — for one thing

`crates/physics/src/celestial.rs:3` states the shape in its first line: **`f(seed, universe_tick)`, never
`Σ f(0..T)`.** A planet's orbit is a STATIC description (`OrbitalElements` — semi-major axis, eccentricity,
inclination, the node, the periapsis, and the mean anomaly AT EPOCH) plus a closed-form evaluation at a
time. Arrive after a thousand years of absence and the planet is where a thousand years put it, computed in
one step — not by replaying a thousand years of ticks.

**That is the shape the ruling generalises.** It is not new machinery; it is an existing discipline that
one module follows and the others were not required to.

### WHAT IT MEANS, AND THE TRAP IT AVOIDS

A generator with the signature `f(seed, realm)` answers the same thing forever. A player who leaves and
returns finds a world frozen at the moment of its birth — *"nothing happened while you were away"*, which
is the opposite of the stated pillar.

**The trap is not the missing feature; it is the missing ARGUMENT.** A generator shipped as `f(seed, realm)`
gets consumers, goldens and wire messages built on a timeless answer. Adding time later is then a flag day
across every one of them. **Taking the argument from the start costs nothing today and is the only cheap
moment to do it.**

**RULED: every generator takes the instant, from day one** — including where today's answer does not vary
with it. A field that is genuinely time-invariant (a star's mass, its spectral class, a planet's radius)
stays invariant; what must take time is anything a player could notice having CHANGED.

### THE CONSEQUENCE FOR S9, CONCRETELY

The planets accessor ruled in Q4 is **`f(seed, system, at)`**, not `f(seed, system)` — even though today
only the orbital positions vary with `at`. That one argument is what keeps the door open for the worldline.

### WHERE THE REST OF IT LIVES

`scripts/dormant_world_simulation_design.md` — "THE WORLDLINE: dormant-world simulation as a main-game
substrate" — is the existing design for what advances while a realm sleeps (NPC life, materials,
construction, the economy's place-bound stocks). **S9 does not implement it.** S9's whole obligation is to
not paint it into a corner, which is one function argument.

⚠ **AND THAT DOCUMENT ALREADY NAMED THE HAZARD S9 IS FIXING**, which is worth recording as a convergence
rather than a coincidence: it observes that `to_realm_id` collapses Universe and Galaxy onto fixed
singletons, that *"every galaxy is currently the same realm"*, that both stand-ins collide with any star
system whose seed is 0 or 1, and that *"a later slice will change those identifiers, orphaning anything
keyed by them."* S9 is that slice. Anything the worldline keys by a realm identity must land AFTER it.

## Q6 — where does the shaped placement land? **S12, NOT S9** (owner agreed, 2026-08-26)

I had put it in S9 on my own reasoning — *"moving the radii without fixing the shape would ship a bigger
hollow shell"* — and then argued against myself: a bigger hollow shell with THREE stars is never seen by a
player. The shape only becomes visible when the census is raised.

**RULED: S12**, which is where the plan always had it and where the gates that exercise it live — the
per-pair minimum gap with its one-light-year floor, the replacement for the all-pairs separation fence, and
the census raise to 150,000.

**S9's subject is now one thing:** the galaxy and the universe become real places at their true sizes.

The design work is not lost. `scratchpad/s9_placement.md` holds it, and the owner's shape rulings (a real
galaxy shape drawn from the seed; never evenly spaced; seed-permanent placement) are recorded in the
2026-08-24 addendum and bind whenever it lands. ⚠ That document's headline numbers — "the world caps at
12.57 systems", "28× inside the floor" — are DISPROVED and must not be built against; see the handoff §6.
