# OWNER DECISIONS — 2026-08-27: what a seed may decide, and what it may never decide

**LATER THAN EVERY DESIGN DOC. Where a plan disagrees with this, this wins.**
**Written BEFORE the block system starts, on the owner's instruction, so the terrain work is built
against it rather than retrofitted to it.**

---

## S1. THE RULE

> **Anything a fixed seed alone determines must be SAFE TO PUBLISH.**
> **Anything VALUABLE must depend on world state that CHANGES.**

The owner asked the question that produced this: *"How dangerous will it be to pass generation function
+ seed to the client? I believe then players would be able to understand where to search resources
without using any of the game mechanics?"*

The danger is real. The cure is not secrecy.

---

## S2. WHY SECRECY IS NOT THE CURE — two facts, neither of them about cryptography

**ONE WORLD, ONE SEED, MANY PLAYERS.** The first player to find the rich system posts it. Within a week
the good locations are on a wiki. **A shared static map becomes public knowledge through ordinary play;**
the seed only gets there faster. A design whose fairness rests on the seed staying secret is already
broken and has not noticed.

**A SEED CAN BE RECOVERED.** The generator ships inside the game the player already runs, and enough
observed world inverts to the seed. This is routine in games built this way, not theoretical.

So hiding the seed buys TIME, not secrecy. Time is worth something — it is not worth designing around.

---

## S3. THE GENERATOR IS NOT ONE THING — split it where the risk splits

| Half | Decides | May the client compute it? | Why |
|---|---|---|---|
| **GEOMETRY** | star positions, orbits, sizes, brightness | **Safe in principle** | A player who computes it learns what a telescope would tell them. The owner already ruled the star map is REAL and shows the true reachable systems — it is public by design |
| **COMPOSITION** | what a place is MADE OF, what is worth taking | **NEVER** | Computing it offline replaces the survey, which is the game mechanic that was supposed to earn it |
| **LIVE STATE** | depletion, ownership, what is built, who is there | **NEVER, and it is not seed-derived at all** | No generator can predict it |

---

## S4. WHY THE TIMING IS FREE

**MEASURED 2026-08-27: the dangerous half does not exist yet.** The generator today produces positions,
orbits, extents, star mass/class/luminosity and a planet taxonomy row. It produces **no resources, no
ore, no deposits — nothing that pays.**

So this is decided before the valuable half is written, while the answer still costs nothing. After the
terrain work it would be a retrofit.

---

## S5. WHAT THIS BINDS THE BLOCK SYSTEM TO

1. **A block's SUBSTANCE may not be a pure function of (position, seed).** If it is, offline computation
   replaces surveying and the mechanic is dead on arrival.
2. **Anything that pays must read world state that moves** — depletion, what NPC life consumed, what
   players built and took. The dormant-world pillar and the player-driven economy already require the
   world to advance while nobody is there; this makes that requirement load-bearing rather than
   decorative.
3. **The survey is the mechanic.** What is in the ground is revealed by looking, and looking is a thing
   the player does in the world.
4. **A seed-derived fact must pass this test before it ships:** *if this were printed on a public wiki
   tomorrow, would the game still work?* If no, it may not be seed-derived alone.

---

## S6. ⚠ WHAT THIS DOES **NOT** DECIDE — the client-side derivation question is OPEN

S3 says the geometry half is safe *in principle* to compute client-side. **It does NOT authorise the
client to compute it**, because that collides with a standing law the owner stated emphatically:

> The client ONLY renders. The SERVER composes every entity's ready-to-draw position and ships it.

That law exists because of a real defect: the floating-origin work drifted into client-side composition,
*looked equivalent at small scale*, and had to be reworked server-side.

**The assistant's own argument for an exception here resembles the reasoning that went wrong then** —
"it is deterministic, both sides get the same answer, so it is safe" — and that is flagged rather than
relied on. The one genuine difference is that a star's position is STATIC and depends on no live
simulation state, where an occupant's pose does; and S9's world-generation tag already refuses a client
whose generator disagrees. Whether that difference is enough is the OWNER'S call and is not made here.

**SETTLED 2026-08-27, the same walk: NO client-side derivation.** The owner ruled the other way —
*"we're passing the Galaxy just once over reliable lane"*. The server keeps composing; the saving comes
from sending a thing that never changes ONCE instead of twenty times a second. The client-only-renders
law stands untouched, and the exception argued above is NOT taken.
