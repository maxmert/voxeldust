# Owner decisions — 2026-09-05 — THE SUIT: a character wears its way to move in space

★ **LATER THAN EVERY DESIGN DOC AND EVERY EARLIER RULING, including 2026-09-02.** Where a plan
disagrees with this file, this file wins. Read it before you touch the walk law, the flight helpers,
or anything that gives a character a speed in space.

The owner ruled this in one exchange on 2026-09-05, after the process gates measured the walk law's
time constant as the server's boot time.

---

## S1. THE MEASUREMENT THAT RAISED IT

The planet chase in the demand-login gate, release build: the walker's speed grew by a fixed factor
per tick with a time constant of about 12 seconds (0 → 1.1 km/s → 6.8 km/s over 5,000 ticks). The
constant is DERIVED from the wake horizon — two area-of-interest beats plus the boot bound plus the
pipeline, in ticks — so a player in a spacesuit may not accelerate faster than a shard can boot. In
a debug build (a 3,500-tick boot bound) the constant is 72 seconds and the walker crawls at 4 km/s.
The hull escaped this because its drive states a rating; the walker never had one.

This is the cap the 2026-08-27 ruling struck out: *"a cap makes the game unfair, because the player
would pay for a server's start-up cost."*

## S2. THE RULING

**Owner:** *"You are right, character will wear the suit, and there can be different ones. Without
the suit character will not be able to move in space. So suit should be similar to the hull."*

- A CHARACTER WEARS A SUIT. The suit is a thing, like a hull: it is worn, there are different ones,
  and it states what it IS.
- WITHOUT A SUIT A CHARACTER CANNOT MOVE IN SPACE. Movement in space is the suit's, never the
  character's. (A character on a floor — inside a hull, a station — walks on foot; that is not
  space movement and is not this ruling's subject.)
- THE SUIT STATES ITS RATING as facts about what it is — an acceleration and a cruise speed, the
  same shape as the hull's engine rating — per entity, from data, never a constant tied to the
  server. The walk's ramp comes from the suit's rating. The wake radius grows with the closing
  speed, as already ruled; the server's boot time never sets a player's acceleration.
- THE SUIT IS SIMILAR TO THE HULL: it is a worn built thing. Until the block system builds suits
  from parts, a suit's rating is a data record the character carries (a suit KIND with its
  facts), exactly as the hull's rating is a fact record today (`BuiltFacts` → `EngineRating`).

## S3. WHAT THIS DELETES

- The wake horizon as the walk's time constant (`FlightTuning::tau_s` on the walk's ramp).
- The approach ceiling on the player's stick. A fence may stand only where CODE states a speed
  (the dev tool's `walk_to`, a spawn, a fixture) — never on the stick (2026-08-27).

## S4. WHAT THIS BINDS FOR THE TESTS

The flight helpers walk a dot at warp speed. A real player boards a hull and pushes, or wears a
suit and pushes. The gates fly the shipped path: berth a hull, board it, push — or wear a suit whose
rating states the speed the leg needs. *Test exactly production* (standing).

## S5. OPEN

- The suit's data source before the block system: a suit kind table in the built store (like a
  berth record), granted at login. Which suit a new character wears is a game decision, not a
  constant.
- On-foot movement on a floor (inside a hull) keeps the foot speed; where the floor ends and space
  begins is the containment question P6 answers.

## S6. THE ORDER (owner, the same evening)

**Owner:** *"I'd suggest that we do suit later, only after a character. But before character I want
to do blocks and terrain — otherwise we can't test character at all."* and *"Let's finalize all
fundament, and then we can switch to terrain."*

- The order is: the remaining FOUNDATION → BLOCKS AND TERRAIN → THE CHARACTER → THE SUIT.
- Until the suit lands, the dot's walk keeps its boot-time ramp as a PLACEHOLDER for a character
  that does not exist yet. This is wrong by S3 and deferred by the owner on 2026-09-05; the ledger
  carries it under D-MOVE-3.
- The gates flying hulls (S4) is test infrastructure and does not wait for the suit.
