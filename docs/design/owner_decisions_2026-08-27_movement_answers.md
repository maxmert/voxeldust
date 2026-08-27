# Owner decisions — 2026-08-27 — the answers to the movement questions

★ THIS FILE IS NEWER THAN `owner_decisions_2026-08-26_movement.md`. Where the two disagree, this file
wins. It closes three items that the movement contract left open.

Two items from the same set are still open. They are listed at the end.

---

## M-A — THE UNGOVERNED ROCK: the containing realm owns its speed, and a transfer never changes it

**The question.** The movement contract deletes the approach governor. A ship states its own forces,
so a ship governs itself. A rock states nothing. A rock has no engine and no pilot. Does a realm's
speed ceiling still govern a rock?

**The ruling.** Yes. The realm that holds a thing owns the speed of that thing when the thing governs
nothing itself. A rock, a piece of debris and a fired shell are all such things.

**THE HARD PART, and this is the new law:** a transfer must NEVER change the speed.

A thing that crosses from one realm into another keeps the speed it had. The destination realm must
not pull that speed down to its own ceiling. The destination realm must not push that speed up either.

**Why.** A re-clamp on crossing makes a thing jump. A player sees the jump. SL8 calls a seam a defect,
and this jump is a seam. The jump is also a lie about the world: nothing touched the rock, so nothing
may change how fast it moves.

**An example.**

```text
  A rock flies through the star system at 900 m/s.
  The star system's ceiling is 1000 m/s.        -> lawful, nothing happens.

  The rock crosses into a planet's realm.
  The planet's ceiling is 400 m/s.

  WRONG (a re-clamp):  the rock's speed becomes 400 m/s at the boundary.
                       It appears to hit an invisible wall. Nothing hit it.

  RIGHT (this ruling): the rock still flies at 900 m/s.
                       The planet's ceiling governs what the planet may DO to the rock,
                       never what the rock ALREADY IS.
```

**What a ceiling therefore means.** A ceiling limits what a realm may ADD. It never rewrites what a
thing arrives with. This matches the movement contract: only the parent writes a placement, and a
placement it writes must follow from what the thing was already doing.

### ★ TODAY'S CODE BREAKS THIS RULING. Here is the exact line.

Measured on 2026-08-27, by reading the path end to end.

The CROSSING itself is clean. `transfer_frame` (`crates/core/src/frame.rs:192-292`) only changes the
frame of reference. It subtracts the destination's own velocity. It holds no ceiling and no clamp. It
cannot hold one, because SL4 forbids a motion symbol on the crossing path.

The re-clamp happens ONE TICK LATER:

```text
  crates/sim/src/stub/transient.rs:248-258

     if let Some(v_allowed) = governed_ceiling_for_frame(..., t.pose.frame, ...) {
         let speed = t.pose.vel.length();
         if speed > v_allowed { t.pose.vel *= v_allowed / speed; }
     }
```

`t.pose.frame` is already the DESTINATION realm's frame, because `place_arriving_pose` re-wrote it. So
the clamp resolves the DESTINATION realm's ceiling and cuts the rock down to it.

**So the answer is yes: today the speed IS changed after a transfer.** It is late by one tick, and it
follows a freeze of several ticks while the item is `Arriving`. The first step after that freeze is one
clamped velocity multiplied by the whole accumulated time.

**This must change.** The clamp must not pull down a speed that a thing arrived with. What the clamp
may still do is limit what the realm ADDS from that point on.

**A second finding, stated honestly.** No test in the tree measures a transient crossing a realm
boundary and being re-clamped. The only test is
`a_transient_faster_than_the_governed_ceiling_is_clamped_on_the_crossing_path`
(`crates/sim/src/stub/tests/movement.rs:279-328`), and despite its name it never crosses a realm — it
plants a held rock in one realm and clamps it there. **The re-clamp is a reading of the code, not a
measurement that could have failed.** The fix must land with the test that was missing.

**A third finding.** When the destination shard has no region forest or no authored book yet,
`governed_ceiling_for_frame` answers nothing and there is NO clamp at all
(`crates/sim/src/stub/transient.rs:246-247`). So the law is absent exactly when a realm is starting —
which is the moment a crossing arrives.

---

## M-D — THE CEILING LEAVES THE FLIGHT PATH, AND CONTAINMENT BECOMES SWEPT

Owner-agreed 2026-08-27, after a correction. This section supersedes any earlier wording that said a
ceiling should "refuse". It says where a refusal may live, and where it may not.

### THE MISTAKE THAT THIS CORRECTS

An earlier draft said: *a ceiling must refuse loudly, never adjust silently.* The owner asked the right
question — **what does a refusal look like in the game? Does movement stop?**

It would. That is worse than the silent clamp it was meant to replace. A wall made of nothing is the
weakest possible answer.

**A PLAYER NEVER ASKS FOR A SPEED.** A player presses a throttle. The engines make a force. The force
makes acceleration. Nobody states a number. So on the flight path there is nothing to refuse, and no
refusal can be written there.

### THE RULING

1. **THE CEILING LEAVES THE FLIGHT PATH COMPLETELY.** No clamp. No refusal. No stop. Not on arrival,
   not on a transfer, not near a body.
2. **A FENCE MAY STAND ONLY WHERE CODE STATES A SPEED DIRECTLY** — a spawn, a fixture, a tool. There a
   bad number is a DEFECT, and a test must go red. A player never reaches this path.
3. **CONTAINMENT IS ALREADY SWEPT — CORRECTED 2026-08-27, AFTER MEASURING.** The draft above proposed
   building a swept test. **Slice S5 already built it.** `region_verdict`
   (`crates/core/src/geometry.rs:1496`) takes the current pose AND `prev_pos`, and its own heading says
   *"The verdict tests the tick's MOTION, not the instant (slice S5)"*. `Shell` and `Aabb` are swept on
   exact integer arithmetic. The point answer runs first, so the sweep can only turn `false` into
   `true`. **`Obb` is the one shape NOT swept** — it falls back to the point test
   (`geometry.rs:1521-1523`).

   ★ **SO DETECTION WAS NEVER THE REASON THE BANDS ARE WIDE.** The real constraint is TIME. A re-home
   is a saga and it needs ticks. `BoundaryTuning::DEFAULT` sets `k_dwell = 5` ticks
   (`crates/core/src/geometry.rs:938-943`), and the measured thinnest governed crossing in the world is
   **5.59 ticks** — barely above it. At the parent's speed the crossing takes far less than one tick.
   The swept test still SEES it. The machinery cannot HAND YOU OVER in time.

   **The cure is therefore not a wider band. It is an earlier start.** The parent authors velocity, and
   the test already reads the line travelled. So the system can see a crossing coming and begin the
   hand-over BEFORE the boundary is reached. That is the same shape as M-B: warm ahead, never slow down.
   With the hand-over started early, the band buys only hysteresis and can stay small.
4. **SLOWING DOWN IS GAMEPLAY.** A ship slows because its own safety block slows it, and a player may
   switch that block off. A rock slows because air slows it. Both take time, and both are visible.

### WHY THE BAND WAS EVER WIDE

The band never protected the player. It protected a SNAPSHOT.

```text
   tick 1                              tick 2
     o------------------------------------>o
     |            |         |              |
   outside      band      inside        outside again

   The test sees tick 1: outside.
   The test sees tick 2: outside.
   It never sees you inside. You never change realm.
```

Sizing a band by speed is a workaround for a sampling problem. That is why deleting the governor broke
the bands: the two were one mechanism wearing two names.

A swept test reads the line instead, so the band keeps only two jobs — stay wider than the coordinate
step, and give hysteresis. Neither job scales with speed.

### THE MEASUREMENT THAT SETTLES IT

The galaxy's ceiling today is **5.12 x 10^16 m/s**. The speed of light is 3.00 x 10^8 m/s. So the
ceiling sits about **170 million times the speed of light**.

In open space it can never bind. It begins to bind only near SMALL bodies — a station, a moon, a
player-built structure. That is exactly where the owner noticed it feeling wrong.

**So the ceiling was never a gameplay device. It was a sampling patch.**

### AT EXTREME SPEED, THE WORK IS BOUNDED — NOT STOPPED

1. The swept test walks the line travelled this tick.
2. If the line crosses several realms, they resolve in order, nearest first.
3. If the line is very long, the test takes a capped number of sub-steps along it.

The only true limit left is what the coordinates can represent. **That number is NOT measured yet, and
must not be quoted until it is.**

### CONSEQUENCE FOR THE BAND RE-SOLVE

**The re-solve does not run yet.** Re-solving today against the parent's ceiling was measured and gives
bands thousands of times larger than the bodies they wrap.

What must land first is **the early hand-over**, not a swept test — the sweep is already there. Once a
crossing starts before the boundary is reached, the band stops having to buy dwell time, and it shrinks
on its own. The `Obb` sweep gap should close with it.

One defect is separate and needs no ruling: the band clamp runs after the band floor and overrides it
(`crates/physics/src/worldgen/config.rs:267`). A station gets 2.5 m and an area 1.5 m against a stated
3 m floor, and a player-built 3 m cube cannot boot at all.

### THE SMALLEST HONEST FIRST STEP

Measure the dwell the transfer saga actually needs, on the real world, at the speeds the movement law
now allows. Today's `k_dwell = 5` was chosen against a governed approach that no longer exists. Until
that number is re-measured on the ungoverned world, any band it produces is sized for a world we do not
run.

---

## M-B — WAKING UP IN TIME: grow the interest radius, never cap the speed

**The question.** A realm needs time to start. A realm that does not run cannot be drawn. At a high
closing speed a player reaches a realm before that realm is ready, and arrives at a dark place.

**The ruling.** Grow the interest radius with the closing speed. **Do not cap the speed.**

**Why not a cap.** A speed cap makes the game unfair. Two players do the same thing and get different
results, because one of them happened to approach a realm that starts slowly. The player pays for a
server's start-up cost. That is not acceptable.

**How the radius answers it instead.** The parent already holds every child's velocity, because the
parent authors it. So the parent can measure how fast an observer closes on a child. It then asks for
that child to start earlier, in proportion to that speed.

**An example.**

```text
  A realm needs 2 seconds to start.

  A player closes on a star system at 50 m/s.
    interest radius = the still radius + 50 x 2   =  the still radius + 100 m

  The same player warps toward it at 200,000 m/s.
    interest radius = the still radius + 200,000 x 2 = the still radius + 400 km

  Both players arrive at a realm that is already running.
  Neither player was slowed down.
```

**The cost is paid by the server, not by the player.** A faster approach warms more realms earlier.
That is the correct place for the cost to land.

---

## M-C — THE INTERIM ENGINE RATING, AND THE TEMPORARY CONTROL SEAM

**The question.** The physics phase is not built. Until it lands, may a ship state its own rated cruise
speed and acceleration as facts about itself?

**The ruling.** Yes.

A rated speed and a rated acceleration are facts about what a ship IS. The movement contract already
allows a ship to state what it is: it sends its mass, its cross-section and its drag coefficient on
change. A rating joins that list. It is not a request, and it is not a placement.

### THE TEMPORARY SEAM (owner-stated, and it is the point of this decision)

We do not have a character to control. We do not have signals. We do not have functional blocks. We
still want to test the whole movement path now.

So we attach the controls to the SHIP REALM directly.

```text
   player input
        |
        v
   +--------------------------+
   |  the ship's own shard    |   the controls make FORCES here, directly.
   |                          |   no signal bus. no thruster block. no hull.
   +--------------------------+
        |
        |  acceleration + torque, per tick, in the ship's own frame
        |  mass + cross-section + drag coefficient, on change only
        v
   +--------------------------+
   |  the parent realm        |   the parent does the physics.
   |  (a star system, say)    |   the parent authors the placement and ships it down.
   +--------------------------+
```

**What this proves.** It exercises the real machinery, end to end:

- a child states forces in its own frame;
- the parent applies them;
- the parent authors the placement;
- the parent ships the placement down, stamped and read-only;
- a crossing hands the ship to another parent.

**What this does NOT do.** It does not invent a second movement path. The forces travel on the lane
the movement contract already defines. Nothing here is a special case for a ship.

**What replaces it later.** The signal system and functional blocks. A thruster block will then publish
the force, and the seam disappears. The ledger must carry this as an interim entry, with the phase that
removes it named.

---

## M-E — THE ORDER OF WORK: fly a ship in a window BEFORE the engine upgrade

Owner-stated 2026-08-27, after being offered the Bevy upgrade and shown its blast radius.

**THE ORDER.** Finish S11. Then S12. Then reach the milestone where the game is **window-tested with a
ship realm actually flown** — which is what M-C's temporary control seam exists to make possible. Only
then upgrade Bevy 0.18 → 0.19.

**WHY.** `bevy_ecs` is the ECS of every shard: sim, node, connection-plane and harness all depend on
it, and the renderer adds `bevy` plus `bevy_egui` (pinned at 0.39 for Bevy 0.18). So the upgrade
rewrites system signatures across the whole server, not just the renderer.

Mixed into feature work, a red suite cannot say which change broke it. The suite is about 2,300 tests,
and it is only useful while a failure points somewhere.

**THE ONE THING THAT REVERSES THIS.** If Bevy 0.19 ships something that makes the starfield materially
better or simpler — a real instancing API, a sprite pipeline — then building the star sprite layout on
0.18 means building it twice. That must be checked BEFORE the vertex layout is written, and said aloud
if true.

---

## STILL OPEN after this file

- **The band re-solve.** Every band in the world was sized on the deleted governor. When does the
  re-solve run, and against what number? The owner has asked for an explanation before ruling.
- **The second shard kind for HR4.** The sky lane's tests run on one rig. The owner has asked for an
  explanation before ruling.
- **The wake-up radius formula itself.** M-B fixes the SHAPE of the answer (grow the radius). The
  constant — how much start-up time to assume — is not ruled here, and must be derived, never guessed.
