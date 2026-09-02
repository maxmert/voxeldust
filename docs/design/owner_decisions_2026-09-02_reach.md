# Owner decisions — 2026-09-02 — REACH: one radius per realm, tested by its parent

★ **LATER THAN EVERY DESIGN DOC AND EVERY EARLIER RULING, including 2026-09-01.** Where a plan
disagrees with this file, this file wins. This file makes the 2026-09-01 ruling (visibility is a
radius, not a tree walk) concrete: what the radius is, who tests it, what data travels, and what is
deleted. Read it before you touch anything on the visibility, wake or picture path.

The owner walked the design in one session on 2026-09-02, corrected it twice, tested it against three
cases, asked for the industry comparison, and then said: *"Ok, then implement this design please."*

---

## R1. THE GALAXY IS ALWAYS VISIBLE, AND IT IS NEVER RE-ANCHORED

**Owner:** *"Galaxy is always visible, we just don't repopulate it, as it doesn't move (we can move when
we are inside the galaxy realm, so all stars will move in parallax, but again the same rendering
machinery should be used)."*

- The star field is the GALAXY REALM'S OWN LOOK. It is shipped once. It never changes.
- The client builds ONE star cloud in the galaxy's frame and never rebuilds it.
- The galaxy is a row in every observer's picture, always. The client places the cloud at the galaxy's
  composed placement, as it places any realm's drawing at that realm's row.
- Parallax is the ordinary camera moving inside one cloud. Nothing draws it on purpose.
- **THE RE-ANCHOR IS DELETED.** Today the client keys the cloud on the observer's own star system,
  deletes the cloud on every crossing, and rebuilds it by searching the catalogue for the observer's own
  realm. The catalogue names star systems only, so the search fails for a hull, a planet, a station, an
  area and the galaxy, and the sky goes black for good (`crates/client-render/src/lib.rs:1188-1198`,
  `crates/client/src/render_snapshot.rs:98-116`). MEASURED 2026-09-01 as the black sky inside a hull.
  **This was never a ship defect. It is a star-system-versus-everything-else defect.**

**Example.** A pilot warps a hull out of the home system. The galaxy's composed placement moves a little
each tick. The one cloud slides. Near stars slide more than far stars. That is the parallax.

---

## R2. A PARENT POSITIONS ITS CHILDREN. IT NEVER DRAWS THEM.

**Owner:** *"Star System realm should not render dots when something is not visible. It knows placements
of it's children, as any other realm, but it should not render them. They render themselves. Parent only
positions."*

- **THE PARENT-AUTHORED MARKER IS DELETED.** Today a parent ships one point-of-light marker per direct
  child on every window, and the gateway falls back to it when the child is silent
  (`crates/sim/src/stub/window.rs:441-483`, `crates/connection-plane/src/window.rs:984-1016`). Under
  this ruling a realm in range runs and draws itself; a realm out of range is not drawn by anybody.
- The ONE exception is R1: the galaxy's star field. It is lawful because it is static, seed-derived and
  safe to publish, and because a star system's own body, when it boots inside its own reach, draws on
  top of its own point of light at the same brightness. No handover message exists, so no pop exists.
- **Deletion order is binding:** the markers go LAST, after the reach test wakes realms and puts them in
  the picture. Delete the markers first and every realm the radius does not yet wake goes dark at once.

---

## R3. THE REACH IS BIGGER THAN THE BOUND, AND BIGGER THAN THE PARENT'S BOUND

**Owner, correcting the assistant's nesting claim:** *"the visibility radius can and should be bigger
then the realm boundaries, and most likely the parent realm boundary in most of the cases. So the
previous case I've described - if i'm not in the planet realm, planet realm will never check if I can see
the station, so it will remain invisible."*

The assistant had proposed that an enclosed realm clamp its radius to its container, and that a child's
visibility sphere always nests inside its parent's bound. **Both are refused.** The Moon is a dot from
5.8 million km; Earth's gravitational shell is 1.5 million km. A station's radius reaches far outside its
planet's realm.

**THE RULE, restated so it survives that case:** every realm states ONE number, its REACH, and the reach
covers the realm AND everything it holds:

```text
   reach(realm) = the larger of:
       the realm's own look radius  (size and brightness against the dot angle)
       max over children ( child's distance from my centre + reach(child) )
```

This nests BY DEFINITION, whatever the bounds are. A child's whole visibility sphere sits inside its
parent's reach. So an observer who can see a realm is always inside that realm's parent's reach, and the
parent is the party that tests.

**A realm's reach comes from its own LOOK, not only its size.** A planet is visible by brightness far
beyond the distance at which its size makes a dot. So a bright body reaches further than a dim body of
the same size. The realm states its own reach, because how it looks is its own statement (SL3).

---

## R4. THE DECISION IS TREE-INDEPENDENT. THE CONSEQUENCE WALKS THE TREE.

**Owner:** *"There is a radius in each realm, doesn't matter where they are in the forest, observant might
cross them and the realm should be waken and should end up in the window for the gateway. ... We need to
have tree and hierarchy for proper placements, proper separation of calculations of the physics, but it
has nothing to do with visibility and wake up / kill."*

- **THE DECISION** is one sentence: an observer inside a realm's reach sees that realm. Depth, parent,
  and who holds whom play no part in the answer.
- **THE CONSEQUENCE** walks the tree, and that is SL1, not visibility: a visible realm's placement has
  exactly one writer, its parent. So every ancestor of a visible realm must run to author the placements
  the drawing needs. The parent is also the lawful sender of demand (owner 2026-08-15, item 11).
- **WHO TESTS.** The parent, for each of its direct children, because the parent is the only party that
  holds both placements. This answers the question the 2026-09-01 ruling left open. The composer in the
  gateway still evaluates no position.
- **NO REALM WALKS THE TREE PER TICK.** Each running realm tests its own direct children against its own
  lookers, one level. A realm out of range does not run and tests nothing. One observer costs about six
  tests per tick, one per realm on its chain, whatever the size of the world.

**Example (owner's test case, walked and accepted).** A pilot flies a hull in a city on a planet. The
planet tests the station's reach against the city's placement; the station is overhead and in range, so
the planet demands it, it runs, it ships its look one hop up, the planet forwards it. The star system
tests the neighbour planet's reach against our planet's placement; in range, so it runs and is forwarded.
The galaxy is an ancestor and always in the picture. Nothing tested depth.

---

## R5. THE LOOKERS A PARENT TESTS AGAINST

1. **Its own occupants.** Exact.
2. **A child that holds somebody, or that somebody watches through an open window.** The parent uses the
   child's placement as the looker's position and the child's size as the error. This is the SL7 proxy.
   An open `Child(c)` window IS a statement that somebody looks out from inside c, made by the gateway,
   which is the party that knows. (The uncommitted 2026-09-01 change that adds the window-derived looker
   is APPROVED by this ruling.)
3. **A looker outside this realm, known by distance only.** The parent above sent the distance (the
   2026-08-26 M7 signal). Direction is unknown, so the test is conservative: it over-wakes, never
   under-wakes.

---

## R6. WHAT DATA TRAVELS — the SL6 ask, answered

**NEW, APPROVED BY THIS RULING:** a child states its REACH to its parent, ON CHANGE ONLY, on the slow
lane beside mass. One number. The parent cannot compute it, because a child's reach folds the reaches of
the child's own children, which the parent may not see (HR1). Without it the parent cannot know that a
moon reaches past its planet's shell.

| Direction | Data | Rate | Status |
|---|---|---|---|
| up, child → parent | my reach | on change | **NEW — approved here** |
| up | somebody is inside me | on change | exists; its ONLY job is liveness (V5) |
| up | mass, cross-section, drag | on change | exists |
| up | acceleration, torque | per tick | exists |
| up | my own look, one hop | on change | exists |
| down, parent → child | your placement, stamped | per tick | exists |
| down | nearest outside looker is this far | on change | exists (M7) |
| parent → orchestrator | run / keep this child | on change | exists |
| realm → gateway | the hop; the placements of my children IN RANGE; my own look; a child's look and interior forwarded verbatim | per tick / on change | **CHANGED: in range only** |
| gateway → client | the picture; the star field once | | **CHANGED: no re-anchor** |

**DELETED:** the parent's marker per child; the occupancy bit as a gate on sight; the whole roster on
every window (the galaxy's window carries the rows in range, not 233,220); the sky re-anchor; the
generator rule that grows a realm's shell until no descendant is visible from outside it (2026-08-15
item 5 — at a dot angle that rule refuses the world, and the reach makes it unnecessary).

**DEFERRED, a second ask if the over-wake measures badly:** when the looker is a hull (a realm, not a
person), the parent may state the hull's placement to the child instead of a distance. Not asked here.

---

## R7. THE INDUSTRY MODEL, AND WHY THE LAWS FORCE THE SHAPE THAT SCALES

The owner asked for the standard. The design is the union of three named things: the aura–nimbus model
(each object states how far it is perceivable), a bounding-sphere hierarchy (a parent's sphere covers its
children's), and nested container streaming (each container has a streaming radius, contents stream by
their own). What is unusual is the lawful distribution: each process holds one level, and no global
picture is assembled anywhere. A fixed grid fails across twenty orders of magnitude; a global index needs
absolutes (SL1) and a central assembly (SL7); precomputed visibility needs static geometry; a per-observer
pose query leaks the pose (SL2). The laws leave exactly this shape.

---

## R8. WHAT MUST BE MEASURED, NOT ARGUED

1. **The galaxy with 233,220 star systems and one looker.** Tick time must not grow with the census. The
   child test must be an index, not a walk (today `crates/sim/src/stub/aoi.rs:602` walks). Star systems
   never move, so the index is built once. Moving children re-insert on change.
2. **The home system with a hundred thousand lookers.** The gateway already folds once per occupied realm
   and shares it; the index query count is the cost.
3. **A chain six realms deep.** Compose time per hop.
4. **The bright-body count.** How many shards run per occupied star system once a planet's reach comes
   from its brightness. This is the price of R2.
5. **Precision at galaxy scale** for the one cloud placed at the galaxy's composed placement.

**Known trades, accepted:** a reach sphere ignores occlusion, so a station behind its planet may run
unseen (over-draws, never under-draws); a walled room's reach passes through its walls (a realm may state
a smaller reach than its size implies, because that is a statement about how it looks — the realm's
choice, never a parent's clamp); hysteresis at the reach edge stays (the grace latch); a fast looker
grows the reach by closing speed × boot time (M-B).

---

## R9. THE ORDER OF WORK (binding, because the wrong order blacks out the world)

**FLOWN BY THE OWNER 2026-09-02, IN THE WINDOW: "Worked!"** — a demand cluster, a hull berthed forty
metres from the spawn by the shipyard's stand-in, the owner's window as the pilot, a headless watcher
beside it; the pilot crossed into the hull, switched to the chase view, and saw the stars and the star
system's own children from inside a player-built realm. The watcher at the spawn drew the same 233,220
stars with the galaxy's disc banded across its frame.

**STATUS 2026-09-02, end of the build session (every claim measured, see DEFERRED.md D-REACH-1):**
steps 1–3 and decisions 2–3 LANDED — the hull subject of `world_from_inside` is GREEN on the cluster
(a player crosses into a built hull and sees the stars and the star system's children). Step 4 (the
reach datum, the dot angle), step 5 (the markers) and step 6 (the galaxy's tick: 61 ms mean against
20 ms after the lookup fold; 1.9 s before it) are OWED. Two things the build found that no reading
had: the hull's window frame was 1,420 bytes against a 1,200-byte datagram budget and EVERY frame was
dropped by the transport; and the galaxy shipped its whole roster on every keep-alive. Both are fixed.

1. The red test: a player crosses into a built hull AND into the home planet; the same body asserts stars
   are DRAWN (a new instrument — today the client reports stars HELD) and the picture names a realm the
   subject does not parent.
2. The chain is complete to the root, always, for every entry path.
3. The galaxy is a row in every picture; the star field is its look; the re-anchor dies.
4. Reach: stated by each realm from its look, sent up on change, tested by the parent; the dot angle
   re-solved; windows ship children in range.
5. The markers and the bit-as-sight-gate are deleted — only after step 4 is green.
6. The index on the galaxy, with the SL9 measurement.

---

## R10. TWO DECISIONS TAKEN DURING THE BUILD (owner: "Sounds good!", 2026-09-02, later the same day)

The assistant stopped mid-build and put two architectural decisions to the owner in full. Both were
approved. They are recorded here because R6's table did not name the first, and R9's order did not
place the second.

**DECISION 2 — THE SKY ANCHOR (approved).** The gateway sends the client ONE new field on the per-tick
realm datagram: the observer's origin realm placed in the galaxy's frame, in the galaxy's own step —
"the galaxy's centre is HERE, relative to you". The gateway computes it by lifting the origin's zero UP
the observer chain through the same books it already walks down to compose the picture. The client
places its one star cloud by it each frame and computes no position. About 130 bytes per tick per
session. It crosses the gateway→client lane, not a realm boundary; no realm learns where it is. This is
what makes R1 buildable: without it the client had to guess its anchor by searching the catalogue for
its own realm, which fails inside every realm that is not a star system.

**DECISION 3 — A WINDOW SHIPS ONLY THE CHILDREN IN RANGE (approved, as the first part of R6).** A
realm answers a window with rows and markers for the children in its range, plus the hop child. A child
in range runs and draws itself, so the gateway needs its placement; a child out of range sleeps and
needs no row. Moving children ship as today until reach gives them a brightness radius (R9 step 4). A
realm with no live range ships everything, as today — without a band nothing is out of range.

THE MEASUREMENT THAT FORCED IT NOW: the galaxy shard shipped 233,220 rows and 233,220 markers on every
window keep-alive, forever, because the gateway could never confirm 22 MB; the link shed frames for
minutes and the hop the sky anchor needs was among them. After the rule the galaxy's window carries one
row, the home system's, and the star field shows every sleeping star in full — *"we should render it
all, we should not render stars based on their proximity to us"* is the star field's job, not the
window's. THE ONE COST, accepted: a far STATIC realm (a station across the system) loses its dot until
reach gives it a bright point.

The rule in one line, for every realm: **a parent ships a row for a child that runs, and no row for a
child that sleeps.**
