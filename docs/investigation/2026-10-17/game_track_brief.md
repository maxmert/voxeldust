# The game's parallel track — the brief for its worktree and session (ruling B105 item 4, 2026-10-17)

> **Read first:** `CLAUDE.md` (the hard rules HR1–HR6 and the standing laws SL1–SL10), `docs/design/PLAN.md` (phases P5–P11
> and the blocking spikes), the newest ruling file `docs/design/owner_decisions_2026-09-22_believability.md` (B99, B104,
> B105) and `docs/design/DEFERRED.md` (D-19). Write every report, comment and commit message in ASD-STE100.

## 1. Why this track exists

For weeks the work went into the terrain. The game around it is thin: realms, transfers (proven under `kill -9`), the
gateway, the client's picture, walking, hulls with ratings, boarding and flight. Physics (rapier is not even declared in
the workspace), blocks (only their wire arms), functional blocks, signals, inventory and player state do not exist. On
2026-10-17 the owner opened this track (B105 item 4; B72 item 2): the game moves in its own worktree and session, while
the terrain worktree finishes the water, the clouds and the weather. The terrain will gain details there, but its world
will not change much.

## 2. What this track owns

Ruling B105 moved B99 steps 5–6 here. In the order the owner approved:

1. **SPIKE-6a — rapier's state saved and restored identically** (blocks P5; `DEFERRED.md` D-19). A crossing between
   realms carries rapier's state (the convention: Category C, checkpoint-carried, never re-simulated across hosts). Prove
   that a snapshot taken in one process and restored in another steps on byte for byte, on the shipped targets.
2. **SPIKE-10a — physics inside a flying ship** (blocks P8). A walker stands and walks in a hull's own frame while the
   hull accelerates and turns in its parent's frame (the two frames of PLAN.md P8).
3. **Physics on the terrain's solid cells** (P5; B99 step 6: standing, collision, delivery). Today a walker stands by a
   height rule (`crates/sim/src/stub/stand.rs` stands parked hulls on the live ground). The aim: the walker's capsule
   collides with the same ground every client draws, players collide with each other, and a crossing carries the
   physics state.
4. **The block record and the dig/place path** (P6; B99 step 5; the one-volume revision §4 with Q6–Q8 as ruled by B105):
   one cell record for the ground and for ships (V4: a hull may hold terrain); a dig or a placed block reaches every
   viewer by distance (Q7) as a one-hop change from the owning realm (SL10) and survives a crash; a built area keeps its
   own ground (Q6, the frozen patch of F1); caves stay solid until a dig breaks in (Q8), then the opening flood (the
   revision §3.3) opens the connected hollow — its bound sized first by G-CAVE-SIZE (the home planet's largest connected
   hollow of the cavern field and the tubes, `vd_terrain::carve`), a reading this track owns.
5. Then, on the owner's word: ships built from blocks (P8), functional blocks and signals (P9), player state and
   inventory (P7).

**The first vertical slice to aim at:** a player digs into a planet's ground, places blocks on a parked hull, walks inside
it while it flies, and lands on another planet — physics, blocks, ships and their storage together, on two kinds of
realm (HR4's G-IDENTICAL).

## 3. The contract with the terrain worktree

The terrain gives the game ONE contract: height, solid or empty, substance, water — the same bytes on the server and on
every client (SL10).

- **Today:** the 8 km artifact (`vd_terrain::artifact`), the 128 m and 16 m finished tiles (`vd_terrain::fine`; the read
  path `fine::read::FineField` and `height::height_field_m`), the solid kernel `s = r − h` (`vd_terrain::cell`), the
  sea's level. The 16 m ground is the finest shape (B104: the 4 m level is parked).
- **Coming from the terrain worktree (B99 step 4):** the substance of each cell by law (rock from its bed, soil depth,
  glacier ice), the water map in each tile, every lake filled at its level (B105 Q1 (b)), rivers at every level.
- **The rule:** this track reads the generator through its public functions and never changes its output. A change of
  the world (a generator, artifact or fine version) belongs to the terrain worktree; ask there, through the owner.

## 4. Who edits what (to keep the two worktrees from colliding)

| Area | Terrain worktree | Game worktree |
|---|---|---|
| `crates/terrain` (the generator), its bench examples, the terrain status file | edits | reads only |
| The client's ground drawing (`chunks`, `ladder_view`, `client-render` terrain, sky, water look) | edits | reads only |
| Physics (rapier, colliders, the character controller), stand and collision in `sim`/`node` | — | edits |
| The block record, edits, the frozen patch, persistence of edits | — | edits |
| Ships from blocks, functional blocks, signals, player state, the client's player and ship interiors | — | edits |
| `crates/wire` (new arms need SL6 asks) and shared seams in `sim` | coordinate through the owner | coordinate through the owner |

Both worktrees merge onto `main` by the owner's word; neither commits or pushes without it.

## 5. Working rules that bind here too

- **Tests (B74):** no suite, gate or coverage run without the owner's word, and each ask states why it is needed now.
  Since generator 27 the terrain crate's tests take about one hour and the bins crate's over three (the seasons' chain in
  the moon solves); run the tests of the crates this track changes.
- **The look decides (B76):** pictures and the owner's look in the window come first; headless readings beside them.
- **The machine:** one Mac (14 cores, 48 GB; B36 Q10) serves both worktrees; the terrain worktree runs long solves (a home
  planet about 45–60 minutes on 12 cores). Name your heavy runs before you start them.
- **The world is one (SL5):** no test-only variant of the world; a spike may build its own small fixture, never a second
  generator.
- **The registry:** every interim enters `DEFERRED.md` with WHAT, WHERE and WHEN.

**Example.** The first green result of this track: two processes, one rapier world each; the first steps a falling
capsule 100 ticks, snapshots it, and the second restores the snapshot and steps 100 more; both print the same 64-bit
digest of every body's pose on every tick after the restore.
