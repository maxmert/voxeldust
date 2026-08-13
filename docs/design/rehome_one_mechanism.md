# Re-home: one mechanism — the settled design (2026-08-12)

BINDING. Agreed with the owner in full, decision by decision, on 2026-08-12. The laws it rests on are
SL1–SL7 in `CLAUDE.md`; this document is how they are realised for re-home, and the ordered work.

---

## 1. The measured defect

Two live answers to "which realm holds this point", disagreeing whenever anything moves.

One point — a star's own centre. One tick. The shipped world generator, not a fixture:

| the five planets of the origin star system | the parent's downward check | the shard's own containment check |
|---|---|---|
| planet 1 | **−4.16 m — INSIDE** | +13.77 m — outside |
| planet 2 | **−4.16 m — INSIDE** | +26.84 m — outside |
| planet 3 | **−4.16 m — INSIDE** | +42.35 m — outside |
| planet 4 | **−4.16 m — INSIDE** | +79.08 m — outside |
| planet 5 | **−4.16 m — INSIDE** | +140.31 m — outside |

The gap is exactly each planet's live distance from its star. Same at tick 1 000 and 50 000.

Cause: an orbiting realm stores its region centre as ZERO (its live position comes from its parent each
tick). One answer re-expresses through the live placement; the other subtracts the stored centre — zero —
so **every orbiting planet reads as sitting exactly on its star**.

For anything that does NOT move the two answers are byte-identical (−150.0 vs −150.0;
11881.398328646887 vs 11881.398328646887). **So the defect did not exist until the world had orbits, and
is invisible to any test whose world stands still — which was every test.**

Owner-observed consequences, since reproduced: logging in at the origin lands the player inside a planet
at (0,0,0); entering a planet lands them outside it and the planet and star then trade them back and
forth without end.

Also measured, separately: the router discarded 14,884 frames from running shards in one cluster run
(now 0); a live cluster was measured with its orchestrator and its own gateway on two different worlds
(the world knob has since been deleted).

## 2. The mechanism

**One placement table per PARENT, rebuilt whole every tick.**

- Contents: the anchor (the realm doing the authoring) and, for each of its DIRECT children, where that
  child sits in the anchor's frame, at one named instant.
- **No cell for the anchor's own position** — nor its parent's, ancestors', siblings' or grandchildren's.
  Not even a zero. SL1 made into a shape rather than a habit: a shard cannot read where it sits because
  there is nowhere to read it from.
- **One table per anchor, not per shard.** A shard co-hosting a chain authors in two different frames;
  separate tables make mixing them unconstructible.
- **The reader takes NO CLOCK.** A lookup that cannot see time cannot integrate motion — SL4 held by the
  compiler, inside the crate where the defect lives. The instant is a property of the table.
- Three answers, no fourth: *you are the anchor* → your own origin; *you are my direct child* → here;
  *anything else* → **I cannot say** (every consumer already has a safe path for the refusal).
- **One writer**, called once per tick per anchor, is the only code in the tree that names an orbit. The
  single "does this child move" test lives there and decides ONLY whether to recompute; both arms write
  the same kind of value into the same row, read identically by everyone.
- **Nothing branches on realm kind.** The writer's only question is "am I the parent of this thing" —
  authority, not type.

## 3. Area of interest and liveness (SL7)

```
LIVENESS   the realm itself, looking at itself and ONE level down:
             do I hold occupants?  do I have a live direct child?
             either ⇒ stay active; neither ⇒ shut down after the cooldown

INTEREST   the PARENT, never the realm about itself:
             for each of MY direct children — is it within the interest of an occupant I hold,
             or of an OCCUPIED CHILD, taken at the placement I authored for it?
             yes ⇒ demand it stays alive
```

An occupied realm is its own occupants' proxy at its parent's scale; the error is bounded by the child's
own size, which is the resolution at which the parent's decision is meaningful. The parent authors its
children's velocity too, so warming-ahead needs nothing told to it. **What crosses a boundary: one bit of
occupancy, upward, which already crosses today. Nothing else, at any depth, with nothing central.**

## 4. The owner's decisions (2026-08-12)

- **Q1 — login home.** Store a hardcoded home for now (a realm plus a pose inside it). Design so it can
  become dynamic later (load-based choice of appearance point, e.g. different apartments in a city).
  **LOGIN IS NOT A SPECIAL PATH**: the realm chain spins up by the ordinary demand mechanics and *where
  the initial position came from must not matter to any machinery*. No login edge cases.
- **Q2 — the outline message.** A shape is *how a realm looks* (the realm's own business); a position is
  *where it is* (the parent's). **They never travel in the same message.** The outline carries no
  position; the per-tick lane is the only place a position is stated.
- **Q3 — the realm feed.** Send a child's position **when the value changed** since the last row sent.
  The test is on the number everyone reads, never on what kind of thing it is. ("Movers only" is not
  available — it cannot be expressed without a motion test on the read path.)
  Owner's correction to keep: a station SHOULD move and will likely be a movable realm inside a planet
  near its edge; the right example of a child that does not move is a CITY AREA on a planet.
- **Q4 — the relay lanes.** Delete both (occupant poses upward, entity sets downward). Under SL7 neither
  is needed; porting them would also silently let them start carrying spin.
- **Q5 — liveness and a shard's scope.** As in §3. A realm has zero control over its parent. A shard told
  to hold a realm ABOVE its own would author its own parent's children — itself among them — and end up
  holding its own position: refuse at boot, loudly, like the existing lineage fence.
- **Q6 — the client's box scene.** The client holds no stored position and takes every position from the
  live feed. Nothing is drawn before a realm streams itself (SL3).

## 4b. MEASURED 2026-08-12, AFTER Step 0 — the order below changed

Step 0 landed as `crates/bins/tests/one_containment_answer.rs`, built through the shipped boot
(`vd_bins::boot_world` + `vd_bins::boot_regions_and_movers` + `RealmRegions::new().with_moving_children()`).
It reproduced §1's table to the last digit, so §1 is now a measurement through the shipped path rather than
a transcription. A THIRD test was added and it **passes**: the shard's per-tick feed (`child_placements`)
and its conversion context (`frame_context`) place every child identically, at every sampled tick.

**So the second answer lives in exactly ONE production place: the router's login descent.** Nothing else in
the shipped code asks the parent's downward question — `child_signed_distance` has one production caller
(`worldgen::deepest_containing_child`), reached only from `container_pose_in`, reached only from
`gateway::home_placement`. Every shard-side consumer already reads one number.

Consequences for the order:

- **Step 1 changes no behaviour today** — that is now proven, not claimed. It remains worth doing as the
  lock that stops a second producer returning, but it cures nothing on its own.
- **Step 2 is the cure**, and it lands first. The router's own comment had already named this and stated it
  was blocked on one owner decision — which Q1 settled.
- The router's remaining world reads (`forest_root`, `render_pin`, `realm_registry_for_home`) are NAMES,
  not positions. Deleting the descent removes the only place the router derives a POSITION.

LANDED so far:

- `vd_core::home` — `StoredHome` (a lineage plus a pose in that realm's own frame) and `HomeRegistry`
  (per account, with one fallback). Three unit tests, green.
- `vd_core::worldgen::default_home_realm` — the home realm chosen by LINEAGE POSITION (root → first child
  → first child), a name and never a measurement. The fallback pose is the realm's OWN CENTRE, so there is
  no magic offset; in this world the nearest child's surface is ~12 m away, which makes a star system's
  origin empty space by construction.
- `gateway::SeedInjectorConfig.spawn_poses` → `homes: HomeRegistry`; `home_placement` is now a LOOKUP and
  `home_spawn_offset` is deleted. The router lib compiles.

### The descent is deleted — what went with it, and what is owed

Deleted from `vd_core`: `container_pose_in`, `container_coord_in`, `container_coord_at`, `container_realm_in`,
`descend_containing`, `deepest_containing_child`, `child_contains`, `WorldView::container_coord`, and
`geometry::child_signed_distance`. The router's `home_placement` is a lookup; `home_spawn_offset` is gone.
`vd_bins::resolve_homes` builds the registry; `VD_SPAWN_POSES` now means metres FROM THE HOME REALM'S OWN
CENTRE.

**COVERAGE OWED — three properties lost their tests with the function that expressed them.** Each must be
rebuilt on the SHARD's own containment once Step 1's table lands; none is covered today:

1. *Exact-integer containment* (was `the_containment_distance_is_exact_when_the_whole_number_part_is_not_zero`,
   geometry): a subject and a region at the same place in two spellings — one at a whole cell with no
   leftover, the other a cell lower with a full-edge leftover — must read distance zero, not one cell edge.
   Its failure mode is re-homing a player who has not moved.
2. *The container sequence, both ways* (was `symmetric_recross_resolves_the_full_container_sequence_both_ways`,
   sim): origin → the gap → the sibling system → back, resolving System 7 → Galaxy → System 8 → Galaxy →
   System 7.
3. *Escape lands in the immediate parent* (was `escape_soi_lands_in_the_immediate_parent_not_a_skipped_ancestor`,
   sim): leaving a planet lands in its star system, never the galaxy.

Those tests used the descent as their ORACLE, so they could not be repointed without a second
implementation, which the standing laws forbid. They are listed here rather than quietly dropped.

### Suite state after the deletion (measured, `cargo test --workspace --no-fail-fast`)

75 suites pass, 4 fail. Not yet diagnosed, and NOT to be described as green:

| failing test | first read |
|---|---|
| `a_dot_re_homes_source_to_dest_over_the_process_dual_shard_tier` | names walk-forest realms 7/8 — suspected the same one-world-collapse fallout that `boot_guard` had |
| `p1_parity_real_binaries_over_quic` | "unexpected class RealmSnapshot" — may be caused by the home move (the fallback home is now a star system, which streams realm rows) |
| `proc_launch_backend_forks_boots_identifies_and_reaps_a_real_shard` | "realm 1000 died immediately… a SURVIVOR still holding this realm's slot in the RLM port band" — the known unreaped-child issue |
| `the_box_and_the_thing_standing_in_it_draw_at_one_point` | "the drawn box is measured from the STAR, not from the planet: 5 m" |

`boot_guard` was ALSO failing before today's edits for the same one-world reason (it booted `planet:7`, a
realm only the hand-placed walk forest has); it now names a planet of THE world and passes.

**⚠ SUPERSEDED — the tree compiles again; kept for the record.** Nine `armed_injector(<poses map>)` call sites in the gateway's
own test module (around lines 8914–10000) and one `SeedInjectorConfig { spawn_poses: … }` in
`tests/tests/realm_lifecycle_e2e.rs:558` still pass the old map. RESUME BY: replacing each with the test
helpers already added beside `armed_injector` — `default_homes()`, `homes_at(realm)`,
`homes_for(account, realm, x)`. Those tests currently build their EXPECTED lineage by calling
`world.container_coord(25,0,0)` — the very descent being deleted — so each has to NAME its home realm
instead. The realm to name is whichever one 25 m from the walk-scale root descends into; read it once, put
it in a named test constant, and the expectation becomes `coord_of_realm(regions, that_realm)`.

Only after those compile can the descent itself go: `worldgen::container_pose_in`, `container_coord_in`,
`descend_containing`, `deepest_containing_child`, `child_contains`, `WorldView::container_coord` and
`geometry::child_signed_distance` — plus their tests. That deletion is what makes Step 0's first test pass.

## 4c. THE DRAWN-POSITION DEFECT — root-caused 2026-08-12, owner-confirmed in flight

The owner flew it: the login fix took (no longer born inside a planet), and **entering a planet still puts
you far from it**. Root cause, measured, and already written into
`tests/tests/frame_conversion_e2e.rs` by the arc that found it:

```text
realm row   frame PlanetCentered{7}      pos (4.9698, 0.5487, 0)
occupant    frame AreaLocal{7,7}         pos (0, 0, 0)          gap 5.000 m
```

**One shard emitted both, in one tick, in two frames.** The occupant is measured from the realm it stands
in; that realm's own box is measured from the realm that carries it. The client draws both as given, so the
player and the ground under them are drawn one whole placement apart — which is exactly "I re-homed onto
the planet and I am outside it".

WHERE THE GAP IS. `emit_realm_frames` has two lanes:

- **(b) the CASCADE**, to a child realm on another shard, **restates every row into that child's frame**
  before shipping (`restate_rows_in_child_frame`) — the parent subtracts, because the parent is the only
  party holding the placement. This lane is correct.
- **(a) the DIRECT emit**, to the gateways of local emitting dots, ships the rows **as authored, in this
  shard's own frame**, to every local observer regardless of which realm that observer is standing in.

So a shard that hosts a parent realm AND a child realm restates nothing between its own two lanes. The
per-level subtraction only happens when the next level has its own shard to relay to.

THE FIX, and it adds no machinery. The direct lane groups its local observers **by the frame their pose is
stated in**, and ships each group the rows restated into that frame — through the SAME
`restate_rows_in_child_frame` the cascade uses. An observer standing in this shard's own realm gets the
rows as authored (the identity, byte-identical to today). An observer standing in a direct child gets them
restated, once, by the party that authored the placement. Anything else is counted and dropped loud, never
forwarded under a label that no longer matches its numbers.

Gate: `the_box_and_the_thing_standing_in_it_draw_at_one_point`.

**LANDED 2026-08-12, and it moved the failure.** `restate_rows_in_frame` is now the one restatement, keyed
by FRAME rather than by recipient; the direct lane groups its local observers by the frame their pose is
stated in and ships each group its own space; an observer in a frame this shard neither is nor authors is
counted (`realm_rows_unplaceable_observer`) and shipped nothing. `vd-sim` compiles.

The gate now fails EARLIER and for a different reason: **zero samples** where it wants two. The cause is a
rule the restatement already had and is almost certainly RIGHT — *the recipient's own row is dropped*, since
a realm's centre measured in its own frame is the origin every tick forever. The only moving realm in that
fixture IS the one the occupant stands in, so restating into the occupant's frame legitimately leaves
nothing to send.

**DO NOT make this green by putting the row back.** That is the pre-SL3 model, where a downstream party
composed. Under SL3 the realm draws itself: the client already receives that realm's own outline at its own
origin, and the box the player stands on belongs there. The open question is whether the CLIENT draws a
realm it occupies from the outline lane when no per-tick row arrives — and if it does not, that is where
the work is. The test samples `realm_view.realm_pose(...)`, which needs a row, so it may be asserting the
old model; settle what the client does BEFORE touching either side.

## 5. The work, ordered — each step landable and gateable alone

**Step 0 — THE FAILING TEST, and everything is gated on it.** One point, one tick, the two answers
asserted equal. It must fail today. It must be built through the SHIPPED path (the world generator the
shard and router boot through, and the shard's real context construction) — the audit's own measurement
transcribed that construction into the test, and a transcription is not proof.

**Step 1 — one producer, in place.** Introduce the table, the reader and the per-anchor writer where the
code already lives. Repoint every consumer: containment, both sides of a hand-off, the realm feed, the
down-cascade, the interest loop, the outline lane, the boot fence. Delete the two rival lookups and the
always-succeeds one. Delete the clock argument from the placement lookup. Delete the motion test that
decides what the realm feed ships, and the outline's fallback to the stored centre.
*Stop here and:* the crossing path is single-sourced, but **a login at the origin still lands inside a
planet**, and the stored field still exists for something new to read.

**Step 2 — delete the stored centre outright.** Not re-baked to an epoch value: a second answer that is
right at one instant is still a second answer. The login descent stops compiling — that is the point.
Land the stored home (Q1) as its replacement.
*Stop here and:* both measured symptoms are gone, but the physics is still nameable from the crossing
code — the rule held by review, which SL4 says is not enough.

**Step 3 — the dependency rule.** The orbit maths and the world generator leave the shared crate, so the
shard and the router cannot name an orbital element at all. Add the check to the pre-merge gate:
production edges forbidden, **test-only edges allowed and stated explicitly** (a fixture must be able to
plant a moving child or every moving-child branch loses its cover). Gate conditions: the generator must
not re-export anything from the motion crate. **This enforcement does not exist in any form today.**
*Stop here and:* it works, and the fence is missing.

**Step 4 — children that arrive and leave.** A roster generation that ticks on any change; a reader
holding a table from another generation gets "I cannot say". Roster changes confined to one point
immediately before the writer. Nothing is broken today; it breaks the day a ship is created or destroyed
near somebody.

**Step 5 — interest by occupied-child proxy (§3), and both relay lanes deleted.**

## 6. Unmeasured — do not restate any of this as fact

- The perpetual hand-in-and-hand-back loop was **never reproduced**. The disagreement behind it is
  measured; the mechanism of the loop is not.
- Whether a shard ever actually takes the path that would let it learn its own position (the extra
  context built when an occupant's realm is one the shard carries rather than authors) — the path was
  read, never observed taken.
- Whether an arrival can occur before a shard's first clock sync (would change a placement into a refusal).
- The cost of building the table once per tick versus solving per lookup.
- The blast radius of deleting the movers-only feed filter — the highest-risk line in Step 1; a test is
  named for the behaviour it removes.
- Whether the client's box scene survives the stored centre's deletion without a visible change.

**A doc comment in the tree is MEASURED WRONG and must not be trusted:** `crates/core/src/worldgen.rs`
around line 203 states the orbital placement has no production producer and carries a dead-code marker to
match, while a production producer some hundreds of lines below makes all five planets of every star
system. That comment is part of why the zeroed centre survived.


## 4d. SECOND FLIGHT, 2026-08-12 — the drawn-position fix did NOT cure it

Owner flew the restatement fix. Result: **login is still correct**, and everything else is unchanged, with
two new details that were not in the first report:

1. Re-home onto a planet still lands OUTSIDE it.
2. Returning to the star system produced **a lot of jitter** — the flap, seen as jitter rather than as a
   clean teleport back and forth.
3. Landing on the planet a second time **froze the whole star system permanently**.

### What this rules out

The two-frames emit gap was real and is fixed, but it is NOT the cause of "I land outside the planet". So
the displacement is upstream of drawing: it is in the POSE the player carries across, or in the containment
answer that decides they should cross at all — not in how it is stated to the client.

### The freeze now has a shape, and it points at a named lane

A player standing on the planet is served the system's realms **only by the CASCADE** — the parent ships its
authored rows down to an ACTIVE CHILD. `active_children` is driven by `RetainedOccupants`, which is fed by
the UP-RELAY of occupants. If that relay stops or was never established for this crossing, the parent keeps
ticking and authoring while the player receives nothing, and the whole system freezes for them — permanently,
which matches exactly what the owner saw.

That relay is one of the two lanes the owner ordered DELETED (§4, Q4), to be replaced by SL7: liveness from
one occupancy bit upward, interest decided by the parent from placements it already authored. So the freeze
and Q4 are the same piece of work, and it is Step 5 — which means Step 5 is not last, it is next.

### METHOD CORRECTION — binding from here

Two fixes have now been shipped on reasoning from code and neither cured the reported symptom. No further
change to the crossing path may be made without a REPRODUCTION AT THE PROCESS TIER that fails first:
the real binaries, a headless dev-control client driven into a planet, and the measured quantities being

- the pose the source shard holds for the player at the moment it hands them over,
- the pose the destination shard admits,
- the containment answer each side computes for that pose,
- whether the cascade reaches the player after the crossing.

HR6 exists for exactly this and has not been used on this defect. The client core is renderer-free with the
`dev-control` feature, so no Bevy build is needed to drive it.


## 4e. ⚠ RETRACTED — I read a STALE DOC COMMENT as if it were the code

`crates/bins/tests/rlm_demand_login.rs`, in the doc of
`a_planet_to_system_return_commits_both_rehomes_and_the_player_rides`, states it outright:

> RESIDUAL (owed, NOT asserted — DEFERRED D-RLM-14 / floating-origin S6 / VU-6): after the return commits,
> the SURROUNDING realm feed (per-tick `RealmSnapshot`) does not yet re-advance on the cross. The dest's
> read sub re-opens only from the `SubscriptionReady` at `on_saga_promote`, and the realm SCENE is
> re-streamed only at a home-entry `Active` promote (never on a cross) — so a returned player is live and
> rides its own realm, but its neighbours can stall. The tail below OBSERVES it (logs the owed status)
> rather than panicking, so the landed fixes gate green.

**That is why three process-tier re-home tests pass while the game is broken.** They assert the location
LABEL and no-flap; the one gap that matters is logged instead of asserted.

### One cause may produce all three reported symptoms

If the realm feed does not re-advance across a crossing, then after crossing into a planet the client holds
a STALE scene:

- **"I land outside the planet."** The player's own pose is now planet-local and correct, but the planet's
  BOX is frozen where it was before the cross. Drawn against a stale box, a correctly-placed player appears
  displaced by however far that realm has since travelled.
- **"A lot of jitter returning to the system."** A live pose lane against a stale/partly-refreshing realm
  lane — the two disagree per frame.
- **"The whole system froze permanently."** The neighbours' feed never resumes at all.

This is ONE defect, already named D-RLM-14 / VU-6, not three.

### The work

Re-stream the realm scene ON A CROSSING, not only at a home-entry promote — and flip that test's logged
residual into a hard assert, so the gate can never again be green while this is broken. The assert is the
first thing to land: it is the failing reproduction the method correction (§4d) requires, and it already
exists in the tree as prose.


### 4e RETRACTION, same day, and the reason it happened

§4e said the freeze was a ledgered gap that nobody asserts. **That is wrong.** I quoted a doc comment and
did not read the body underneath it. The body was rewritten on 2026-08-06 after a 17-agent audit: the
residual IS a hard assertion now (`neighbour_feed_live`, 20 applied frames in 15 s), the audit could not
reproduce the stall in four runs, and it found the stall has no mechanism — the per-tick placement forwarder
gates on nothing but a live session and an open channel, and that channel is opened by the very event that
commits the crossing.

Both directions are asserted and both pass: the cascade climbing after crossing INTO a planet, and the
neighbour feed resuming after crossing back OUT. The doc above them still describes the old world.

This is the project's own standing rule — trust the code, never the comments — and I broke it while quoting
a comment about a defect that comments had already caused once. Nothing was built on the wrong conclusion;
it cost one turn.

### THE DIFFERENCE THAT IS REAL, and it is not in the code

Every one of those passing process-tier crossing tests sets:

```
VD_VISUAL_ORBIT_SLOWDOWN = 300
```

The owner flies with it unset — full orbit speed. So the suite proves the crossing machinery on a world
whose realms move **three hundred times slower** than the one being flown. Every symptom reported is one
that a slow or lagging realm feed would hide at 1/300 speed and expose at full speed: a box drawn where the
planet used to be, a live pose arguing with a stale box (jitter), neighbours that appear to stop.

NEXT, and it needs no new machinery to try: run the existing round-trip and arrival tests with the slowdown
REMOVED. If they fail, the reproduction is free and has been one line away all along. If they pass, the
difference is the renderer or the client, and that is where to look next.

⚠ Note for whoever runs it: the knob is set with `std::env::set_var` inside the test and removed at the end;
a leak quasi-freezes any later full-speed test in the same process.


## 4f. ★ REPRODUCED 2026-08-12 — the crossing tests only pass because the world is slowed 300x

`VD_TEST_ORBIT_SLOWDOWN=1 cargo test -p vd-bins --features dev-control --test rlm_demand_login
a_flying_occupant_re_homes` — the SAME shipped test, at the SAME orbit speed the owner flies. It FAILS.
(The three crossing tests now read the knob from `VD_TEST_ORBIT_SLOWDOWN`, default 300, so a full-speed run
needs no code change.)

```text
[repro] inner planet Planet(7701581858760374086) epoch_len=17.9 — flying to its FIXED epoch center
[repro] inner leg  0: tick=Some(68)  loc="System 7" own_len=Some(0.0)                best=17.93
[repro] inner leg  5: tick=Some(121) loc="System 7" own_len=Some(17.500067234724444) best=0.43
[repro] inner leg 10: tick=Some(121) loc="System 7" own_len=Some(17.500067234724444) best=0.43
...  legs 10 through 80 IDENTICAL: tick 121, own_len 17.500067234724444, best 0.43
[repro] inner leg 85: tick=Some(224) loc="System 7" own_len=Some(10.423263212127889) best=0.43
[repro] inner leg 90: tick=Some(724) loc="System 7" own_len=Some(11.053729935262252) best=0.43
```

TWO facts, both matching what the owner flew:

1. **The player reaches 0.43 m from the planet's centre and is NEVER re-homed into it.** `loc` stays
   `System 7` throughout. A planet of radius 4.16 m, and an occupant 0.43 m from its middle, still belongs
   to the star. That IS "I am at the planet but not on it".
2. **The player's own state FREEZES.** Tick 121 and own position `17.500067234724444` repeat identically
   across sixteen consecutive polls, then jump to tick 224, then 724. The sim stops advancing for that
   player and then skips. That is the freeze and the jitter.

At a 300x slowdown the planet barely moves and neither happens, which is the whole reason every crossing
test in the suite is green while the game is broken.

**THIS IS THE REPRODUCTION.** Deterministic, free, at the process tier, with the real binaries, and one
environment variable away from the shipped gate. No further change to the crossing path may be proposed
without running it.

FIRST QUESTION TO ANSWER, and it is now a measurement rather than a guess: which comes first — the frozen
tick, or the missed crossing? A player whose ticks stop cannot cross; a crossing that never fires cannot by
itself stop a tick. The freeze is the more likely cause and the logs to read are the per-realm ones the
fixture captures.


## 4g. ★★ THE DEFECT, MEASURED AT THE HAND-OFF — 2026-08-12

From the captured per-realm logs of the full-speed reproduction (§4f), one crossing, one tick:

```text
SYSTEM shard, tick 2312 — HAND-OFF DOWN
  from_pos  = (0.00067364, 0.00085675, 0.00012717)                 the occupant, in the STAR's frame
  child_at  = (16.310805,  -2.766803,  -0.513391)                  where the star put that planet
  landed_at = ( 8.022876,  26.366292,   3.977385)   len 27.845422   what it SHIPPED

PLANET shard, same tick — ARRIVAL
  accepted_at = (8.022876, 26.366292, 3.977385)     len 27.845422   accepted as given, correctly
```

**`landed_at` is not `from_pos − child_at`.** That difference is `(−16.31, +2.77, +0.51)`, length **16.55**.
The shipped value is length **27.85**, and it is not the negation, the transpose or any permutation of it.
The arrival end is blameless: it accepts what it is handed, which is what the ground rule requires of it.

Two things are wrong at once and both are visible in that one line:

1. **`from_pos` is essentially ZERO.** The source shard believes the occupant is at the STAR's own centre at
   the moment it hands them away — while the client, the same second, reports its own distance as
   `17.500067234724444`. The party doing the subtraction is subtracting from the wrong position.
2. **The subtraction does not produce the difference.** Whatever `transfer_frame` composed — a rotation, a
   placement resolved at a different instant than the pose is stamped at, or both — it is not the
   `pose − placement` the log line claims to be describing.

The planet's radius is 4.16 m. The occupant is measured by the test as 0.43 m from that planet's centre. It
arrives 27.85 m out. **That is the owner's "I re-homed to the planet and I am way outside it", in metres.**
The planet then re-homes them straight back out, the star hands them down again, and the log shows exactly
that: fences 2, 3 and 4 on one entity in 700 ms — the flap, and the jitter.

### Why a 300x slowdown hides all of it

Both faults scale with how far the planet travels between the instant the pose was stamped and the instant
the placement was resolved. At 1/300 speed that distance is a few centimetres and every assertion in the
suite still holds. At the speed the game is played it is tens of metres.

### The next step, and it is now arithmetic and not archaeology

Read `flush_pose_for_dest` (the producer of the HAND-OFF DOWN line) and establish, in order:
(a) why `from_pos` is the star's origin rather than the occupant's actual position, and
(b) which placement and which tick `transfer_frame` actually composed, given `child_at` was logged beside it.
The two logged vectors and the shipped result are three of the four numbers needed; the fourth is inside
that call.


## 4h. THE HAND-OFF, WITH THE MISSING NUMBERS — 2026-08-12, three findings

Re-ran §4f with the occupant's own frame and its placement added to the line. Three hand-offs of ONE entity:

```text
1) at_tick=1099  from_realm=System(7)  from_frame=SystemSpace{7}  from_at=(0,0,0)
   own_frame=SystemSpace{7}
   from_pos=(7.9e-5, 8.5e-5, 4.9e-4)   child_at=(16.659, 2.419, -0.494)
   landed_at=(7.787, 17.263, 3.981)  len 19.35

2) at_tick=1413  same shape, child_at=(-17.172, 0.879, 0.529), landed_at len 18.57

3) at_tick=5853  from_realm=System(7)  from_frame=SystemSpace{7}
   own_frame=PlanetCentered{7701581858760374086}      child_at=(0,0,0)
   landed_at=(-18.839, 8.205, -0.693)  len 20.56
```

**(a) ⚠ MY EARLIER ARITHMETIC CLAIM (§4g) IS UNSAFE.** `landed_at` logs `placed.pos.offset()` — the SUB-CELL
RESIDUAL of a tiered `LatticePos`, not the whole position. `transfer_frame` writes
`LatticePos::at(cell_out, new_pos)`, so any whole-cell part lives in the CELL and never appears in that
field. So "landed_at is not from_pos − child_at" may be an artefact of logging a partial value. It must be
re-measured by logging the CELL beside the offset, or a `delta_m` against the origin, before anything is
concluded from it. The arrival line has the same flaw. **Do not build on §4g until this is settled.**

**(b) A SHARD HANDS AN OCCUPANT TO ITSELF.** Line 3: `own_frame` is the PLANET, the hand-off target is that
same planet, and `child_at` is ZERO. A realm subtracting a zero placement to hand somebody to itself is the
flap, stated in one line. It also means `direct_child_frame(from_realm, to_realm)` answered yes for a realm
this shard IS rather than one it AUTHORS.

**(c) `from_realm` AND `own_frame` HAVE PARTED COMPANY.** Same line: `from_realm=System(7)` derived from the
pose's frame label, while the shard's own frame is the planet. `flush_pose_for_dest` builds its conversion
context from `from_realm` — a realm read off the OCCUPANT'S LABEL — not from the realm this shard actually
authors. When those differ, the context is anchored on the wrong realm and every placement in it is the
wrong one. Under SL1 the anchor may only ever be the shard's own realm; taking it from a pose's label lets
an occupant's stale label choose the frame the arithmetic happens in.

Next: settle (a) with a cell-inclusive log line, then (b)/(c), which are one bug — the anchor must be the
shard's own realm, and a realm can never be its own direct child.


## 4i. THE ANCHOR FIX LANDED, AND THE TEST'S TARGETING IS ONLY VALID AT 1/300 SPEED

**LANDED.** `flush_pose_for_dest` now anchors on `config.realm` — this shard's own realm — instead of on
the realm read off the OCCUPANT'S FRAME LABEL. A shard's region set contains its ancestors, so a stale
label resolved to an ancestor and the shard then built its conversion context anchored on a realm it does
not author; the placement it subtracted came back as zero and it handed the occupant to the realm it was
itself hosting. Counter `flush_anchor_not_own` counts and logs every such label. `vd-sim`: 452 tests green.

**MEASURED, and it is a second real defect:** the warning fires with

```text
labelled=Planet(1164718096683563219)   own=System(7)
```

An occupant's pose reached the star's crossing path carrying the frame of a DIFFERENT planet — not the
target, not the star. Where that label comes from is not yet established and it is the next thread.

**⚠ THE TEST'S TARGET IS WRONG AT FULL SPEED, so §4f's run is not a faithful reproduction.** The fixture
flies to the planet's FIXED EPOCH CENTRE (`inner_planet_mover` → `orbital_state(elements, 0.0)`), which is
where that planet sits only while it barely moves. That is exactly what `VD_VISUAL_ORBIT_SLOWDOWN=300` buys.
At full speed the planet has long left that point, so the run chases empty space: closest approach stayed
17.93 m against a 4.16 m boundary, and the occupant's own distance wandered 12 → 31 → 41 → 23 m.

So the full-speed run proves the SUITE is not exercising a moving world, but it cannot yet prove the
crossing arithmetic. **The fixture needs a MOVING target: fly to where the planet IS at the current tick,
re-aimed as it moves, exactly as a pilot does.** That is the next piece of work and it is the honest
reproduction. (The very first full-speed run did reach 0.43 m from the planet's centre without crossing —
that observation stands and is not explained by the targeting.)


## 4j. ★★★ THE FREEZE IS PRIMARY — reproduced at flight speed with a moving target, 2026-08-12

The fixture now flies at WHERE THE PLANET IS, re-aimed every leg, instead of at its epoch centre. Run at
flight speed (`VD_TEST_ORBIT_SLOWDOWN=1`):

```text
NO CROSSING (inner): chased the inner planet's LIVE centre for 151s but location never left "System 7";
closest approach 5.29 m vs ~4.16 m SOI.

[repro] inner leg 55: tick=Some(680) loc="System 7" own_len=Some(42.046683199800626) best=5.29
[repro] inner leg 60: tick=Some(680) loc="System 7" own_len=Some(42.046683199800626) best=5.29
[repro] inner leg 65: tick=Some(680) loc="System 7" own_len=Some(42.046683199800626) best=5.29
[repro] inner leg 70: tick=Some(680) loc="System 7" own_len=Some(42.046683199800626) best=5.29
[repro] inner leg 75: tick=Some(680) loc="System 7" own_len=Some(42.046683199800626) best=5.29
```

**The client's universe tick STOPS at 680 and never moves again.** Its own position stops with it, to the
last digit, across every remaining leg. The approach reached 5.29 m from a boundary at 4.16 m — a metre
short — and then everything stopped.

**So the ordering is settled, and it is the opposite of the way this was being chased.** The freeze is
PRIMARY; the missed crossing is its consequence. A player whose ticks have stopped cannot close the last
metre, cannot trip containment, and cannot cross. Every previous hypothesis treated the crossing as the
fault and the freeze as a symptom.

This also matches the owner's flight exactly: approach the planet, and it stops.

THE QUESTION IS NOW ONE QUESTION: **what stops the client's tick as an occupant approaches a child realm's
boundary?** Candidates, in the order the evidence favours them — the shard stops emitting snapshots for
that session; the session's subscription is closed or re-keyed on the approach; the shard itself stalls or
dies. The per-realm logs of this run and the gateway's counters answer it directly, and the run is free to
repeat.

This fixture is now the gate. It must stay aimed at the LIVE position and must be run at flight speed.


## 4k. ★★★★ THE CAUSAL CHAIN, COMPLETE — from the frozen run's own logs, 2026-08-12

The per-realm logs of the §4j run say it outright:

```text
WARN  an occupant's pose names a realm this shard does not author
        labelled=Planet(1164718096683563219)  own=System(7)
ERROR refusing a crossing this shard cannot place — the entity is NOT adopted
        err = the arriving pose's frame is neither mine nor one of my direct children
WARN  Promote DEFERRED — the crossing pose has not landed yet (the saga re-drives the Promote)
```

plus fences 2, 3, 4, 5 and 6 on ONE entity inside six seconds.

**The chain, end to end:**

1. An occupant standing in the STAR carries a pose labelled with the frame of one of the star's PLANETS —
   a realm it is not in. (`Planet(1164718096683563219)` while the shard is `System(7)`.)
2. The hand-off ships that pose. The label travels with it.
3. The DESTINATION refuses to adopt: *the arriving pose's frame is neither mine nor one of my direct
   children.* It is right to refuse — it cannot place a number stated in a space it knows nothing about.
4. The promote therefore never lands, and the saga re-drives it forever.
5. **The occupant is left with no live authority. Its ticks stop. That is the freeze**, and with the freeze
   the last metre to the boundary is never closed, so no crossing ever completes.

Every symptom the owner reported falls out of this single chain: no arrival on the planet (step 3), the
fight (steps 4–5 re-driving, fences 2–6), the jitter (the same), and the permanent freeze (step 5).

**THE ROOT CAUSE IS THE WRONG FRAME LABEL ON THE OCCUPANT'S POSE**, produced while it is still in the star.
Nothing downstream is at fault: the destination's refusal is correct and the saga's re-drive is correct.

NEXT, and it is now a single question with a small search space: what writes a PLANET's frame onto an
occupant that is in the STAR? The candidates are the containment detector choosing a container and
relabelling, and `rebind_pose_to_dest` on an earlier attempt of the same crossing. The
`flush_anchor_not_own` counter already fires exactly when it happens, so the label can be caught at the
moment it is written by logging where a pose's frame is set on the source side.


## 4l. WHERE THE BAD LABEL IS NOT, and the one measurement left

Traced the write sites for an occupant's frame label on the source side:

- **The durable flush does NOT write back.** `flush_source` reads `dot.pose` through `&Dots` and ships the
  converted copy; the local dot keeps its own frame. So a refused or re-driven crossing cannot leave the
  source's own dot relabelled. Ruled out by the type.
- **The transient flush** goes through the same helper and is documented as having no receiver-side
  conversion, but the subject here is a Durable player, so it is not on this path.

**The label that appears is not the target's.** The warn says `Planet(1164718096683563219)`; the planet
being flown to is `Planet(7701581858760374086)`. The occupant is labelled with a DIFFERENT planet of the
same star — one it is nowhere near.

That points back at containment, not at the hand-off: a wrong container was chosen, a crossing to THAT
planet was started, and the label follows the decision. It is the same shape as §1's original defect — every
planet reading as containing the same point — except the surviving path is the shard's own frame-context
answer rather than the deleted downward one. If the star's conversion context is missing or zeroing the
placements of some of its children, they all sit at its origin and the deepest-wins fold picks whichever
wins the tiebreak, which would be an arbitrary planet exactly like this one.

**THE ONE MEASUREMENT LEFT, and it needs no new machinery:** in the containment detector, for the tick where
the wrong container is chosen, log every region's signed distance together with the placement the context
returned for it. If several planets report a negative distance for one point, §1's defect is alive on the
surviving path and the placement table (Step 1) is the cure rather than a lock. If only one does and it is
the wrong one, the fault is in the fold or the band.


## 4m. ★★★★★ ROOT CAUSE — the containment reframe returns the wrong number, measured 2026-08-12

From the shard's own containment scan, at flight speed, with the placement it used printed beside it:

```text
CONTAINMENT: a child claims this point
  claimed_by      = Planet(7701581858760374086)
  signed_distance = -2.0641759961527923            NEGATIVE = INSIDE
  placement       = (-9.78905578539251, 14.985323592247008, 0.3856707995226916)   |p| = 17.90
  pose_at         = (0.000174, 0.000875, 0.000169)                                 the STAR's centre
  pose_frame      = SystemSpace { system_seed: 7 }
```

**The occupant is at the star's own centre. The planet is 17.90 m away, and the shard's own containment says
the occupant is 2.06 m INSIDE it.** The planet's boundary is ~4.16 m, so the honest answer is `+13.7 m,
outside`. The placement is right; the pose is right; the answer is wrong.

Work it back: a signed distance of −2.06 against a 4.16 m boundary means the reframed position came out
**2.10 m** from the planet's centre. It should be 17.90 m. So `transfer_frame` is not producing
`pose − placement` — it is producing something an order of magnitude smaller.

Both origin cells are zero here (every placement in the tree is cell-anchored at zero), so the subtraction
should be a plain `0 − placement`, length 17.90. It is not. The suspect is the tiered coordinate write on
the way out — `LatticePos::at(cell_out, new_pos).convert_tier(...)` — where a value larger than a fine cell
edge is normalised into the CELL, and something downstream reads only part of it back. That is the same
foot-gun already recorded against D-41 ("cell-preserving map_offset, a foot-gun for live moves"), and it is
the one place these numbers can lose their magnitude.

**THIS IS THE DEFECT BEHIND EVERY SYMPTOM.** A point at the star reads as inside a planet ⇒ a crossing is
started to a realm the occupant is nowhere near ⇒ the pose is labelled with that planet's frame ⇒ the true
destination refuses it ("neither mine nor one of my direct children") ⇒ the promote never lands, the saga
re-drives forever, the occupant has no live authority, and its ticks stop. Arrival outside the planet, the
fight, the jitter, the permanent freeze — one arithmetic fault, at the bottom.

It is also EXACTLY §1's defect: a point at a star reading as inside its planets. §1 was measured on the
downward lookup, which is now deleted. The same wrongness lives on the surviving own-frame path, which is
why deleting the other answer changed nothing that could be flown.

NEXT: a unit test in `vd_core::frame` — one pose at the origin, one placement 17.9 m out, cells zero —
asserting `transfer_frame` returns 17.9 m. It is arithmetic, it needs no cluster, and it will fail.


## 4n. ★★★★★★ THE FAULT, TO THE DIGIT — the source frame's placement is not the identity

Live shard, containment scan, with the reframed position printed beside the inputs:

```text
pose_at    = ( 0.00096,  0.00039,  0.00030)        the occupant, at the star's centre
placement  = (-9.92281, 14.89284,  0.38921)        the planet, |p| = 17.90 m
reframed   = (-1.16412,  1.74719,  0.04566)        |r| =  2.09999 m      cell = (0,0,0)
region_shape  = Shell { r: 4.160705354131045 }
signed_distance = -2.060717498430848
```

`reframed` must be `pose − placement` ≈ `(9.92, −14.89, −0.39)`, length 17.90. It is not. Component by
component:

```text
-9.92281 x 0.117320 = -1.16412
14.89284 x 0.117320 =  1.74719
 0.38921 x 0.117320 =  0.04566
```

**`reframed` is the placement itself, same sign, scaled by 0.11732.** Not negated, not subtracted — scaled.

Work it back through `transfer_frame`: `new_pos = inverse(dest.orientation) * (world_pos − dest.origin)`,
and `world_pos = from.origin + from.orientation * local` with `local ≈ 0`. For `new_pos` to come out at
`+0.11732 × dest.origin`, `from.origin` must be `≈ 1.11732 × dest.origin` — a placement pointing the same
way as the planet's and about 20 m out.

**So `from` — the placement of the occupant's OWN frame, `SystemSpace{7}`, in the context doing the
arithmetic — is NOT the identity.** In a context anchored on System 7 it must be exactly the identity: a
realm is its own origin. It is instead being given an orbit-shaped placement roughly the size of a planet's.

That single fact produces everything: the star's own centre reframes to 2.10 m from a planet whose boundary
is 4.16 m, so the planet claims it; a crossing starts to a planet the occupant is nowhere near; the pose is
labelled with that planet's frame; the true destination refuses it; the promote never lands; the occupant
loses its authority and its ticks stop.

NEXT, and it is a two-line check: in the containment scan the context is
`ctx.anchors.get(&owning).unwrap_or(ctx.frames)`. Log which of those two was taken, and log
`frames.placement(pose.frame, tick)` directly. Either the anchors map is handing back a context anchored on
somebody else's realm, or the shard's own frame has been registered in its own moving-child roster — both
are single-line faults with the same symptom, and the log distinguishes them immediately.


## 4o. ⚠⚠ RETRACTION of 4g, 4m and 4n — I logged `.offset()` and read it as the position

§4h(a) already warned that `LatticePos::offset()` is the SUB-CELL RESIDUAL and not the position. I wrote
that warning and then built three more sections on top of the same mistake.

`pose_at` in every containment line is `pose.pos.offset()`. The CELL was never printed. At a fine cell edge
of `1/1024` m, an occupant twenty metres from its star carries roughly twenty thousand cells and an offset
of almost nothing — which is exactly what "pose_at ≈ 0.0009" means. **The occupant was not at the star's
centre. It was out where the test flew it.**

Redo the arithmetic with that: occupant ≈ 20 m from the star in the planet's direction, planet at 17.90 m,
difference ≈ 2.10 m, planet boundary 4.16 m ⇒ `signed_distance = -2.06`, INSIDE. **The containment answer
is correct.** `reframed = 2.0999 m` is correct. There is no scaling bug and no wrong placement. The
"0.11732 factor" is the ratio of two real distances and means nothing.

So §4g ("landed_at is not from_pos − child_at"), §4m ("the reframe returns the wrong number") and §4n ("the
source frame's placement is not the identity") are all WITHDRAWN. Every one of them read a residual as a
position.

**WHAT SURVIVES, and it is still the whole defect:**

- The containment decision is RIGHT: the occupant really is inside that planet, and a crossing really
  should fire.
- The crossing is REFUSED at the destination — *"the arriving pose's frame is neither mine nor one of my
  direct children"* — while the occupant's pose carries `Planet(1164718096683563219)`, which is NOT the
  planet it is inside (`Planet(7701581858760374086)`).
- The promote then defers forever, the occupant loses its authority, and its ticks stop. §4k's chain is
  intact from step 2 onward; only its step 1 was misdiagnosed.

**THE ONE QUESTION, unchanged and now isolated: what stamps a DIFFERENT planet's frame onto the occupant's
pose?** Not the containment answer, which is correct.

**BINDING RULE, from this session's own cost: never log or compare `LatticePos::offset()` as a position.**
Log `delta_m` against the origin, or the cell beside the offset. Three wrong conclusions and two wasted
flights came from that single habit. Every diagnostic line added in this arc must be re-checked for it.


## 4p. TWO SIBLINGS CLAIM ONE POINT — the isolated defect, 2026-08-12

From the same run's per-realm logs, grouped by second:

```text
13:50:24  x20  Planet(7701...)
13:50:25  x11  Planet(7701...)
13:50:26   x2  Planet(1164...)      <-- BOTH, same second
13:50:26   x3  Planet(7701...)
13:50:28   x3  Planet(1164...)
13:50:30   x2  Planet(1164...)
13:51:26   x3  Planet(1164...)
13:52:36   x3  Planet(1164...)
```

and the two claims, each with the placement it was measured against:

```text
Planet(7701...)  placement (-9.92, 14.89,  0.39) |p|=17.90   reframed |r|=2.0999  sd=-2.0607
Planet(1164...)  placement (-25.29, 16.95, -0.71) |p|=30.45   reframed |r|=3.0811  sd=-1.0796
```

Those two planets are **15.55 m apart** and each is **4.16 m** across. One point cannot be 2.10 m from one
and 3.08 m from the other. **Both claims cannot be true, and the containment fold has no way to know that:**
`depth_beats` breaks a same-depth tie by `RealmId` ASC, so `Planet(1164...)` wins purely because its id is
the smaller number — and its frame is what gets stamped on the pose, which is the label the destination then
refuses.

Two claimants also appear in TWO different shard logs (`realm-1002`, `realm-1003`), so the next step must
establish, per shard and per tick, whether one scan really sets two sibling bits at once, or whether two
shards are each answering for the same occupant.

**THE PRECISE NEXT MEASUREMENT:** print the shard's own realm and `pose.universe_tick` on the claim line
(both are absent today), and make the line log the FULL position — `pose.pos.delta_m(origin, tier)` — never
`offset()`. Then one scan's claims can be read together and the impossible pair identified at a single tick.

**IF one scan does set two sibling bits**, the fault is the membership bit itself — a stale hysteretic bit
that never released, which the fold then resolves by id. The cure is in the plan already: containment must
read ONE placement table per parent (Step 1), and a sibling that no longer contains the point must lose its
bit rather than persist and win a tiebreak.


## 4q. NOT A DOUBLE CLAIM — the container ALTERNATES BETWEEN SIBLINGS ACROSS TICKS, 2026-08-12

With the shard and the tick on the claim line, grouped:

```text
  22  scanned_by=System(7)  at_tick=68    Planet(7701...)
   3  scanned_by=System(7)  at_tick=350   Planet(1164...)
   2  scanned_by=System(7)  at_tick=384   Planet(1164...)
   3  scanned_by=System(7)  at_tick=431   Planet(7701...)
   2  scanned_by=System(7)  at_tick=3330  Planet(1164...)
```

**One shard, one claimant per tick — never two at once.** §4p's "two siblings claim one point" was an
artefact of grouping by wall-clock second across different pose ticks. The tiebreak-by-id theory is
therefore NOT the fault either; the fold never had a tie to break.

What the data actually shows is worse and simpler: **the container ALTERNATES between two sibling planets
from tick to tick** — 7701, then 1164, then 1164, then 7701, then 1164. Each flip is a fresh crossing to a
different realm, each stamps that realm's frame on the pose, and they overtake one another: a crossing to
one planet arrives while the pose already carries the other's label, and the destination refuses it exactly
as it should. That is the fight, the stamping of a "wrong" planet's frame, and the stranding, all from one
cause.

**A SECOND FACT ON THE SAME LINES: `at_tick=68` repeats twenty-two times.** The scans keep running while the
occupant's pose stays stamped at one instant. A pose that does not advance is re-decided against placements
that DO advance, so the same stale position is compared against a moving planet — which is precisely how a
container can alternate between two bodies the occupant is not moving between.

**THE ORDER TO CHASE, and it is now narrow:**
1. Why does the occupant's pose tick stop advancing while the shard keeps scanning? A stale pose re-decided
   against live placements is sufficient on its own to produce the alternation.
2. Only if the pose IS advancing: why do two siblings each contain it on alternate ticks when they are
   15.55 m apart and 4.16 m across.

Both are answerable from the same line by adding the shard's OWN current tick beside the pose's tick — the
gap between them IS the staleness, and it is one field.


## 4r. ★★★★★★★ THE ANSWER — the occupant flies further in ONE TICK than a planet is WIDE

With the FULL positions on the claim line (not the sub-cell residual), the two "impossible" alternating
claims resolve completely, and both are CORRECT:

```text
tick 114   pose (-22.56, 19.34,  0.79)   Planet(1164) at (-24.70, 17.90, -0.70)   gap 2.98 m   inside 4.16
tick 117   pose (-18.84, 10.06,  0.63)   Planet(7701) at (-14.95,  9.37,  0.51)   gap 3.94 m   inside 4.16
```

Containment is right at both instants. **What moved is the occupant: 10.0 m in 3 ticks — 167 m/s**, and the
dev cluster's boost speed is `DEV.move_speed = 500 m/s` at `tick_dt = 0.02 s`, i.e. **up to 10 metres of
travel per tick against planets that are 4.16 m across.**

**The occupant is a step larger than the targets.** It passes through one planet, and the next tick it is
inside a different one. Every such tick is a legitimate container change, so every one starts a crossing;
the crossings overtake each other, each stamps a different realm's frame on the pose, a destination
correctly refuses a label that no longer matches, the promote never lands, the occupant loses its authority
and its ticks stop. **The fight, the wrong frame, the refusal, the stranding and the freeze are ALL
downstream of this one physical fact.** No crossing machinery can be correct under it: the containment band
invariant (`width_safe_for`: dead-zone >= per-tick travel x K_SAFETY) is unsatisfiable when the body is
smaller than one tick of travel.

**It is not a crossing-logic defect. It is a WORLD-SCALE defect**, and §4's own warning already named it:
*"⚠ ITS SIZES ARE NOT FINAL … the geometry below still carries the tiny-world numbers."* A star system
150 m across with 4 m planets, flown at 500 m/s, cannot work however the hand-off is written.

**And a 300x orbit slowdown does not fix it** — that slows the PLANETS, not the occupant. It only made the
tests pass by keeping the target still enough to be hit.

### Two legitimate cures, and they are the owner's to choose

1. **The world grows to real scale.** Bodies become large relative to a tick of travel. This is already owed
   ("true astronomical scale" under SL5) and everything else follows from it.
2. **The containing realm caps occupant speed** — which is already the owner's stated warp law: *the
   containing realm controls occupant speed*. A star system whose children are 4 m across must not permit
   10 m per tick inside it. Derived from the realm's own smallest child, never a hand-tuned literal.

Both are the same rule stated from either end: **an occupant may never travel more than a fraction of the
smallest thing it can be inside, per tick.** That is a one-line invariant and it is checkable at boot.


## 4s. ⚠ 4r WITHDRAWN, and the overlap theory with it — owner's objection was correct

The owner rejected the world-scale conclusion: *"it worked before perfectly; on smaller sizes it should work
even better as the magnitude of the error is smaller."* Both parts hold.

- The containment band is SIZED FROM THE OCCUPANT'S SPEED (`for_containment_velocity_safe` widens the dead
  zone past one tick of travel), so "a step bigger than the body" is the case it was built for, not one it
  cannot express. §4r's "unsatisfiable invariant" was wrong.
- The follow-up theory — that a speed-sized band makes two SIBLING bands overlap, so an occupant is a member
  of two at once and `depth_beats` picks the lower realm id — was then MEASURED and is also wrong. Logging
  MEMBERSHIP (what the fold reads) rather than "geometrically inside": **no tick has two different sibling
  members.** The repeated counts are repeated scans of one tick, not two claimants.

**So containment is clean, at every level tested: one member, one container, correct geometry, no tie.**

### What still stands, and it is now the only thing left

1. The occupant legitimately passes near several planets at flight speed, so the container legitimately
   CHANGES from one sibling to another within a few ticks.
2. Each change starts a crossing.
3. A destination refuses with *"the arriving pose's frame is neither mine nor one of my direct children"*
   while the pose carries a DIFFERENT planet's frame.
4. The promote defers forever, the occupant has no live authority, its ticks stop.

Nothing in 1 is a fault. So the defect is that **successive crossings RACE**: one is still in flight when the
next begins, and the pose is stamped for the second while the first is still expecting the first. The
existing guard for exactly this is `should_rehome`'s `k_dwell` — a post-commit anti-thrash cooldown that
suppresses a new re-home until the last one has settled. If it is zero, unwired, or measured against the
wrong clock, back-to-back crossings are free to overtake each other.

**NEXT: read what supplies `since_commit` and `k_dwell` on the live path, and log both at the moment a
crossing is started.** If the cooldown is not actually gating, that is the defect, it is one value, and it
explains every surviving observation without contradicting any measurement.


## 4t. THE SURVIVING CANDIDATE — a SIBLING-TO-SIBLING crossing, which SL2 forbids

Checked the anti-thrash path rather than assuming it: `k_dwell = 5` ticks by default, `since_commit` comes
from `CrossingState.last_commit_tick`, and a durable `RequestInFlight` latch already stops a second attempt
starting while one is in flight. So duplicate storms are guarded, and the five fences seen in one second are
five crossings that COMMITTED, not five retries. The cooldown theory is therefore also not the fault.

**What the refusal message actually implies.** The destination said *"the arriving pose's frame is neither
mine nor one of my direct children"*, and the pose carried a planet's frame. A destination can always place
a pose from its own PARENT (the parent is not in its context, but the parent converts before shipping) and
from its own children. The one thing it cannot place is a pose stated in a SIBLING's frame — and a sibling's
frame is exactly what a planet's frame is, to another planet.

SL2 forbids that hop outright: *travel is always out into the shared parent and in again, never
sibling-to-sibling.* So either

- a shard is deciding a SIBLING is the container (it should be structurally unable to: a planet's region set
  is itself, its ancestors and its own children — never a sibling), or
- the STAR is firing a crossing to planet B for an occupant that is already authoritative on planet A, using
  a pose it still holds in A's frame from before the first hand-off.

The second fits every measurement: the star legitimately sees the occupant's last known position inside B,
it still holds a retained ghost for it, and its crossing carries the pose as it last had it — labelled for A.

**THE NEXT MEASUREMENT, and it is one line at the crossing start:** log the SOURCE realm, the occupant's
pose frame, the chosen destination, and whether the source still holds authority, at
`fan_out_crossing`. A source firing a crossing for an entity it has already handed away, or a destination
that is a sibling of the pose's frame, will be visible immediately — and both are structural faults with
one-line guards.


## 4u. THE CODE CHECK, 2026-08-12 — every load-bearing claim of this document verified against code

Nine independent readers, code-only evidence (comments and this document explicitly distrusted), each
claim carrying file:line. Spot-checked by hand afterwards. Verdicts that CHANGE the plan first.

### Refuted or sharpened

1. **Step 2 is NOT complete — the stored centre survives.** `RealmRegion.center` exists
   (geometry.rs:931-937), is authored ZERO for every orbiting realm (worldgen.rs:214-219, written at
   :251, and PINNED by a test at worldgen.rs:3163-3165), and has production readers: the outline lane's
   verbatim fallback when `child_placements` lacks a row (stub.rs:7762-7774), `frame_context`'s static
   arm (stub.rs:1321-1329), `place_child`'s static arm (stub.rs:1431-1456), the boot fence
   (geometry.rs:1161-1167), the `VD_REALM_BOUNDARIES` adapter (bins/lib.rs:2897), and the client's boot
   scene (client/realm_scene.rs:234-249, reached from bins client.rs:124). Only the login-descent READER
   died; the second answer's storage and a live fallback that serves it verbatim remain.

2. **§4t's candidate B is dead — a source cannot re-fire for a handed-away entity.** The scan evaluates
   only Owned dots (`simulates()` filter, stub.rs:4390; authority.rs:93-95 is Owned-only); the retained
   post-commit Ghost is never scanned and cannot accept input (stub.rs:2875-2878); the latch is
   positively cleared in the same demote handler that ghosts the dot (stub.rs:3081-3133); the redrive
   iterates only held latches (stub.rs:4919).

3. **The dwell cannot gate committed crossings at all.** `CrossingState` is per-shard RAM — no
   Serialize, never on the wire, evicted by `retain_live` (stub.rs:4347). Every commit moves the entity
   to a shard where `since_commit = None`, so `should_rehome` fires on the first differing tick
   (geometry.rs:996-1001). `last_commit_tick` is armed at request-EMIT, not commit (stub.rs:4837/4865;
   the field's doc comment lies). §4t's "cooldown is not the fault" stands, but for this stronger
   reason: k_dwell only throttles repeat attempts from ONE shard. It is also hardcoded, not
   env-tunable (shard.rs:278). DEFERRED.md:3315 already records the non-survival.

4. **THE STRAND IS STRUCTURAL, and it is the permanent freeze.** `EmitCrossing` ships only post-CAS
   (saga.rs:861-874). On a destination-side refusal (`place_arriving_pose` → `UnplaceableArrival`,
   refusal at stub.rs:4140-4153): the dest never adopts and the promote defers forever
   (stub.rs:3357-3372); the source demotes unconditionally to Ghost and acks (stub.rs:3081-3096,
   3147-3155); and post-commit saga arms NEVER abort — "Post-commit timeouts retry the same command
   (idempotent), never abort" (saga.rs:1066-1078); only `Swapping` re-emits the crossing, `Promoting`
   re-emits Promote alone (saga.rs:1187-1206). Both shards healthy ⇒ the entity is permanently
   un-simulated on both sides. No timeout returns authority. The pre-commit refusal, by contrast,
   self-heals: a failed flush parks the saga in `Freezing`, `FreezeTimeout` aborts, the thaw restores
   the source (saga.rs:857, 1365-1384). The asymmetry is the whole defect surface.

5. **HOW a foreign frame ships — the verbatim arm, and who flushes.** `flush_pose_for_dest`'s
   upward/sideways arm ships the stored pose VERBATIM with whatever frame it carries
   (stub.rs:4003-4006); `flush_anchor_not_own` is warn-AND-SHIP, not a refusal — and it is
   roster-resolved (stub.rs:3992 `unwrap_or(config.realm)`), so a frame from a realm this shard has NO
   region for ships with no signal at all. The destination accepts exactly {its own frame} ∪ {its
   direct children's frames} — never its parent's (frame_context registers no parent placement,
   stub.rs:1308-1313; frame.rs:244-246's contrary claim is stale). The saga couriers the pose verbatim
   (saga_runtime.rs:913-937) and resolves the FLUSH SHARD from the directory head at request-handling
   time (saga_runtime.rs:1629) — not from the shard whose scan fired the request. So a request that
   outlives its own commit context is flushed from a DIFFERENT realm's shard and ships that realm's
   frame to a destination that cannot place it. `rebind_pose_to_dest` (§4k's candidate) runs NOWHERE in
   production — test-only; the comments naming it live are stale (wire/intershard.rs:1031/1049/1068,
   stub.rs:816, shard.rs:131).

6. **A SECOND stranding path needs no refusal.** An unresolved dest (realm or session head not yet
   resolved) is counted and dropped with no saga and no abort reply — the code's own note: "the
   source-latch-clearing abort-reply is 3f-D" (saga_runtime.rs:1640-1646) — while the source latch
   suppresses every further attempt (stub.rs:4848) and the only re-emitter is disabled on the live path
   (`request_ttl_ticks: 0`, shard.rs:281). The entity keeps simulating but can never cross again from
   that shard. This fits §4f's "reaches 0.43 m and is NEVER re-homed" with no frame mislabel needed.

7. **The containment band is NOT speed-sized in production.** stub.rs:4705-4706 ("the band is sized
   from the occupant's SPEED") is a stale comment. The shipped band is static inset 1 m / outset 2 m,
   built with v_rel = 0, dt = 1 ("the widening is inert", worldgen.rs:1223-1233; consts at :43-45).
   Containment is point-sampled, never swept (stub.rs:4662; `prev_offset` is write-only). Movement is
   clamped PER AXIS with no vector normalization: boost on three axes is 500·√3 ≈ 866 m/s ≈ 17.3 m/tick
   (kinematics.rs:76-82, stub.rs:2922-2927) against 4.16 m planets and a 3 m dead zone —
   `width_safe_for` at the flown speed would demand ≥ 20 m and nothing ever checks it. NO per-realm
   speed cap exists anywhere; the warp law is unimplemented (`time_multiplier` is dormant — no
   production setter of either env var). §4s's geometric measurements stand; its reassurance about the
   band does not.

8. **Durable pose stamps advance only on input.** stub.rs:2946 (`integrate`) is the ONLY re-stamp of a
   durable dot's pose tick; the scan and the flush resolve MOVING placements at the pose's stamp
   (frame.rs:135-140), so an input-idle occupant is measured against placements frozen at its
   last-input tick, indefinitely. Transients are re-stamped every tick (stub.rs:5001-5007); durables
   are not. This is §4q's "pose stamped at one instant, re-decided against placements that DO advance",
   now with its mechanism.

9. **The fixtures over-claim.** Only the inner-planet test re-aims at the live position
   (rlm_demand_login.rs:671-736); BOTH round-trip tests still chase the fixed epoch (:868, :1100 — their
   own comments say so). The re-aim math is exact only at slowdown = 1: the fixture derives elements
   from an UNMODIFIED config while the shard divides the star mass by K² before deriving movers
   (bins/lib.rs:2639-2644) — at the gated 300× default the fixture chases a phantom orbiting 300×
   faster. NO gate runs ANY crossing at flight speed (justfile:247-248, :253; no env var set anywhere;
   §4j's "this fixture is now the gate" is not wired). The env knob is set with a non-panic-safe
   `set_var`: a mid-test failure leaks 300× into the full-speed tests that FOLLOW in the same binary
   (`--test-threads=1` name order), making the parked-ship motion assert fail spuriously. Full-speed
   gate coverage exists ONLY for non-crossing behaviour (the parked and spin-up-ahead tests).

### Confirmed (landed as this document describes)

- The descent is fully deleted from production (all nine names; remaining hits are comments, docs, and
  one unrelated test helper that reuses a name). The home registry, `default_home_realm` (lineage-only),
  `home_placement`-as-lookup, `resolve_homes`, and the VD_SPAWN_POSES semantics are as stated.
- The restatement fix: ONE `restate_rows_in_frame` serves both lanes; the direct lane groups observers
  by pose frame; identity for own-frame observers; `realm_rows_unplaceable_observer` counts (per frame
  group per tick, not per observer); the recipient's own row is dropped — POSE lane only; the SHAPE lane
  deliberately keeps and re-centres the recipient's own outline (stub.rs:7716-7745).
- The anchor fix: `from_realm = config.realm` unconditionally (stub.rs:4002); the counter exists; a
  shard's region set is self ∪ ancestors ∪ direct children (worldgen.rs:990-1007).
- Movers are production (shard.rs:120-126, 305-308); the worldgen comment at :195-200 still claims "no
  production producer" over a `#[allow(dead_code)]` marker while the producer sits at :637-643 — §6's
  warning stands, unfixed.
- The gateway derives no POSITION from its world copy (names only) — but it still HOLDS the full
  WorldView (positions one line away, convention-guarded; its own comment at gateway.rs:2007-2010 owes
  the retirement), and `SeedInjectorConfig::default()` constructs a hand-placed walk-scale world
  (gateway.rs:157-176) — an SL5 foot-gun with no production caller found.
- Fences bump at commit only (`cas_next` from `commit_cas`/`ReHomeCommit`; aborts are fence-neutral;
  `abort_cas` has no production producer) — §4t's "five fences = five commits" holds, with the caveat
  that recovery re-homes also commit.
- The in-flight latch is per ENTITY per SHARD (`BTreeMap<EntityId, TransferId>`, stub.rs:1004): while
  latched, a second crossing cannot start from the SAME shard; crossings that raced must have been
  interleaved COMMITS (each new owner starts unlatched, see refutation 3) or paths that bypass the
  detector (TTL redrive — inert; dead-owner reaper and D-37 forward re-home — liveness-chosen
  destinations, no should_rehome).

### Latent items surfaced for the machinery audit (not this fix)

Transient adopt has NO receiver-side conversion (stub.rs:5145; the known-gap note at :5056-5060). The
ghost feed writes the DEST-frame pose verbatim into the source's retained ghost (stub.rs:3689-3743,
5889-5895) — the star ends up holding a planet-framed pose by design, consumed only by the emit lane's
counted degrade (`entity_rows_foreign_labelled`). The client's `RealmView.placements` is insert-only
(realm_view.rs:85-89) and `overlaid_at` prefers a streamed track forever (realm_scene.rs:403-426): a
realm entered AFTER having been streamed as a mover draws frozen at its stale pre-crossing centre in the
parent's space, and C4's own-row drop guarantees no row ever heals it — the likely residue of "I land
outside the planet" even once crossings are fixed. The `granted & !departing` predicate family omits
`simulates()` (stub.rs:4064-4066, 4224-4226, 3791-3793) — safe only while flush routing follows the
directory head. The demote latch-clear is id-blind (stub.rs:3131). The M-2 boot assert described at
worldgen.rs:1244-1246 does not exist. Stale comments measured wrong this pass, to fix with the work:
worldgen.rs:195-200 (+`dead_code` at :199), stub.rs:4705-4706, stub.rs:1356-1360 (gateway converts —
it forwards verbatim), stub.rs:3855-3856 (ghost feed is a second ingress), stub.rs:3964-3965 (doc vs
body), shard.rs:275-277 (detector "INERT in production" — it is live), gateway.rs:101 (`spawn_poses`),
frame.rs:244-246 (parent placements), wire/intershard.rs:1031/1049/1068 (rebind), rlm.rs:1405/1440
(descent named live).

### The corrected order, from here

1. **The reproduction stays first** (§4f/§4j fixture, `VD_TEST_ORBIT_SLOWDOWN=1`), with FOUR log
   points, not one: `fan_out_crossing` (own realm, pose label, container, to_realm, latch state);
   `handle_crossing_request` (requesting shard vs the directory head it resolves; unresolved drops);
   `flush_pose_for_dest` (arm taken + the frame actually shipped); the dest refusal line. The race to
   catch: a request FLUSHED from a different shard than the one whose scan fired it, and a `to_realm`
   that is not the flush shard's direct child.
2. **The cure candidates — each already a verified hole; the measurement decides which one(s) the live
   defect convicts:** (a) refuse-at-flush instead of warn-and-ship: a pose that cannot be stated for the
   dest fails the flush, which parks the saga PRE-commit in Freezing and self-heals via
   FreezeTimeout→abort→thaw — converting the permanent strand into the recovery that already works;
   (b) the 3f-D abort-reply (or arming the TTL) so an unresolved-dest drop cannot latch an entity
   forever; (c) a dwell that survives the hand-off, only if the flap needs damping after (a)+(b);
   (d) the OWNER's standing decision on speed versus band — the warp law (containing realm caps
   occupant speed) or swept/speed-sized containment; nothing enforces either today.
3. **The gate gains a flight-speed crossing.** Fixture aim derived from the slowdown-adjusted config;
   panic-safe env guard; the two epoch-fixed tests re-aimed.


## 4v. ★★ STAGE A RUN 1 — the four log points fired; the chain is measured end to end (2026-08-12)

`VD_TEST_ORBIT_SLOWDOWN=1`, the re-aiming inner-planet test, log-only instrumentation (no behaviour
change). The test FAILED as required (151 s, no crossing held, closest approach 5.96 m vs 4.16 m).
Run dir: `vd-rlm-demand-innerrehome-72028`. Seven sagas, fences 1→8, all within 43 s; then 100 s of
total silence. Every number below is from the run's own logs.

### (1) THE MISLANDING IS THE DECISION-TO-FLUSH GAP — not arithmetic, not frames

Every crossing INTO a planet shipped a pose that was already far outside it, because the pose is
RE-READ LIVE at flush, several ticks after the containment decision, and the player keeps integrating
input meanwhile (the freeze is a gateway round-trip away):

```text
fence 1  DECIDED at_tick=68 (inside the band)  →  FLUSH at_tick=73:
         from_pos_m len 30.0, child_at len 17.9, landed_m len 12.15   — 12.15 m outside a 4.16 m planet
fence 4  DECIDED at_tick=1121 → FLUSH at_tick=1128: landed 23.00 m out
fence 7  DECIDED at_tick=2213 → FLUSH at_tick=2218: landed 20.72 m out
```

The subtraction itself is CORRECT to the digit (from_pos − child_at = landed, verified per line). The
arrival accepts as given (its log agrees), the dest's scan instantly finds the player outside → the
bounce-back = the flap = the owner's "I re-homed onto the planet and I am outside it". The bounce-OUT
then ships with the SAME gap mirrored: by ITS flush the chasing player was back INSIDE the planet
(pos_m len 2.37 at fence 2's flush), so the star re-adopts an inside-the-planet position and decides IN
again. The ping-pong is two staleness windows facing each other.

### (2) THE LABEL CORRUPTION IS REAL, AND THE GHOST FEED IS THE WRITER

Measured on the star's own scan after each bounce-back:

```text
scanned_by=System(7)  owning=Planet(7701…)  pose_frame=PlanetCentered{7701…}  at_tick=96   (fence 3)
scanned_by=System(7)  owning=Planet(1164…)  pose_frame=PlanetCentered{1164…}  at_tick=1153 (fence 6)
```

The star OWNS a dot whose pose is verbatim planet-frame. Mechanism (all verified pieces): after the
hand-off down, the new owner feeds the star's retained ghost with its OWN-frame poses and
`refresh_source_ghost` writes them VERBATIM (the known no-conversion gap); the return crossing's
converted arrival pose is then superseded by a later ghost delta before the promote flips the ghost
back to Owned — so the star ends up owning a planet-labelled pose. The scan anchored on that label can
place NOTHING (a context anchored on a child has no rows), members come up empty, `outward_dest` of the
label = System(7) — and the star fires a **SYSTEM(7)→SYSTEM(7) SELF-CROSSING** (fences 3 and 6, both in
the orchestrator log with from==to), a full saga that burns a fence and happens to LAUNDER the label
back through `place_arriving_pose`. §4k's refused sibling-frame hand-off is this same corrupted label
shipped VERBATIM to a third realm instead of laundered — which of the two you get is a race.

### (3) THE TERMINAL STRAND — the freeze, reproduced, with its exact shape

```text
14:59:59.040  star DECIDED  → fence-7 request (System→Planet 1164)
14:59:59.175  star HAND-OFF DOWN (landed 20.72 m out)
14:59:59.234  planet 1164 ARRIVAL accepted (its log's LAST LINE, ever)
14:59:59.255  star: "held at fence fence-8 — source self-demoted to a retained Ghost" (ITS last line)
              … 100 seconds: no scan decision on 1003, no saga, no abort, no retry, anywhere
```

The commit went through; the star demoted; the destination accepted the pose bytes but the player
NEVER re-entered simulation there — no scan line ever again, which under the Owned-only scan filter
means no Owned dot. The dest's retained ghost from the previous visit had been reaped by the 76-tick
hand-off hold TTL ~21 s earlier, so the crossing envelope had no target dot to write; the arrival's
"no target — buffered/dropped" path LOGS NOTHING (a fifth log point is owed); the promote defers
forever; post-commit arms never abort (§4u refutation 4). Meanwhile the CLIENT still sees ticks
advance (the realm feed rides the star sub the gateway kept open) and its own position OSCILLATING
14→44 m with the planets' period — a stale pose composed against moving placements. So the lived
symptom is exactly the owner's: **the world keeps moving; you are stuck; the location label never
changes; nothing recovers.** The silent unresolved-drop (3f-D) did NOT fire this run — a verified hole,
but not this run's killer.

## 4w. STAGE B — what landed, and what each re-run measured (2026-08-12)

Owner decisions: cure the decision-to-flush gap by RE-VALIDATION AT FLUSH (not a source freeze); skip
the interim ghost-feed guard and go straight to Step 5 (the SL7 lane deletion).

**B1 — flush re-validation (landed, vd-sim 457 green).** `flush_pose_for_dest` now refuses a hand-off
whose subject the flush-time pose no longer supports: the downward arm re-asks the destination's own
band (`member(true, sd)`) on the CONVERTED pose (`flush_stale_entry`); the upward arm refuses while the
shard's own region still holds the re-read pose (`flush_stale_exit`); a self-crossing skips the exit
check; a roster with no own region ships as before (the guard never invents a refusal). A refusal ships
no `SourceFlushed` ⇒ the saga parks in Freezing ⇒ FreezeTimeout aborts ⇒ the source keeps authority —
the pre-commit path that already self-heals, at the 24-tick abort budget.

**B2 — post-commit recovery (landed, vd-sim + vd-node green).** (a) The `Promoting` re-drive re-emits
the CROSSING beside the Promote (the promote alone could never cure an undrained crossing; the journal
dedups the re-emit — the D-37-2d template). (b) An `OpenInputSlot` arriving before the realm lease is
BUFFERED (`PendingInputSlots`) and drained at the lease-affirm None→Some flip through the identical
adopt path (`input_slots_drained`); it was a permanent drop — the exact fence-7 strand. The D-8 input
half (the gateway's take-drained post-marker buffer) remains deferred; the ADOPT no longer dies.

**RUN 2 (B1+B2 in): both fixes measured live.** One `flush_stale_exit` refusal fired, the saga aborted
and re-attempted cleanly (attempt 1), and the crossing then COMMITTED — location flipped to the chased
planet at 2.22 m closest approach. No mislanding, no flap, no strand: every §4v symptom is gone. The
test then failed LATER, in its cascade window: the run-2 client sat on the planet only briefly (a
WalkTo re-aimed in the OLD frame walked it back out), and the observation cascade did not accumulate
20 frames in the window — the remaining failure is FIXTURE flying, not crossing machinery.

**RUNS 3 and 4 (fixture at flight speed): the chase itself was the obstacle.** Chasing the live centre
(run 3) grazes: 15 attempts, every one correctly refused at flush (the player crossed the 8.3 m shell
in under the decision-to-flush latency at ~500 m/s), best approach 0.39 m, zero commits — the guard
telling the truth: a 150 m/s+ fly-through is not an entry. A led aim (run 4) alone did not cure it.
The fixture now flies a RENDEZVOUS: to where the planet WILL BE (~200 ticks out), then PARKS and lets
the planet sweep over the stationary ship at its own ~6 m/s — which is how a pilot actually lands.

**B4 machinery fix — the parked ship was invisible to the sweep (§4u refutation 8 cured).** A durable
pose's stamp advanced only on input, and BOTH the scan and the flush resolve moving-child placements
at the pose's stamp — so a planet sweeping over a parked (input-idle) ship was measured at the stamp's
old instant and never contained it. New `readvance_dots` system: every simulating dot re-stamps to the
current tick each tick (position untouched — `Frozen` continuity; the transient tier already did this;
HR2 one machinery). Scheduled after `process_inbound`, before the scan/flush group.

**B4 gate.** `just rlm-demand-login-flight` (the inner test at slowdown 1) added to the merge gate.
The fixture derives its aim through `vd_bins::boot_regions_and_movers` (the slowdown-adjusted elements
the star actually authors — the old direct read aimed at a 300×-faster phantom), and the slowdown knob
is now a panic-safe Drop guard (a mid-test failure no longer leaks 300× into the full-speed tests that
follow). The two round-trip tests still gate at the 300× default; their re-aim is owed with Step 5.

### 4w-final — the Stage-B verdicts, measured (2026-08-12, end of arc)

- **★ THE FLIGHT-SPEED CROSSING PASSES** (run 8, 10.4 s): rendezvous → park → the planet sweeps
  over the ship → the crossing commits and HOLDS (0.89 m approach) → the cascade keeps the whole
  system orbiting around the crossed-in player. `just rlm-demand-login-flight` gates it forever.
- **★ THE REPEATED-ROUND-TRIP FREEZE IS CURED**: 5 Planet↔System cycles, no freeze, tick live
  every cycle. The final missing piece was the GHOST-FEED FRAME POISON (§4v fact 2): a fed
  dest-frame pose written into the retained ghost that a return promote resurrects — measured
  live as a Planet→Planet launder self-crossing whose PARENT-frame pose the destination rightly
  refuses forever (the cycle-1 freeze). Cure: `refresh_source_ghost` refuses a fed pose whose
  frame differs from the dot's own (counted on `ghost_refresh_stale`) — Step 5 slice F's semantic
  core landed early, no wire change; the lane itself still dies in slice F (owner-approved).
  The return round-trip test also passes (rides the realm, 0.88 m gap; neighbour feed live).
- **Steering lesson, twice-measured**: the Move input is sticky — cut the throttle before any
  park, or the ship runs away ~1 km. And the walk controller's arrival is only honest when the
  approach cannot overshoot: the brake must be sized to the DELIVERED-pose lag (~4 steps), not to
  one step.
- **`a_flying_occupant_streams_a_culled_planet…` is BROKEN AT HEAD, pre-existing** (measured with
  Stage B stashed: fails at HEAD too, earlier in its body — the un-braked walk exits the system).
  Commit 32fcc5c's wider visibility (8°→1.5°) makes every planet of a system visible from its
  star (limit ~318 m > the 142 m outer orbit; HEAD's own state shows all five planets streamed at
  login), so "an outer planet culled at login" is unsatisfiable in THE world. The scenario is not
  deleted; it needs a REWRITE against the multi-star geometry (the culled realm is now a
  NEIGHBOUR SYSTEM on the ~12 km ring) — ledgered in DEFERRED.md.

## 4x. THE OWNER FLEW THE STAGE-B TREE — the picture is unchanged, and the reason is MEASURED (2026-08-12)

The owner ran `scripts/demand-visual-run.sh` and reports every symptom identical. The script deletes
its logs at teardown, so the flight itself cannot be read back. What CAN be measured: the
drawing-level gate `the_box_and_the_thing_standing_in_it_draw_at_one_point` (§4c's own gate) STILL
FAILS — in 600 ticks the client never twice holds both render feeds for the occupied moving realm at
one instant. §4c's open question ("does the client draw a realm it occupies from the outline lane
when no per-tick row arrives?") was never settled, and §4u's client findings answer it: the client
folds frames from ANY held sub into one track per entity (the per-entity picker was deliberately
deleted), `RealmView.placements` is insert-only (no eviction, ever), and `overlaid_at` prefers a
streamed track forever — so a realm entered after having been streamed as a mover DRAWS AT ITS STALE
PRE-CROSSING CENTRE, in the OLD home's space, indefinitely; the own-row drop guarantees no row heals
it. The machinery cures of §4w are real and gated; THE PICTURE the pilot sees is composed by the
client, and the client's crossing composition is the unfixed half. A second, smaller factor: a manual
approach at warp (SHIFT, 500 m/s) now meets the B1 flush re-validation exactly as the fixture did —
refusals, not commits — until the pilot slows or parks ahead; the game-side cure is the owner's
standing warp law (the containing realm caps occupant speed), still an open owner decision (§4v d).

**LANDED 2026-08-12 (§4y below). CONSEQUENCE — the order changed again.** The client crossing-render
slice came BEFORE Step 5:
(1) at the scene swap (promote / SubscriptionReady) the client EVICTS the old home's tracks — every
stored row is in the OLD space and mixing spaces is the jitter; this is the "one feed wins" rule,
whose measure-first condition is now MET by this measurement; (2) the realm the player occupies
draws from its OWN outline at the player's origin (SL3) whenever no per-tick row exists for it;
(3) the §4c gate flips green and joins the merge gate beside the flight-speed crossing. Step 5's
slices follow.

## 4y. THE CROSSING-RENDER SLICE — landed, reviewed, and it cured one more pre-existing strand (2026-08-12)

**The client (one space at a time):** realm rows fold only when stated in the space the avatar
stands in (`foreign_space_rows` counts the old feed's grace-window stragglers); the same rule on the
entity lane with the OWN avatar exempt (its new-frame row IS the crossing signal); at the delivered
own-frame flip the client forgets every stored placement (`forget_space`) — the stale planet box
becomes unrepresentable. The interpolation track gained the FRAME-STABILITY guard: a non-newer row
in another frame is ignored, never folded — without it the demoted source's retained-ghost rows
flipped the location backward per datagram (the review's CRITICAL find, caught before it flew).
The harness client mirrors every rule, gated on Apply exactly as the shipped client.

**The server (one space per observer):** the realm lane's direct-emit observers are grouped by the
SAME space the entity lift delivers — a co-hosted child-stander now receives its realm's row LIVE in
the one space its own pose arrives in, which closed D-RLM-17 by construction (the box keeps turning
under a standing player, gap 0).

**The flush converts along the full hosted chain** (the review chase's biggest catch, and it
predates this slice): the §4i anchor fix had silently broken every hand-off spanning more than one
level on a co-hosting shard — the one-level child lookup sent grandchild descents to the verbatim
arm (where B1's exit re-check rightly refused, forever) and could not express the reverse lift at
all. `conversion_path` now walks roster parent pointers through the nearest common ancestor: each
ascending link's parent ADDS the placement it authors, each descending link's SUBTRACTS — SL1 in
both directions — under the AUTHORSHIP CONSTRAINT (only the shard's own realm and its roster
descendants may convert; a roster ancestor is a name, never arithmetic — its mover children's
stored centres are ZERO, the §1 defect). `crossing_same_node_e2e` 3/3 green again.

**The gate** (`the_box_and_the_thing_standing_in_it_draw_at_one_point`, 19/19 suite): half (1) — a
rider co-moving beside the moving area and the area's streamed box draw in ONE space (gap exactly
0, box genuinely turning, radius pinned to the observer's space); half (2) — the test's name,
literally: the occupant STANDS IN the moving area and box + occupant draw at one point, live.

**Verified state at slice close:** every vd-tests suite green; client 141 / harness 96 / sim 457 /
node 184 green; the flight-speed crossing gate green with all client rules active; clippy 0; the
sole red is the pre-arc baseline failure (`dual_cluster_crossing_smoke`, §4b's known list). Honest
residuals, ledgered in the code: a one-tick own-frame flicker under datagram loss during the grace
window (bounded, self-healing); a fast A→B→A re-entry admits one same-frame straggler for ≤ one
feed period; the own-track eviction disables the filter until re-delivery (pre-slice behaviour).

## 4z. STEP 5 SLICES B+C LANDED TOGETHER — and the neighbour-system gap became a green gate (2026-08-13)

The narrative record for this landing lives in `step5_sl7_lane_deletion.md` (the implementation
block dated 2026-08-13) and DEFERRED D-RLM-16 (now 🟩). The one finding that belongs in THIS ledger,
because it is a measured root cause of the §4x family: **the parent-resolution HeadRead sat BELOW
`aoi_decide`'s zero-occupant early-return.** A realm nobody had ever entered could therefore never
resolve its parent node, and every up-lane it owns (occupant relay, ChildLive bit, RealmObservation
rows, RealmShapeObservation outlines) was structurally mute — which is exactly why the owner saw the
EXITED system keep working (its parent had been resolved while it was occupied, and stayed cached)
while an APPROACHED system's interior never existed client-side. Moved above the Empty return: an
armed realm owes its parent observation even while empty (SL3 — visibility is the spin-up trigger).
Gate: `a_flying_occupant_streams_a_neighbour_system_in_ahead_then_the_vacated_realm_is_reaped`
(also the honest D-RLM-16 rewrite). PROTO_MINOR → 11.

### What Stage B must cure, in the order the evidence ranks them (§4v original ranking)

1. **The decision-to-flush gap.** The pose that ships must be the pose the decision was made on — or
   the source must stop integrating the subject the moment the crossing latches (the freeze applied at
   the source's own tick, not after a gateway round trip). This alone removes the mislanding AND the
   flap. Design decision owed to the owner.
2. **Post-commit recovery.** A committed crossing whose dest cannot materialize the dot must recover:
   re-emit the crossing on the Promoting re-drive (it re-emits only Promote today), or abort-to-re-home
   after a budget. Plus the fifth log point + counter on the dest's no-target-dot path.
3. **The ghost feed must stop writing foreign frames** into a dot that can be promoted back to Owned —
   or the promote must re-place the pose. Note this lane is already ordered DELETED by Q4/SL7 (Step 5);
   the corruption is more ammunition for doing it, not for patching it.
4. The latch and dwell behaved exactly as designed this run (suppress lines at k_dwell spacing) — no
   cure needed there for THIS chain.

## 4aa. STEP 5 SLICE D LANDED — the per-occupant lanes are dead (2026-08-13)

The lanes §4 (Q4) ordered deleted are now deleted, not dormant. `OccupantInterest` and
`ProxySceneSet` are wire TOMBSTONES (PROTO_MINOR → 12; positional discriminants kept, received
frames count `undecodable`); the up-relay, `RetainedOccupants`, `proxy_observer`, the parity gauge
and every stat only that machinery fed are gone from production. Point 3 of the Stage B cure list
above is therefore discharged for THIS lane family: no per-occupant ghost feed exists to write a
foreign frame (the §4u corruption's remaining writer, `refresh_source_ghost`, is the ENTITY lane —
it dies in slice F). Liveness across the hand-off window rides the ChildLive bit + the `speaks_for`
carry alone, and the chain scenarios in `frame_conversion_e2e` now MEASURE SL2 (no pose above the
owner) and SL7 recursion (a planet live on its child's bit alone) instead of exercising the breach.
Narrative record: `step5_sl7_lane_deletion.md` (slice D block). Remaining: slice E (owner sign-off
owed — entity-lane tombstone), slice F (SpawnV2 cutover, `Delta` death), then the Stage C audit.
