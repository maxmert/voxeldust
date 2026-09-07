# Verdict — the law refuter — report 10, the law audit and the stale-text register

**Date:** 2026-09-07. **Lens:** the cross-cutting law audit.
**Target:** `docs/investigation/2026-09-07/10_law_audit_stale_register.md`.
**Verdict: REFUTED.** Two load-bearing claims fail. One states a fact about the code that is false.
One authorises new data across a realm boundary without the ask the law demands.

Every code claim in the target report went through `grep` and `sed` on this tree. Most of them hold.
The refutation is narrow and it is real.

---

## 1. What holds

I re-measured the report's fact table (its §1). These rows are correct on this tree today:

- `InterShardFlow` holds 44 arms. I counted them with `awk` over the enum body from
  `crates/wire/src/intershard.rs:124` to the closing brace. The 30-arm ceiling is passed.
- The saga step ids run 7 to 18. Id 19 is free (`crates/wire/src/intershard.rs:65-115`).
- `ChannelKey` does not exist. `grep -rn ChannelKey crates` returns nothing.
- The transient re-clamp is deleted (`crates/sim/src/stub/transient.rs:229-236`). A rock that
  falls into a moon's realm keeps the speed it arrived with.
- `pin_abs` and `anchor_epoch` are removed (`crates/wire/src/version.rs:67-79`). No realm can
  reach an absolute position.
- The parent-authored marker no longer ships. `current_bodies` states the realm's own look and
  nothing about its children (`crates/sim/src/stub/window.rs:497-503`). The `BodyStmt::Marker`
  arm and the gateway's `marker_of` store stay as a decode path with tests only
  (`crates/connection-plane/src/window.rs:283-290`). The report reads this correctly.
- The galaxy tick figure is a real measurement: 279,380 direct children, 20 µs, 5 candidates
  (`docs/design/DEFERRED.md:5702-5705`, release, probe
  `measure_the_galaxy_shards_tick_against_its_census`).
- Neither `rapier3d` nor `noise` is in `Cargo.lock` (0 hits each). `noise = "=0.9.0"` sits in the
  workspace table at `Cargo.toml:78` and no crate names it.
- The report's marking discipline is good. Every number carries MEASURED or ESTIMATED, and the
  report says UNMEASURED where it could not measure. It never argues "no drift" in place of a gate.

**Example, in the game's words.** A pilot flies a hull into the galaxy. The galaxy's shard no longer
draws one point of light per star system it holds. Each system in reach wakes and draws itself. The
report states this and the code agrees.

---

## 2. Finding 1 — WRONG. "No cap exists" is false. The walk is capped today.

**The claim.** The one-way door table, door 42: *"42 (max flight speed) | REFUSED as a door; no cap
exists (M-D)"*.

**The code.** A cap exists and it runs on the live path.

- `crates/sim/src/stub/dot.rs:465` — `walk()` calls `governed_ceiling_for_frame` for the occupant's
  own frame every tick.
- `crates/sim/src/stub/dot.rs:474` — the result scales the occupant's commanded axes through
  `flight::throttle_axes_scale`. The step at `dot.rs:479-482` multiplies by that scale.
- `crates/core/src/flight.rs:93` `realm_speed_cap_mps`, `:105` `approach_ceiling_mps`, `:118`
  `ramp_cap_mps` are all live functions, called from `dot.rs:372, 397, 414, 420, 457`.

**The law.** The 2026-08-27 movement answers delete the governor. The 2026-09-05 suit ruling S3
deletes the approach ceiling on the player's stick by name
(`docs/design/owner_decisions_2026-09-05_suit.md:44-46`). The same ruling's S6 then DEFERS the
deletion: *"Until the suit lands, the dot's walk keeps its boot-time ramp as a PLACEHOLDER … This is
wrong by S3 and deferred by the owner on 2026-09-05; the ledger carries it under D-MOVE-3"*
(`:69-71`; the ledger row is `docs/design/DEFERRED.md:5052`).

**Why it is load-bearing.** The report contradicts itself. Its own fact row 47 says the ceiling and
the ramp *"still exist as a PLACEHOLDER"* and names the four symbols. The door table then says no cap
exists. A synthesis that reads the door table will conclude that nothing must be deleted, and the
deletion that the terrain and block work must not re-anchor onto will stay in the tree.

**Example.** A character walks toward a moon inside a star system. Today the moon's stated ceiling
falls as the character closes, and the character's own stick is scaled down to follow it. The ruling
says the stick is never scaled. The door row says the scaling is already gone. It is not.

**Fix.** Rewrite door 42 as: *"REFUSED as a door — no cap may bind a stick (M-D, S3). The cap is
still IN THE CODE as a placeholder (`dot.rs:465-474`, `flight.rs:93-118`) and its deletion is the
owner-deferred D-MOVE-3."* Then make the deletion an explicit precondition row in the slice order,
so the terrain and block slices size nothing on it.

---

## 3. Finding 2 — BREAKS_LAW (SL6). A sibling's placement is declared lawful with no ask, and the fence that forbids it is not named.

**The claim.** Table A, the SL2 row: *"A ship is a realm; its parent authors and MAY state its
placement to siblings … Locating a SHIP by radio is lawful"*.

**What the code says.** No realm-inbound message carries a placement or a centre at all, and that
absence is pinned structurally:

- `crates/wire/src/intershard.rs:425-433` — the `ChildSceneSet` tombstone: *"no realm-inbound
  message type carries a placement or a centre at all any more … That absence is pinned structurally
  (`crates/wire/tests/intershard_closed.rs` —
  `no_realm_inbound_payload_carries_a_placement_or_a_centre`)"*.
- The same tombstone records why: the down-reflect of scenery was the lane that could tell a realm
  something about itself, and the owner deleted it together with its guard.

**The law.** SL6 says: ask before new data crosses a realm boundary, default NO; state what data,
from which realm to which, why the receiver cannot compute it, and what doing without costs. SL1
clause 2 licenses one thing only — a parent stating to a child the placement it authored FOR THAT
CHILD. A SIBLING's placement is different data on a lane that does not exist.

The report is not wrong about the owner's words: SL2 does say *"placements the parent authors and
may state"*. But the report turns that half-sentence into a licence, and it does so in the one report
whose job is to catch exactly this. Its §4 item 7 lists four SL6 asks. This is a fifth, and it is
bigger than the other four, because granting it deletes a passing structural test.

**Example.** A hull's targeting computer wants the station in the next orbit. The station is the
hull's SIBLING under the same star system. Today the star system authors both placements and states
neither one sideways. Giving the hull the station's placement is a new lane into a realm, and the
wire test that says no such payload exists goes red.

**Fix.** Add the fifth ask to §4 item 7, in the SL6 shape: *the data* — one sibling's placement in
the shared parent's frame; *from which realm to which* — the star system down to the hull;
*why the receiver cannot compute it* — a child holds no placement but the stamped reading of its own,
and folding an absolute is refused by SL1 clause 4; *the cost of doing without* — a ship's contact
list is empty, so targeting, docking approach and collision warning have no source; *what it breaks*
— `no_realm_inbound_payload_carries_a_placement_or_a_centre`. Mark the row REFUSED-UNTIL-ASKED, not
lawful.

---

## 4. Finding 3 — MISSING. SL10 and SL3 disagree about whether a sleeping realm can be drawn, and no row says so.

**SL3** (`CLAUDE.md:177-181`): *"A realm that is not running cannot be drawn — which is WHY visibility
is the spin-up trigger."*

**SL10 V1.1** (`docs/design/owner_decisions_2026-09-07_voxels.md`): the client MAY derive the static
shape from `(seed, address)`.

A client that holds the generator crate can draw a moon's hills with no shard for that moon running
anywhere. The reason visibility spins a realm up was that nobody else could draw it. SL10 removes
that reason for the static half of the picture. The whole reach machinery rests on it: R2 deletes
the parent's marker precisely because a realm in reach wakes and draws itself
(`docs/design/owner_decisions_2026-09-02_reach.md:36-50`).

The report's SL3 row talks only about which rung is the coarsest. Its SL10 row never mentions the
spin-up trigger. This is the largest cross-cutting contradiction the new law creates, and the audit
carries no row for it.

**Example.** A pilot warps toward a star system. The client already holds the seed, so it can draw
the third planet's coastline before any planet shard boots. Does the planet still wake? If it does
not, its live state — the tunnels players dug, the buildings they placed — is not there, and the
coastline the pilot sees is a world nobody edited. If it does wake, the trigger is no longer "so it
can be drawn" and needs a new reason stated.

**Fix.** Add a row: *SL10 against SL3 — what wakes a realm once the client can draw its static
shape?* Recommended answer for the owner: the trigger stays, and its stated reason changes from
"so it can be drawn" to "so its DIFF exists" — a realm must run to state the edits, the placements
and the live deposits the seed does not decide (SL10 V1.6). Then the seam to measure is a client
that draws seed shape with no diff yet applied.

---

## 5. Finding 4 — MISSING. Four of the six hard rules get no row.

The report's Method says it read HR1 to HR6. The body produces one incidental HR3 mention
(§4 item 5) and nothing else. `grep` over the report: HR2 = 0 hits, HR4 = 0, HR5 = 0, HR6 = 0,
"G-IDENTICAL" = 0, "coverage" = 0.

Each of the four binds the voxel foundation directly:

- **HR4 — features once, run anywhere.** Every feature passes the identical fixture on two shard
  kinds or it does not land. The report's own §4 item 5 recommends two surface extractors chosen by
  a per-cell shape class. That recommendation must pass one fixture on a PLANET realm and on a HULL
  realm. `crates/sim/src/capability.rs:37-41` already plants the split as data
  (`VoxelGeometry::{Spherical, Cartesian}`). The report cites that line in its fact table and never
  names the gate.
- **HR5 — 100 % region and branch coverage on Tier-A crates.** SL10 makes ONE generator crate that
  both hosts compile. The base's O7 proposes vendoring about 400 lines of noise into it. Vendored
  code inside a Tier-A crate must reach 100 % or carry a written exemption.
- **HR6 — agent-operable end to end.** SL10 V1.3's no-drift gate must build chunks on a CLIENT build
  and compare them byte for byte. `vdctl` is the shipped way to drive a client. No row says so.
- **HR2 — generic transfer.** A sub-metre block, a no-volume attachment and a one-record tree are
  new kinds. Anything that crosses a shard crosses through the `TransferableKind` registry.

**Example.** A player chisels a 25 cm block onto a hull wall, then flies that hull from a moon's
realm into the star system. The block must cross with the hull through the one transfer machinery,
and the fixture that proves the chisel works must pass on the moon AND inside the hull. Neither
sentence appears in the audit.

**Fix.** Add four rows to Table A, one per hard rule, each naming the base text it binds and the gate
it owes.

---

## 6. Finding 5 — MISSING. SL10 creates a seam the eleven-kind taxonomy does not name.

The report's SL8 row maps each base mechanism to a seam kind and concludes *"No base text
conflicts"*. That is true of the BASE. It is not true of SL10.

SL10 splits the picture in two: the client derives the static shape at once, and the owning realm's
diff arrives over the wire afterwards. The order is a seam. The eleven kinds have no row for it —
the nearest, "arrival pop", is about a realm arriving, not about a surface correcting itself under a
standing player.

**Example.** A player lands on a moon and walks toward a hill. The client drew the hill from the
seed while the hull was still descending. Three frames after the boots land, the moon's shard sends
the diff for the tunnel another player dug last week, and the ground opens under the player's feet.
Nothing in the taxonomy names that, and no tolerance is owed for it.

**Fix.** Propose a twelfth seam kind to the owner — call it the *diff lag* — with a physical
tolerance: the diff for every cell inside the character's own reach is applied BEFORE the derived
shape for that cell is first drawn, and the gate measures the frame count between the two.

---

## 7. Finding 6 — WRONG, narrow. M7 is cited for a purpose it did not approve.

**The claim.** §4 item 3: *"M7 approved ONE scalar distance parent→child."*

**The ruling.** `docs/design/owner_decisions_2026-08-26_movement.md:200` and `:245`: M7 approves
*"one scalar, the distance from the nearest outside LOOKER to this realm. No direction."* It is the
interest signal. Its purpose is visibility — so a realm can decide which of its own children clear
the angular threshold (`:226-228`).

The report borrows it as a general parent-to-child distance a light-lag calculation could read. The
report's conclusion is still right — *"Do not build §7.6 as written"* — but the citation over-reads
the ruling and a synthesis could take it as a licence.

**Example.** A radio call goes from a station in one star system to a hull in another. M7 would tell
the hull's realm how far the nearest looker is. It would not tell anyone how far the station is.

**Fix.** Restate as: *"M7 approved one scalar for VISIBILITY only — the distance from the nearest
outside looker. It is not a distance between two named realms and it cannot carry light-lag."*

---

## 8. Finding 7 — WRONG, cosmetic. The doc-comment count is understated.

Fact row 43 says `FrameSpace`, `SphericalSpace`, `reanchor()` and `AnchorGen` are named by *"two doc
comments"*. There are six sites: `crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11`,
`:18`, `:35`, `:40`, and `crates/sim/src/stub/transient.rs:239`. The substance holds — none of them
is code. Correct the count, so the next reader who greps keeps confidence in the table.

---

## 9. Law-by-law result

| Law | Result on this report |
|---|---|
| SL1 — a realm is told where it is, one hop, never fold an absolute | HOLDS. The report refuses the base's absolute-distance text and the grandparent field by name, and it is right that `pin_abs` is gone |
| SL2 — no occupant pose crosses a boundary | BREACHED IN THE REPORT'S OWN RULING — Finding 2. It licenses a SIBLING's placement into a child's shard with no SL6 ask |
| SL3 — a realm draws itself | HOLDS on the marker. INCOMPLETE against SL10 — Finding 3 |
| SL4 — physics and re-home separate | HOLDS. No motion symbol is proposed on a crossing path |
| SL5 — one world | HOLDS. The report refuses the test-only generator (O57) by name |
| SL6 — ask before new data crosses | BREACHED — Finding 2. Four asks are listed; a fifth is declared lawful instead |
| SL7 / SL9 — unbounded children, no per-child per-tick cost, no scan | HOLDS. The report demands a lookup and cites the measured galaxy tick |
| SL8 — a seam is a defect | INCOMPLETE — Finding 5. The new SL10 seam has no kind and no tolerance |
| SL10 — one generator, two hosts, no drift | HOLDS on the arithmetic and the gate. INCOMPLETE against SL3 — Finding 3 |
| Movement contract | HOLDS on the rulings. WRONG on the code — Finding 1 |
| Seed ruling | HOLDS. The secret-keyed placement is refused by name |
| Reach and visibility | HOLDS. A radius, not a tree walk; a dormant realm drawn by nobody |
| HR1–HR6 | INCOMPLETE — Finding 4. No row for HR2, HR4, HR5 or HR6 |
| FINAL backend; no stand-in, no second implementation | HOLDS. No port, no config-selected second world, no placeholder mesh is proposed |
| The owner's V2.1–V2.9 | HOLDS. No door is shut on any of the nine |

---

## 10. What the synthesis must do with this report

Use it. It is a sound register and most of it survives a hostile re-measurement. Before it feeds the
synthesis, apply seven corrections:

1. Rewrite door 42 — the cap is refused by the ruling and it is still IN THE CODE (Finding 1).
2. Move the sibling-placement claim out of "lawful" and into the SL6 ask list, and name the wire test
   it would break (Finding 2).
3. Add the SL10-against-SL3 row: what wakes a realm once a client can draw its shape (Finding 3).
4. Add one row each for HR2, HR4, HR5 and HR6 (Finding 4).
5. Propose the twelfth seam kind for the derived-shape-then-diff order (Finding 5).
6. Narrow the M7 citation to visibility (Finding 6).
7. Correct the doc-comment count in fact row 43 (Finding 7).
