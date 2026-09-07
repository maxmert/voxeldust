# Verdict — feasibility refutation of `06_storage_diff_lane.md`

**Lens:** storage, the edit pyramid, and the diff lane, refuted on technical grounds.
**Date:** 2026-09-07. **Report under test:** `docs/investigation/2026-09-07/06_storage_diff_lane.md`.
**Result: REFUTED.** Three load-bearing claims are wrong on today's code or on the report's own
arithmetic. Five cases the domain needs are absent.
**Method:** I read the code for every claim about the code. I re-did every number the report used to
decide something. I mark each number MEASURED (with how) or ESTIMATED.

---

## 1. What survives

These claims are true. I checked each one against the source.

| Claim in the report | My check |
|---|---|
| `StoreRole::RealmStore = 4` holds the berths and the body | MEASURED by reading `crates/core/src/store_stamp.rs:74` and `crates/sim/src/stub/built_store.rs:31,33,78,81` |
| Schema ids 30 and 31 are taken; 32 and 33 are free | MEASURED: `grep -rn "SchemaId("` over `crates/core` and `crates/sim` gives 1, 2, 3, 10, 11, 12, 13, 20, 30, 31 |
| A realm's file is named from the `RealmId`; a hull gets `realm-ship-<id>.redb` | MEASURED: `crates/bins/src/lib.rs:2310,2323,2332` |
| `commit` blocks on the PRIOR batch, so at the next tick the last batch is durable | MEASURED: `crates/sim/src/io/mod.rs:489-494`. The report's "ship the diff one tick after the edit" (§2.2 step 4) is sound |
| One redb write transaction and one fsync per batch, all or nothing | MEASURED: `crates/io-prod/src/store.rs:371-373` |
| `InterShardFlow` has 44 arms, last discriminant 43 | MEASURED: `awk` over `crates/wire/src/intershard.rs:124-604`, `ReachStated` at line 603 |
| The header prose says 16 arms and is stale | MEASURED: `crates/wire/src/intershard.rs:9` |
| Step ids 7 to 18 are taken; 19 and 20 are free | MEASURED: `grep -n "_STEP: u32" crates/wire/src/intershard.rs` gives 7..18; `grep -rn "u32 = 19\|u32 = 20" crates/wire/` is empty |
| `ChannelKey` does not exist; `last_owner_fence` does not exist | MEASURED: both greps over `crates/` are empty |
| `BulkMsg` is named in no crate outside `vd-wire` | MEASURED: `grep -rn BulkMsg crates/` gives only `crates/wire/src/lib.rs:9`, `channels.rs:12,285,1133,1139` |
| `MsgClass` has no bulk arm and is append-only pinned | MEASURED: `crates/sim/src/io/mod.rs:51-85` |
| `RealmKindTag::Star = 6`, `Ship = 7` | MEASURED: `crates/core/src/realm_path.rs:52,61` |
| `action_bits` rides the unreliable input datagram and names the two owed consumers | MEASURED: `crates/wire/src/channels.rs:263-267` |
| The durable outbox is on for every shard; the gateway holds no store | MEASURED: `docs/design/DEFERRED.md:917-940` |
| The fold rule is exact integer arithmetic, so it cannot drift between two hosts | Sound. Masks, a rounded integer mean and a lowest-id tie-break carry no float. FMA, fast-math and libm cannot enter |

*Example: a hull's shard opens `realm-ship-4711.redb`, writes the berths it authored for its built
children, and the file's label refuses a file from another world before it reads one row. The block
families join that same file. That half of the report stands.*

One qualification. `open_realm_store` returns "no store" unless `VD_REALM_STORE` or
`VD_REALM_STORE_DIR` is set (`crates/bins/src/lib.rs:2367-2378`). So "the store already exists" is
true of the machinery and conditional on the deployment.

---

## 2. Finding F-1 — ONE TIER PER WINDOW IS IMPOSSIBLE ON TODAY'S GATEWAY (load-bearing)

**The report's claim.** §3.3 and plant 6: *"the gateway states ONE plain tier number on the window:
'serve rung 5 and coarser'"*, and A-4 says the cost of doing without is *"the realm ships every rung
to every window: a rung-0 city to an observer in orbit"*.

**What the code does.** The gateway opens ONE window per `(shard, scope)` and shares it across every
session behind that gateway:

- `crates/connection-plane/src/gateway/window_lane.rs:287-296` skips opening a second window when a
  held window already has the same shard and the same scope.
- `crates/sim/src/stub/window.rs:41` keys the shard side `(NodeId, WindowId)` — the gateway NODE, not
  a session.
- `WindowScope` has two arms only: `Occupants` and `Child(RealmId)`
  (`crates/wire/src/session_flow.rs:582-590`). Neither names a person.
- The lane's own design says the same: *"ONE fold per occupied realm per gateway per tick, then a thin
  per-session band filter"* (`docs/design/window_lane.md:578-580`).
- The sky lane already learned this exact lesson and wrote it down: *"a gateway aggregates many
  clients, which may hold many different skies or none; 'what do you have' has no single answer at
  this hop"* (`crates/wire/src/session_flow.rs:167-169`).

**Why the report is wrong.** Two pilots behind one gateway look at the same moon. One sits in orbit.
One is 20 km out and descending. They share ONE window, because the shard and the scope are the same.
A single tier byte on that window has no correct value. Serve the coarse rung and the descending pilot
sees a smeared city — an arrival pop, which SL8 calls a defect. Serve the fine rung and the moon ships
the rung-0 city to the window anyway, which is exactly the cost A-4 says the byte buys back. So A-4's
benefit is not real as written, and §3.3's example ("At 20 km the gateway lowers the rung to 5") cannot
happen.

**The retrofit cost the report did not state.** Either

- the moon serves the FINEST rung any session behind that window needs and the gateway drops rows per
  session (the band filter the lane already runs). Then the byte is a floor, not a selection, and the
  moon's egress is sized by the closest pilot behind every gateway; or
- a window becomes per session. Then the window count grows with sessions, and the static roster the
  S11 counter comparison protects (14.2 MB, 28.4 MB/s per window at the S12 census,
  `crates/wire/src/session_flow.rs:136-137`) is paid per session instead of per gateway.

Neither option is free, and the report costs neither.

**Fix.** State plant 6 as a per-window FLOOR that the gateway computes as the minimum over the sessions
it fans that window to, and add the gateway-side per-session row filter to the plant list. Measure the
moon's egress with one close pilot and 127 distant ones behind one gateway.

---

## 3. Finding F-2 — THE COARSE GENERATOR IS THE WHOLE MECHANISM, AND THE REPORT NEVER DEFINES IT (load-bearing)

**The report's claim.** §1.3 rule 1: *"An entry is deleted the moment its summary equals
`generate_summary(address)`"*. §0 item 1: a chunk with no edits costs zero bytes *"at every rung"*.
Every storage number in §1.3 and §2.2 rests on this rule.

**What is missing.** The report never says how `generate_summary` produces a summary at a COARSE
address, and it never says that the answer equals the exact fold of the generator's own tier-0 cells
under that address. Both are required, and the second is the one that breaks.

- The cheap way to coarsen a fractal terrain is to drop octaves. This project already treats that as
  free (the standing LOD note: *"fractal terrain coarsens FREE by dropping octaves"*, echoed at
  `docs/design/window_lane.md:589-590`).
- Dropping octaves gives a DIFFERENT number from folding the fine cells. Call them `coarse(a)` and
  `fold(fine under a)`. They agree nowhere in general.
- The pyramid stores `fold(fine under a)` for an edited chunk, and prunes against `coarse(a)`. If the
  two differ, NOTHING prunes. An edit inside one hill writes an entry for every coarse cell of the
  chunk it touches, at every rung. The sparsity table in §1.3 (0.14 entries per edit for a quarry) does
  not survive that.
- If instead the generator is forced to define `generate_summary(a) := fold(fine under a)`, the cost
  is 8^L tier-0 evaluations for one rung-L address. At rung 13 that is 5.5e11 evaluations
  (ESTIMATED, 8^13). At the MEASURED 12.19 ns per noise evaluation (`scripts/noisebench`, cited by the
  report at M-5) that is about 1.9 hours for ONE coarse cell.

**Check the report's own bench against this.** M-5 quotes *"5.85 µs at depth 13"*. 5.85 µs ÷ 12.19 ns
is about 480 evaluations, which is 13 rungs times about 37 — that number only exists if a coarse
summary costs a handful of evaluations. So M-5 silently assumes a cheap closed-form coarse generator,
which is the thing the report never designs and never names as a door on the generator crate.

**Why this is not another domain's problem.** §1.6 and §10 make `generate_summary` a HARD door of THIS
format, and §1.3's prune rule is what makes the store sparse. The report freezes a door on a function
whose cost, definition and consistency with the fold it does not state.

**Fix.** Add a decision to §9: *the generator crate must expose a closed-form coarse summary at every
rung, and the pyramid's fold must be DEFINED as that function.* Add a bench: for 10,000 random coarse
addresses at every rung, compare `generate_summary(a)` against the fold of its children, and measure
the cost of both. The gate is red on one differing entry.

*Example: a player digs one hole in a hill on a moon. Under the report's rule the moon's shard should
store a few entries. If the coarse generator is not the fold of the fine one, the moon's shard stores
an entry for every coarse cell of that hill's chunk, at thirteen rungs, for one hole.*

---

## 4. Finding F-3 — THE CATCH-UP MANIFEST GROWS WITH THE MOON, NOT WITH THE EDITS (load-bearing, SL9-shaped)

**The report's claim.** §3.4 item 2: the shard sends a manifest per interest cube on entry, listing
*"the chunk keys in that cube that carry a diff, with a digest per key (8 + 8 bytes each, ESTIMATED)"*,
so that absence means "the seed's shape".

**Why it does not hold.** The manifest must be COMPLETE for absence to mean anything. At a coarse rung
the interest cube covers a large part of the moon. The report's own sizing case (§1.3 last row, and
D-4) is *"ten thousand players leaving marks 57 m apart over a whole moon"*, 100 M edits.

- The grid domain's chunk edge is 62 cells (the report's own §1.1). At 1 m cells that is a 62 m chunk.
- Marks 57 m apart put about one mark in every tier-0 chunk. ESTIMATED, from the report's own two
  numbers.
- At rung 7 (a chunk edge of 62 × 2^7 ≈ 7.9 km) a moon of 3,000 km diameter has about
  4.5e5 chunks (ESTIMATED: 4π × (1.5e6)² ÷ (7.9e3)²). Essentially all of them carry a diff.
- The rung-7 manifest is then about 7 MB (ESTIMATED, 4.5e5 × 16 B). For a 10,000 km body it is about
  80 MB.

That cost grows with the MOON'S SURFACE, not with the occupants and not with the edits, once edits are
dense. It arrives at the moment a pilot enters orbit, which is the moment SL8 forbids a hitch.

**What the report benches.** M-2 benches a city of 2,400 buildings. Nothing benches the manifest of the
scattered case that §1.3 and D-4 say is the sizing case. The report itself calls that row "the one the
base's table missed", then leaves it out of the lane numbers.

**Fix.** Either make the manifest hierarchical — one digest per coarse chunk that covers its whole
column, so absence at a coarse key means "nothing below it differs" — or make absence mean "not yet
known" and draw the seed's shape at once with an explicit completion signal. Then bench the scattered
case, not only the city.

---

## 5. Finding F-4 — THE SEAMLESS ANSWER POINTS AT A GATE THAT IS NOT IN THE REPORT

§1.2's table answers V2.9 with *"The fly-away/fly-back gate (§6): a dug tunnel and a built tower are
visible at every rung boundary; nothing pops."*

§6 lists M-1 to M-12. None of them is a fly-away/fly-back gate (MEASURED: `grep -n "fly-away\|fly-back"`
over the report gives one hit, line 104, the reference itself). So the report's ONLY answer to the
seamless law is a pointer to nothing.

This matters because the report's own record makes the pop real. `fill_256` is 8 bits.

*Example: a player digs a quarry 20 m across into solid rock on a moon. At a 128 m coarse cell the
missing volume is 8,000 m³ out of 2,097,152 m³. `fill_256` rounds from 255 to 255. The coarse summary
now EQUALS the generator's summary, so §1.3 rule 1 DELETES the entry. The quarry exists at rung 4 and
vanishes at rung 7. A pilot climbing away watches the quarry blink out. That is the "detail-by-box" seam
and the "arrival pop" seam, both named in SL8.*

**Fix.** Write the gate: fly a hull from the ground to orbit and back over a dug tunnel, a filled
quarry and a built tower, sample the drawn shape at every rung boundary, and require that the edit stays
visible. State the rung at which each edit shape stops being representable, and decide with the owner
whether the pyramid needs a "differs from the seed" sticky bit that survives quantisation.

---

## 6. Finding F-5 — THE §1.2 EXAMPLE MISREADS THE RECORD IT FREEZES

*"At 128 m cells it is one entry: 'steel, 2 % full, bottom octants'."*

- The tower is 20 m on a side (the same example says 1,000 entries at 2 m cells, so 10 × 10 × 10).
- 20³ ÷ 128³ = 0.38 %, not 2 %. `fill_256` is 1, not 5. WRONG by a factor of five.
- A 20 m object inside a 128 m cell sits in ONE octant. `occ_mask` has one bit set, not the four
  "bottom octants" says.

The arithmetic error is small. What it hides is F-4: at that rung the tower is one unit of 256, and one
more rung up it is zero.

---

## 7. Finding F-6 — THE IDEMPOTENCY JOURNAL HAS NO RETENTION BOUND AND NO MINTER

**The report's claim.** §0 item 5 and §4.2 step 4: the forward is *"idempotent on `(TransferId, step
20)`"*, checked against *"its applied-steps journal"*.

**What the code is.** `AppliedSteps` is an in-memory `BTreeSet<(TransferId, u32)>`
(`crates/sim/src/stub/crossing_receive.rs:52`). Its own doc says:

> *"OWED (1d.1, when the receiver feeds this): a RETENTION BOUND — drop a transfer's steps on its
> terminal ... so a long-lived dest shard does not accumulate one entry per `(transfer, step)`
> forever."* (`crates/sim/src/stub/crossing_receive.rs:48-51`)

The durable `applied_steps` table does not exist (MEASURED: `grep -rn applied_steps crates/` finds only
doc comments and `crates/connection-plane/src/gateway/session.rs:109`).

**Why the report's use is worse than a transfer's.** A transfer has a terminal, so its steps can be
dropped. A block edit has none. A pilot who digs from inside a landed hull forwards a batch per tick.
Each batch adds a permanent row to the moon's in-memory set. The set grows with EDITS and never
shrinks. The moon's shard runs for weeks.

**And nobody mints the id.** The report reuses `TransferId` (`crates/core/src/ids.rs:98`, *"THE one
correlation id"*) for an operation that is not a transfer, and never says which party mints it, whether
it is stable across a redelivery from the durable outbox, or how a replayed frame from the outbox finds
the same id after the origin restarts.

**Fix.** Either give the forward its own idempotency key with a natural terminal (the edit batch's
sequence per origin realm, with a low-water mark the ack advances), or land D-22's retention bound
first and define the terminal for an edit. Name the minter. Add a bench: 200 edits per second for an
hour, then measure the journal's size.

---

## 8. Finding F-7 — THE SL9 ARGUMENT IN §3.3 REVERSES THE CODE IT CITES

**The report's claim.** *"Observers are sorted into cubes; a dirty `(chunk, rung)` finds its holders by
a range read of the cube slab, never a scan over all observers (SL9)"*, citing
`crates/sim/src/stub/interest.rs:120-180`.

**What that function does.** `plan_interest` walks EVERY observer, and for each one range-reads a slab
of CELLS (`crates/sim/src/stub/interest.rs:129-160`). It is observer-major. The range is over the first
coordinate only, then it filters each cell in the slab. So the cited code is exactly the shape the
report says it is not.

The inverted index the report describes — from a dirty chunk to its holders — does not exist. It may be
buildable, but the report presents it as the existing pattern and therefore states no cost and no
bench for building it. M-11 benches the ratio the design should have, not the design.

**Fix.** Say plainly that the chunk lane needs a NEW cube-to-observer index, state who maintains it and
what it costs per tick, and keep M-11.

---

## 9. Finding F-8 — "NO CHUNK DATA CROSSES A REALM BOUNDARY, EVER" IS NOT ESTABLISHED

§5.2 and §8 supersede the base with *"no chunk data crosses a realm boundary, ever"*, and §5.2 asks for
NO fetch verb. §5.2 also admits *"The crossing domain must confirm that no ghost or overlap band needs
a neighbour's collider surface."* §8 then states the strengthened claim as settled. It is not.

**The case the report does not answer.** A hull is a built realm made of blocks (V2.3). A hull lands on
a moon and becomes the moon's built child. The moon integrates its children's collisions
(`integrates_children`, `crates/sim/src/capability.rs:90`). To resolve the contact between the hull's
belly and the moon's rock, the moon needs the hull's SHAPE, and a hull's shape is its blocks. The look
bag carries an outline only (`Boundary`, `crates/core/src/look.rs:29-30`), and the one-radius law
forbids more (`crates/core/src/look.rs:37-42`).

So either the hull's collider surface crosses to the moon — chunk data crossing a realm boundary, an
SL6 ask the report does not make — or the contact is resolved against a box, which shows a hull sinking
into a hill or floating above it.

**Fix.** Move this from a footnote to an SL6 ask with the data named, or state the box contact and its
visible cost, and let the owner choose.

---

## 10. Finding F-9 — FIVE CASES THE DOMAIN NEEDS AND THE REPORT NEVER NAMES

MEASURED by grep over the report: none of these words appears in it except as listed.

1. **A mined cell under a placed block.** A player places a steel beam, then mines the rock the beam
   stood on. Two authored cells, one of them empty. The report's tag 4 ("cells returned to the seed")
   and tag 1 (authored cells) never meet in one example, and §1.3 rule 1 prunes on equality with the
   SEED, which the placed beam is not. What the coarse entry says about that column is undefined.
2. **A sub-metre block on a slope.** V2.4 is answered by one line: `fill_256 = volume_units × 255 /
   CELL_VOLUME_UNITS`. A quarter-metre block sitting on a sloped seed surface shares its 1 m cell with
   the seed's rock. The cell's fold must combine an authored sub-cell with a seed-derived partial
   volume, and the report never says whether the stored record holds the sub-cells alone or the whole
   cell's resulting volume. That decides whether a change to the generator moves a player's block.
3. **A tree standing on a mined edge.** §1.4 handles FELLING a tree only. It does not handle mining the
   ground the tree stands on. The tree's cell is not authored, so no entry exists, and the client keeps
   drawing a tree hanging in the air over the new hole.
4. **A HUD or a rotating joint on a turret (V2.5).** Attachments ride tag 5 on the tier-0 chunk row,
   and the row is reliable, ordered, send-on-change, on a PACED bulk class (§3.6). A joint's angle and
   a HUD's displayed value are LIVE state at the tick rate. Putting them on that lane queues them
   behind a 23 MB city catch-up. The report reserves tag 7 to another domain and never says which lane
   carries a moving turret.
5. **An edit during a crossing.** §4.4 covers a re-shard of the TARGET realm (step 19). It does not
   cover the AUTHOR crossing. §2.2 step 4 ships the diff at tick T+1. If the pilot's session re-homes
   at tick T, the echo's session id, the ack route and the `EventMsg` destination are undefined.

---

## 11. Finding F-10 — TWO DOORS ARE MIS-STATED

**A door the report leaves out of §10.** Two star systems with the same seed in different galaxies
lower to the SAME `RealmId::System(seed)` (`crates/core/src/realm_coord.rs:6-8`, stated there as a
known aliasing), and the store's file name is built from the `RealmId`
(`crates/bins/src/lib.rs:2323-2332`). So two star systems share one `realm-system-<seed>.redb` and
overwrite each other's edits. §8 records this as *"an identity-domain item"* and §10 does not list it.
It is a DATA-LOSS door for THIS domain, with the same deadline as the rest: before the first world is
saved.

**A door whose cost is overstated.** §10 says a wrong family prefix or schema id gives *"a wrong-shaped
file [that] decodes into records, not into an error (positional encoding)"*. The realm store's families
are TLV-FRAMED with a schema id, and `TlvReader::parse` refuses a foreign schema
(`crates/sim/src/stub/built_store.rs:143,170`; the refusal is tested at `crates/core/src/look.rs:259-274`).
So a wrong SCHEMA is caught. Only a wrong PREFIX BYTE is silent, because it decides which rows one scan
returns. State the door as the prefix byte alone.

---

## 12. What I could not refute, and what I did not test

- The determinism reasoning for the FOLD is correct and needs no defence: masks, a rounded integer
  mean and a lowest-id tie-break carry no float, so libm, FMA contraction and fast-math cannot enter.
  The report is right to push the float question into the generator crate under SL10 clause 4.
  It is wrong only in freezing `generate_summary` as its own door without defining it (F-2).
- The report proposes no new library. It reuses redb, postcard and the existing seams. That is correct.
- I ran no build, no test and no bench, as the task directs. Every number I produced is marked
  ESTIMATED and shows its arithmetic. Every claim about the code is MEASURED by reading the file and
  is cited by `file:line`.
- I did not test the report's storage totals (0.8 GB, 4.9 GB, 41 MB, 5,508× amplification). The report
  marks them ESTIMATED and books them as M-3, M-4 and M-7. That is honest, and D-4 is put to the owner
  rather than decided.

---

## 13. Verdict

**refuted = true.**

Three load-bearing claims fail:

1. **F-1** — one tier per window cannot work, because a window is shared by every session behind a
   gateway (`crates/connection-plane/src/gateway/window_lane.rs:287-296`). Plant 6 and A-4 need rewriting.
2. **F-2** — the coarse generator, on which every sparsity number and the prune rule depend, is
   neither defined, nor costed, nor proved equal to the fold.
3. **F-3** — the catch-up manifest grows with the moon's surface once edits are dense, which is the
   report's own sizing case, and no bench covers it.

Two more are serious: the seamless answer points at a gate that does not exist (**F-4**), and the
idempotency journal for the forward has no retention bound and no minter (**F-6**).

The store half of the report is sound and well cited. The lane half is not ready to freeze.
