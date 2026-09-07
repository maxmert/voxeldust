# Law verdict — 06 Storage, edit pyramid, and the diff lane

**Report under test:** `docs/investigation/2026-09-07/06_storage_diff_lane.md`
**Lens:** the law refuter. Every binding rule of `CLAUDE.md` (HR1–HR6, SL1–SL10), the rulings of
2026-09-07, 09-05, 09-02, 09-01, 08-27 (three), 08-26, 08-24, `DEFERRED.md`, and the seamless law SL8.
**Date:** 2026-09-07.
**Verdict: REFUTED.** One load-bearing claim breaks SL1. Two claims about the code are false.

---

## 1. The refutations

### F-1 — BREAKS SL1. The hull states a position in the moon's frame.

**The claim.** §4.2 step 3 sends the forward with *"the target realm and chunk"* in the payload. §3.2
fixes what a chunk key is: *"the address in the realm's OWN frame"*. The stating realm there is the
TARGET. So the hull's shard computes a MOON-frame chunk address and ships it up.

**The law.** SL1 clause 1: *"going up the child ships its own-frame pose and the parent adds"*. Clause 3:
*"A child NEVER derives, adjusts, computes or states its own placement"*. Clause 4: *"What you are told
about yourself, you NEVER pass on."*

**Why the claim breaks it.** The hull can only name a moon-frame cell by adding its own berth reading to
a local offset. The moon authored that berth and stated it to the hull as a stamped reading. The chunk
key hands the reading straight back, folded into an address. Subtract the local offset and you recover
the hull's own placement. The forward therefore carries a placement the child asserts about itself.

**Example.** A pilot inside a hull, berthed on a moon, breaks a rock beside the ramp. Under the report
the hull says *"moon, chunk 4/17/2, cell 91"*. Chunk 4/17/2 is a moon-frame address. The hull knows it
only because the moon told the hull where the moon put the hull.

**The fix.** The hull states the target point in the HULL'S OWN frame. The moon adds the placement it
authored for the hull and resolves its own chunk and cell. The moon already holds that placement, because
the moon is the only writer of it (SL1 clause 1). This fix also carries the reach test of step 4 for free:
the moon tests the resolved cell against the placement and the extent it already holds. Amend the ask A-1
so `BlockEditForward` carries a hull-frame point, never a moon-frame chunk key.

---

### F-2 — WRONG. `coordinator_of` does not exist.

**The claim.** §4.2 step 2: *"asks the directory who owns it (the `coordinator_of` resolver D-32 already
provides)"*.

**The code.** `grep -rn coordinator_of crates/` returns ONE hit, and it is a comment:
`crates/node/src/saga_runtime/execute.rs:164`. `DEFERRED.md:74` states it plainly: *"There is NO
`RegionId` / `coordinator_of` / `resolve_coordinator` symbol anywhere"*. `DEFERRED.md:83-84` defers the
resolver to N-orchestrator scaling: *"additive — code at N-orchestrator scaling, NOT before"*. D-32 is
🟧, not 🟩.

**The second defect in the same step.** The lookup is not needed at all. The report's own §4.1 admits a
forward ONLY from *"an occupant of a DIRECT CHILD realm of the target"*. So the target is always the
parent, and a hull's lineage coord names its parent. The report contradicts itself and buys an unbuilt
dependency on the way.

**The fix.** Delete step 2. State the rule as *"a forward goes to my parent, whose name my lineage coord
carries"*. Drop the `target realm` field from A-1; the routing key is the parent.

---

### F-3 — WRONG citation. `window.rs:225-233` is the window pruner.

**The claim.** §4.2 step 4: the owner admits the forward when *"the origin is in its attested roster as a
live direct child (the same predicate every up-lane uses, `crates/sim/src/stub/window.rs:225-233`
pattern)"*.

**The code.** `crates/sim/src/stub/window.rs:224-234` is `prune_expired_windows`. It drops a window whose
keep-alive lapsed. It tests no sender and no roster. The predicate the report wants is
`window_sender_is_head` at `crates/wire/src/session_flow.rs:647`, named by the window lane's own arm doc
at `session_flow.rs:333` (*"sender must be the roster head for the stating realm"*).

**The fix.** Cite `crates/wire/src/session_flow.rs:647`.

---

### F-4 — UNMEASURED AS FACT. The regime handover is argued, not measured (SL8, V2.9).

**The claim.** §3.3's landing example ends: *"At no moment did the picture jump, because each rung
arrived before the previous one left."*

**Why this is an argument.** §3.3 gives TWO deciders of the drawn rung, with two different rules:

- outside the realm, the gateway states ONE plain tier for the whole window;
- inside the realm, the moon's shard picks a rung PER OBSERVER, tier 0 in the near disc and rung L in an
  annulus, with a speed lead and hysteresis.

The pilot crosses from one decider to the other at the moment the pilot walks out of the hull and becomes
the moon's occupant. Nothing in §6's twelve benches measures that moment. M-2 measures a catch-up; M-9
measures click-to-pixel; neither drives the handover. SL8 says a tolerance is PHYSICAL, and the standing
rule of method says a continuity claim is a MEASUREMENT that could have failed.

**The fix.** Add a bench. A pilot flies a hull to a moon that holds a city, lands, and walks out. Record
the rung the client draws per chunk, per frame, across the re-home. The number that must hold: no chunk's
drawn rung goes BACKWARD, and no chunk is undrawn for one frame. Say in the report whether the client
KEEPS its `(realm, chunk)` cache across the re-home; if it re-fetches, the walk out of the hull is a lane
flood and a black frame at once.

---

### F-5 — MISSING. No G-IDENTICAL gate (HR4).

**The measurement.** `grep -n "HR4\|G-IDENTICAL\|two shard kinds\|shard kind"` over the report returns
nothing. HR4 reads: *"every feature passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it
doesn't land"*. `crates/sim/src/capability.rs:85-90` repeats it for children: *"the identical fixture
must pass on TWO REALM KINDS"*.

**Why it bites here.** §4.3 says *"A hull is a realm like any other (V4 of 2026-09-01): no ship-specific
branch on this path."* That is an argument. The block families, the walk-up, the checkpoint and the diff
lane must run byte-identically on a moon's shard and on a hull's shard.

**The fix.** Add M-13 to §6: the identical edit fixture — break a cell, walk up, checkpoint, crash,
replay, ship the rows — runs on a moon realm and on a ship realm, and both stores and both row sets
compare byte for byte.

---

### F-6 — MISSING. The seed ruling's survey mechanic is never asked about.

**The measurement.** The report never names the ruling of 2026-08-27 on seeds and secrecy except in the
header list. `grep` for "valuable", "deposit" and "survey" over the report returns nothing.

**The law.** S5.2: *"Anything that pays must read world state that moves"*. S5.3: *"The survey is the
mechanic. What is in the ground is revealed by looking, and looking is a thing the player does in the
world."*

**Why it bites here.** Under SL10 clause 6 a valuable deposit is live state, so it is NOT in the
generator. So it is a diff. §3.4 ships every diff of every chunk in a client's interest, complete, and
§3.4 point 2 even ships a manifest that names every chunk that carries one. A client that holds a chunk
holds every deposit inside it, in memory, before the player has surveyed anything. The letter of the law
holds — the substance is not a pure function of seed and position — but S5.3's mechanic is handed away by
the lane.

**Example.** A prospector lands on a moon. The moon's shard ships the tier-0 diffs for the chunks around
the boots. A changed client reads the ore in the rock 60 m down and drills straight to it. Nobody
surveyed.

**The fix.** Add an owner decision row: does a deposit diff ship on entry to interest, or only after the
realm records that this account surveyed that chunk? Name the cost of each. Do not decide it here.

---

### F-7 — MISSING. A tick hitch is accepted without its continuity (SL8).

**The claim.** §2.1 records that `commit` blocks on the prior batch's durability, *"so a disk stall stalls
the tick loudly"* (true: `crates/sim/src/io/mod.rs:489-495`). §2.2 step 4 then makes every diff wait for
that durability. M-10 states one number: `backpressure_stalls == 0`.

**Why it bites.** A stalled tick is seam kind seven of the eleven. SL8 says never ship a capability
without its continuity. The report says what GOOD looks like and says nothing about what the moon does
when the disk is slow.

**The fix.** State the degrade. Example: the moon keeps ticking and keeps shipping poses; only the edit
lane parks; the shard counts the parked ticks and says so; the pilot's pick swings and the rock breaks
late, but the pilot never freezes.

---

### F-8 — MISSING. What the client draws while the manifest is outstanding.

**The claim.** §3.4 point 2: *"The client marks a chunk complete when its manifest is held and every
listed key has arrived."* The report never says what the client draws before that.

**Why it bites.** Two possible behaviours, and both are named seams. Draw the seed's hill and then cut the
tunnel in, and that is an arrival pop. Draw nothing, and that is a black frame. §3.4 point 3's
"coarse first" softens it, because a coarse row already carries the tunnel's volume, but the report never
says so and never measures it.

**The fix.** State the rule: the client draws the derived shape at the coarsest rung it holds, and never
draws a chunk at a FINER rung than the rung whose diff it holds. Measure it in the F-4 bench.

---

### F-9 — WRONG, not load-bearing. The tombstone count.

**The claim.** §5.1: *"Four are tombstones (`RealmObservation`, `RealmShapeObservation`, `ChildSceneSet`,
plus the entity relay pair)."*

**The code.** The list names five, not four. The file marks EIGHT arms `★TOMBSTONE`:
`OccupantInterest` (`intershard.rs:279`), `ProxySceneSet` (`:294`), `RealmCascade` (`:310`),
`EntityInterest` and `EntityCascade` (`:317`, `:328`), `RealmObservation` (`:381`),
`RealmShapeObservation` (`:400`), `ChildSceneSet` (`:416`).

**The fix.** Say eight, and list them.

---

## 2. What survives the test

Every one of these was checked against the code or the ruling, not accepted.

| Claim | Verdict | Evidence |
|---|---|---|
| 44 arms, last discriminant 43 = `ReachStated`; the prose says 16 and 24 | STANDS | counted 44 arm heads in `crates/wire/src/intershard.rs:124-604`; `ReachStated` at `:603` with *"APPENDED (discriminant 43)"*; `intershard.rs:9` says 16, `crates/wire/src/lib.rs:11` says 24 |
| Step ids 19 and 20 are free | STANDS | `RE_SOLICIT_STEP = 18` at `intershard.rs:118` says *"18 is the next free id"*; `grep "_STEP: u32 = 19\|= 20"` returns 0 |
| None of `Tier0Key`, `ChunkKey`, `CellIndex`, `PyramidEntry`, `BlockStore`, `ChannelKey`, `last_owner_fence` exist | STANDS | `grep -rn` over `crates/` returns 0 for each |
| `MsgClass` has no bulk arm; nothing outside `vd-wire` names `BulkMsg` | STANDS | `crates/sim/src/io/mod.rs:51-85`; `grep -rn BulkMsg crates/` hits `wire/src/channels.rs` and `wire/src/lib.rs` only |
| `StoreRole::RealmStore = 4` holds berths and body; schema ids 30 and 31 taken | STANDS | `crates/core/src/store_stamp.rs:74`; `crates/sim/src/stub/built_store.rs:30-33, 78-80` |
| `commit` is block-on-prior, one redb transaction and one fsync, all or nothing | STANDS | `crates/sim/src/io/mod.rs:483-495`; `crates/io-prod/src/store.rs:371-373` |
| The realm store is named by `RealmId`, and `Ship(EntityId)` is birth-immutable | STANDS | `crates/core/src/pose.rs:40`; `crates/bins/src/lib.rs:2310-2333` |
| `RealmKindTag::Star = 6`, `Ship = 7` | STANDS | `crates/core/src/realm_path.rs:49, 61` |
| `FrameFor` names recipients; `EntityOutOfInterest` is Retained; `plan_interest` is pure and cube-ranged | STANDS | `crates/wire/src/session_flow.rs:487-493`; `crates/sim/src/stub/frames.rs:304-316`; `crates/sim/src/stub/interest.rs:128-177` (a `BTreeMap::range` over the cube slab, never a full scan) |
| `WindowOpen` carries `static_held`, the digest the subscriber already holds | STANDS | `crates/wire/src/session_flow.rs:123-147` |
| The look bag is TLV with skip-unknown tags; `SceneRow` is a pose plus a bag and branches on no realm kind | STANDS | `crates/core/src/look.rs:25-40`; `crates/wire/src/channels.rs:363-392` |
| The gateway may state a plain tier number on the window | STANDS | `docs/design/window_lane.md:575` heads the section *"owner Q&A 2026-08-15, recorded"*; `:586-593` records the request shape. SL3 is kept: the REALM still authors the rows, the gateway only asks for a rung |
| The outbox is on for every shard since 2026-09-06 | STANDS | `docs/design/DEFERRED.md:915-950` |
| D-39.1 owes the forward and the reliable client action carrier, with PvP fire as the second consumer | STANDS | `docs/design/DEFERRED.md:360-372`; `crates/wire/src/channels.rs:262-267` |
| No occupant pose crosses on the forward; the owner tests the ORIGIN REALM's placement and extent | STANDS (SL2) | §4.2 step 4 and §8's correction of the base at `block_system_design.md:4671-4675`. This is SL7's occupied-child proxy used correctly |
| No chunk data crosses a realm boundary; no fetch verb between shards | STANDS (SL6, HR1) | every shard holds the generator (SL10 clause 2); the owner alone holds the edits; observers are served through the gateway, which is not a realm (SL2 clarification 2026-08-24) |
| No velocity crosses upward; no parent sets a speed; no re-clamp | STANDS (movement contract) | the diff lane carries addresses and cells; the speed lead lives in the moon's own interest planner, which is not the crossing path (SL4) |
| One world; no reduced or test-only variant | STANDS (SL5) | M-2 and M-3's fixtures are player-authored rows in the one world's store, never a second generator |
| No drift is a gate, not an argument | STANDS (SL10) | M-8 compares `generate_summary` at every legal tier on x86-64 and aarch64, byte for byte |
| The client derives shape only, never state | STANDS (SL10 clause 7) | the row carries no pose (§3.2); the client places the realm by its composed `SceneRow.pose` |
| Six asks are listed under SL6 with data, direction, and the cost of doing without | STANDS | §7, A-1 to A-6. A-1's PAYLOAD must change under F-1; the ask itself is properly made |

---

## 3. The corrections, in order of weight

1. Rewrite §4.2 and A-1: the hull states a HULL-frame point, the moon resolves its own chunk (F-1).
2. Delete the `coordinator_of` step; the target is the parent (F-2).
3. Re-cite the attestation predicate as `crates/wire/src/session_flow.rs:647` (F-3).
4. Add the regime-handover bench and stop claiming "no jump" (F-4).
5. Add the G-IDENTICAL fixture on a moon and a hull (F-5).
6. Add an owner decision row for the deposit diff and the survey (F-6).
7. State the degrade when the disk stalls (F-7).
8. State the never-draw-finer-than-you-hold rule (F-8).
9. Correct the tombstone count to eight (F-9).

Nothing here asks the report to give up its design. F-1 makes the forward SIMPLER: the hull says what it
did in its own frame, and the moon, which already owns both the placement and the store, does the rest.
