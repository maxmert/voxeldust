# Verdict — the law refuter — "Grid family and the geometry seam"

**Report under test:** `docs/investigation/2026-09-07/01_grid_family.md`
**Lens:** every binding law and ruling. `CLAUDE.md` (HR1–HR6, SL1–SL7, SL9, SL10),
`docs/design/owner_decisions_2026-09-07_voxels.md` (SL10, V2.1–V2.9), the 2026-09-02 REACH ruling,
the 2026-09-01 visibility ruling, the 2026-08-27 seed ruling, the 2026-08-26 movement contract,
`docs/design/PLAN.md`, and SL8 (a seam is a defect).
**Method:** I read the code and I ran `grep`. Every finding below names a file and a line.
**Date:** 2026-09-07.

## The verdict

**REFUTED.** The grid family itself survives. Two load-bearing claims about the code are WRONG, and
both of them are claims that clear a law. One clears SL6 for a hull. One supports SL10. A third
statement contradicts another statement in the same report. Three doors stand open that the report
does not name.

The report must not go to the owner as it stands. The cube-sphere recommendation (D1) and the
1/8 m recommendation (D2) survive and may go forward. The SL6 clearance table (§6.1) and the landing
slice (§4.4) must be rewritten first.

---

## What survives

These I tried to break and could not.

- **SL1, the address.** The cell address `(body, face, tier, i, j, k)` names no position outside its
  own realm. A planet counts `k` from its own floor radius. No chain folds an absolute. *Example: the
  star system moves the home planet along its orbit, and the landing pad's cell address does not move
  by one bit.* STANDS.
- **SL2.** No occupant pose enters another realm's simulation. The grid adds no upward lane. STANDS.
- **SL3.** The parent authors WHERE a planet is; the planet states its own look at
  `crates/physics/src/worldgen/generate.rs:1232` (`look: Some(shell(taxon.radius_m))`), and the report
  keeps that. STANDS.
- **SL4.** `addr_of` is not on the crossing path. Containment reads a `Boundary`
  (`crates/core/src/geometry.rs:338-348`) and never asks which cell a thing is in. STANDS.
- **SL7 and SL9.** The report adds no per-child per-tick cost, no bounded child set and no scan.
  STANDS.
- **The movement contract.** No velocity crosses upward. No parent sets a speed. No re-clamp. STANDS.
- **HR3.** The two arms match on a geometry VALUE, not on a shard kind. `crates/sim/src/capability.rs:11`
  blesses that shape by name. STANDS.
- **`SceneRow` carries `RealmId`** (`crates/wire/src/channels.rs:378-379`), so the client reads the
  `body` field of the address off the row it draws, as §1.4 claims. STANDS.
- **The lattice facts.** `FINE_CELL_EDGE_M = 1.0/1024.0` (`crates/core/src/pose.rs:255`),
  `CELL_DOMAIN_MAX = i64::MAX / 2` (`pose.rs:455`), three live rungs (`pose.rs:477-487`), and
  `translated` normalises inside itself (`pose.rs:623-630`). All true. STANDS.
- **`rapier` is absent from `Cargo.lock`** (`grep -c rapier Cargo.lock` returns 0, MEASURED). **`noise`
  is declared and unused** (`Cargo.toml:78`; `grep -rn "noise::" crates/` returns nothing, MEASURED).
  Both true. STANDS.

---

## F1 — WRONG. The hull's grid domain is not in the field the report names, and it does not reach the client at all

**The claim.** §6.1, row 2: a hull's grid half-extent is *"the hull's slot box, already in the built
record's look (`built.rs:78`)"*. The report uses that sentence to clear SL6 for the identity grid.

**The code.**

- `crates/core/src/built.rs:76` is `pub bound: Boundary` — *"The box that decides who is inside it. The
  slot it was sold, not the size of its hull."*
- `crates/core/src/built.rs:78` is `pub look: Boundary` — *"How big it draws."*

The slot box is `bound`, at line 76. Line 78 is a different field with a different meaning. The
report's own §4.4 item 2 says the identity grid comes from `BuiltBody.bound`, so the report
contradicts itself inside four pages.

**Worse: the box does not reach the client today.** The type that carried a realm's boundary to the
client is `RealmShape`, and `crates/wire/src/channels.rs:350-353` states plainly that *"the client
never receives this type any more (`SceneRow` is the client's row)"*. `SceneRow`
(`channels.rs:378-391`) carries a realm id, a parent link, a pose and a look bag. It carries no
authority box.

**Why this breaks a law.** SL10 lets the client derive only what the SEED decides. A hull's slot is
sold by a spaceport to a player. It is live state, not a seed draw. So the client cannot derive the
identity grid's domain, and today nothing ships it. Putting it on the client-facing row is NEW DATA
and a wire change. SL6 says: ask first, default NO. The report clears SL6 by pointing at a field that
already ships, and it points at the wrong field.

*Example: a player buys a berth at the shipyard and starts a hull. The client must know the hull's
slot box before it can draw one block, because the box IS the identity grid's domain. The gateway
holds no such statement for the client today.*

**Fix.** Delete row 2 of §6.1. Write it as an explicit SL6 request: the datum is the built realm's
`bound` (`built.rs:76`); it travels from the built realm's shard, through the gateway, to the client;
the receiver cannot compute it, because a slot is sold and not seeded; without it the client cannot
bound the grid it draws. Then let the owner rule.

---

## F2 — WRONG. The shipped client does not link `vd-physics`

**The claim.** §10: *"the client already links `vd-physics` (`crates/client/Cargo.toml:22`)"*. The
report uses this to show that SL10's "one generator, two hosts" costs nothing new.

**The code.** `crates/client/Cargo.toml:20-22` reads:

```
[dev-dependencies]
# DEV-ONLY: scene tests build THE world; the shipped client never names a motion (SL4).
vd-physics = { workspace = true }
```

It is a DEV dependency. The comment on line 21 says so in words. The shipped client does not link
`vd-physics`. Only the client's own scene tests do.

**Why it matters.** The report treats an existing link as evidence that the generator already rides in
the client. It does not. Linking the generator into every shipped client is a real, new, unmeasured
cost: binary size, cold-build time, and the aarch64 no-drift gate SL10 V1.3 demands. The report's §7
list does not own that cost either.

*Example: a pilot's client on an aarch64 laptop must carry the same generator crate as the moon's
shard, or the boots and the drawn slope disagree. Nothing in the tree carries it there yet.*

**Fix.** Restate the row: the shipped client links the generator NOWHERE today; SL10 requires the
link. Add "the client binary's size and cold-build time with the generator linked" to §7 as an owed
measurement.

---

## F3 — BREAKS_LAW. §4.4 says the landing slice touches no store, and its own item 2 changes a stored record

**The claim.** §4.4 closes with *"Nothing in this slice touches a wire arm, a store or a realm
boundary."* Item 2 of the same list says `GridParams` becomes *"a seed-derived field on the body
record (`ShellGrid`) and … a field on the built record's slot box (`IdentityGrid` from
`BuiltBody.bound`)"*.

**The code.** `BuiltBody` derives `Serialize, Deserialize` (`crates/core/src/built.rs:59-61`) and the
crate's own test round-trips it through postcard (`built.rs:167-171`). It is a stored record. A new
field on it is a store change, and the store refuses a file written under a different stamp
(`crates/core/src/store_stamp.rs:256-267`).

**Fix.** Either keep `GridParams` off the records for the first slice and derive it from the realm's
seed and `bound` at each use, or delete the closing sentence and state the record change with its
stamp move.

---

## F4 — BREAKS_LAW (SL8). The report accepts a build refusal at eight places on every planet and never books it as a seam

**The claim.** §0.1: *"A hull is never built on planet cells, so a player who builds a ship never meets
a corner defect."* §1.2: *"A box prefab cannot be stamped across a corner. A blueprint is a relative
walk from an origin cell, and the walk returns a typed `CornerDefect` instead of overwriting a cell."*

**The problem.** The first sentence is true and it disposes of nothing. A player who builds a BASE
builds on planet cells. The report's own §1.2 example is *"a player builds a wall of stone blocks
along the equator of the home planet."* At each of the eight cube corners, at latitude ±35.264°, the
blueprint refuses. A refusal the player did not cause is one of SL8's eleven named seam kinds. §6.5
lists the seams and omits this one. It offers instead that *"the generator MAY put a landform there"*
— a mitigation, stated as a maybe, and not a cure.

**Why this is load-bearing.** D1 asks the owner to pick the grid family. The owner must sign with the
price in view, and the price includes eight regions per planet where the build tool says no. Today
that price sits in a prose aside, not in the decision register and not in the seam list.

*Example: a colonist lays a straight wall out from a spaceport. Forty kilometres later the wall
crosses a cube corner. The stamp returns `CornerDefect` and the wall stops. The colonist sees a wall
that will not continue, and nothing in the world explains why.*

**Fix.** Add the corner refusal to §6.5 as an ACCEPTED SEAM. Add it to §9 as its own owner decision,
with the options: accept the refusal; make the generator ALWAYS place a mountain, a crater or a sea at
the eight corners, so no flat build site exists there; or draw a per-body landform so the corners
differ per planet. State which one the recommendation assumes.

---

## F5 — UNMEASURED_AS_FACT. The warp constants' identity is called MEASURED and no measurement is owed

**The claim.** §1.3: *"The `k₁ + k₂ + k₃ == 1.0` identity holds exactly in `f64` (MEASURED in the
investigation; a `const` assertion pins it in the crate)."*

**The problem.** The measurement lives in another document, written on 2026-08-03, not in this tree.
The report is honest about exactly this for the cell-edge spread — §1.2 says *"UNMEASURED in this
tree — a measurement is owed, §7"* — and then drops that honesty here. §7 owes eight measurements and
this identity is not one of them. It is also not free: `k₃ = 1 − k₁ − k₂` is a rounded `f64`, and
whether `(k₁ + k₂) + k₃` returns exactly `1.0` depends on the evaluation order the crate writes. No
crate writes it yet.

**Fix.** Relabel it UNMEASURED and add it to §7 as measurement 9: a `const` assertion in the grid
crate, run on x86-64 and on aarch64, that the three constants sum to exactly `1.0` in the order the
warp evaluates them.

A second, smaller instance: §5.1 labels *"±0.476 ly at FINE"* as *"MEASURED by the doc's own
arithmetic"*. Reading a doc comment is not a measurement. The arithmetic from `i64::MAX / 2`
(`pose.rs:455`) is sound, so call it DERIVED.

---

## F6 — MISSING. §3.1.2 fixes the cell's arms with no room for smooth terrain (V2.1) or a tree (V2.2)

**The claim.** §3.1, consequence 2: *"a cell is `Empty`, or `Block(shape, orient)`, or `SubGrid` …
Nothing else is placeable."*

**What is missing.**

- **V2.1** asks for smooth, realistic terrain built of voxels, and says *"when we place voxels, the
  terrain should change accordingly."* A smooth surface needs a per-cell scalar or a per-cell vertex,
  not an occupancy bit. §4.3 names *"the smooth-surface extractor (V2.1)"* above the seam and never
  says what it reads. Nowhere does the report say how a placed block reshapes the surface around it.
- **V2.2** asks that a tree be ONE object on the surface, drawn as a tree by the client, with a shape
  the server knows for collision. A tree is not `Empty`, not a `Block(shape, orient)` from the ~20
  square shapes, not a `SubGrid`, and not a no-volume attachment (§3.5 is V2.5's, and a tree has
  volume). The three-arm list leaves a tree nowhere to live.

**Why it matters now.** §8 calls the cell address a one-way door that must shut before any P4 code. A
door that shuts with no slot for two of the owner's nine requirements is a door shut ON them.

*Example: a player fells a pine on a moon and then places a stone voxel in the stump's cell. Under
§3.1.2 the moon's shard has no arm to hold the pine before the felling, and no rule for how the stone
reshapes the hillside after it.*

**Fix.** Add a §3.1 sentence: this report's cell-arm list covers PLACEABLE state only. Name the two
arms the generator report and the saved-record report must add — a terrain arm carrying the
smooth-surface datum, and a landscape-object arm carrying a seed-parameterised object (V2.2). State
that the address in §1.4 is unchanged by either, so §8's door stays a door about the ADDRESS.

---

## F7 — MISSING. The report drops `PLAN.md`'s stateful `FrameSpace` seam without booking it as an owner decision

**The claim.** §0.5 states as settled that *"The geometry seam is `GridMapping`, a stateless value in
`vd-core`."* §5.4 states that no anchor exists on the server for the grid.

**The binding text.** `docs/design/PLAN.md` HR4 names the seam as *"a **stateful `FrameSpace`**
(`CartesianSpace` for ships/stations; `SphericalSpace` with a `SurfaceAnchor` + `reanchor()` and
`AnchorGen` fencing for planets …)"* and makes G-IDENTICAL *"the identical fixture on a Spherical AND
a Cartesian profile (with one fixture forcing a reanchor)"*. `DEFERRED.md:331` still owes that
fixture. `crates/sim/src/capability.rs:11,18` still names `FrameSpace` and `reanchor()` as the plan.

**The problem.** The report's reasoning is good — no `f32` sits on the authoritative path today
(`pose.rs:548-551`) — but it changes a binding spec by argument, in the answer-in-one-page section,
and sends only the LIBRARY question to the owner as D7. The seam's shape is the bigger of the two
questions, and it is not on the register.

**Fix.** Add D8: *"Does the geometry seam stay `PLAN.md`'s stateful `FrameSpace` with `reanchor()` and
`AnchorGen`, or become the stateless `GridMapping` this report proposes?"* Recommend the stateless
form. State that D7's answer decides whether an anchor comes back BELOW it, and that D8's answer
decides the fate of the owed reanchor fixture at `DEFERRED.md:331`.

---

## F8 — MISSING. SL1 clause 4 is not tested for the area realm, and single-writer ownership of a planet's cells is unresolved

**The claim.** §6.2 and D4: an `Area` realm names a planet's cells by running the planet's
`GridMapping` on *"(told berth + local)"*, and the told berth is *"the stamped, read-only reading SL1
clause 2 allows"*.

**What the report tests.** Clause 2 only.

**What it does not test.**

- **Clause 4 — one hop.** *"What you are told about yourself, you NEVER pass on."* A cell address the
  area computes from its berth CARRIES that berth, folded into `i` and `j`. The address is stored and
  it goes on a lane. The area therefore passes its own placement on, in a disguised form. The report
  does not name this.
- **Single-writer ownership.** `RealmId` is *"a persistence/ownership realm: the unit of single-writer
  durable state"* (`crates/core/src/pose.rs:35-36`). SL10 V1.6 says a diff comes from *"the realm that
  owns the cell"*. If a spaceport area holds diffs for cells addressed in the home planet's grid, then
  the planet realm and the area realm both write one address space. The report does not say who the
  single writer is, and it does not say how the gateway composes one picture when two realms answer
  for one cell.

*Example: a district shard and the home planet's shard both hold an edit for the cell under the
spaceport's landing pad. The player's client derives the planet's shape and must then apply two diffs
that disagree. Nothing states which one wins.*

**Fix.** Split D4. D4a: may an area name its parent's cells through its told berth, against SL1
clause 4? D4b: which realm is the single writer of a cell inside an area's box, and how does the
client learn which realm to take the diff from? Give each a recommendation.

---

## F9 — MISSING. SL10 lets the client derive the shape of a realm the 2026-09-01 ruling says nobody may draw

**The claim.** §0.2 and §1.4: the client computes the same address as the shard, on its own.

**The collision.** The 2026-09-01 visibility ruling says *"a dormant realm is never drawn by
anybody"*. SL10 says the client MAY derive the static shape from the seed. A client that holds the
generator can derive and draw a moon's hills while the moon's shard is asleep. The report never names
this. §2.3's example (*"when the first chunks arrive at rung 12"*) is silent on whether the chunks
ARRIVE from the shard or the client derived them.

**Fix.** Add one sentence to §6: the client derives the static shape ONLY for a realm the gateway has
already put in the window, so the running-realm rule of 2026-09-01 still decides WHAT is drawn, and
SL10 decides only WHO computes the pixels of a realm already admitted. If the owner wants the client
to derive ahead of the shard's spin-up, that is a new ruling and it belongs in §9.

---

## F10 — Citation drift. Four line references point at the wrong item

The types all exist. The line numbers do not.

| The report says | The code says |
|---|---|
| `RealmId::Area` at `pose.rs:44` (§1.2, §6.2) | `pose.rs:44` is `Station(u64)`; `Area(u64)` is `pose.rs:47` |
| `FrameRef::AreaLocal` at `pose.rs:101` and `pose.rs:101-102` (§1.2, §6.1) | `pose.rs:101` is a doc line for `StationLocal`; `AreaLocal` is `pose.rs:105` |
| `StationLocal` at `pose.rs:106-107` (§1.4) | `StationLocal` is `pose.rs:102`; lines 106-107 are `StarCentered` |
| `VoxelGeometry` at `capability.rs:33-40` (§4.1) | the enum is `capability.rs:37-42`; lines 33-36 are its doc |

**Fix.** Re-run every citation. A report whose job is to be the only current truth cannot cite a
neighbour.

---

## The table in §2.3 prints an `N` its own rule forbids

This is not a law break, but the owner will read this table. §2.2 rules that `N` is a multiple of
`2^(T−1)`. The table's `N` column prints the UNSNAPPED value: Earth's `10,007,543` is not a multiple
of `4,096`, and the moon's `314,159` is not a multiple of `128`. The snapped-radius column IS computed
from the snapped `N` — Earth's `6,370,354 m` comes from `N = 10,006,528`, which is the number §2.3's
own worked example prints.

I re-derived every row by hand (ESTIMATED, 2026-09-07). Every rung count, snapped radius, error and
chunk count is arithmetically right, and the snap rounds to the NEAREST multiple (Luna's `+310 m`
proves it). Only the `N` column is the pre-snap number under a post-snap heading.

**Fix.** Print two columns: `N` from the seed, and `N` snapped.

---

## What the owner should do with this report

1. Take D1 (the cube-sphere) and D2 (1/8 m shipped, 1/16 m reserved) forward. They survive.
2. Send §6.1 and §4.4 back. They clear SL6 and the store with claims the code refutes (F1, F3).
3. Rule on the corner refusal as a named seam BEFORE signing D1 (F4).
4. Add D4a, D4b and D8 to the register (F7, F8).
5. Require F2's and F5's measurements before the first grid code lands.
