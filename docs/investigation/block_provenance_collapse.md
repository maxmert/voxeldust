> ### ⚠ INVESTIGATION BASE — NOT A DECISION RECORD
>
> **This document is input to an investigation, not the output of one.** It is analysis produced to
> explore a problem space, and it is deliberately more confident in tone than its status warrants —
> that was useful for finding defects and is misleading for planning.
>
> **Nothing here is committed.** Every ruling, recommendation, number and "settled" verdict is a
> *proposal to be re-validated by the pre-implementation investigation for its phase*, including the
> owner rulings recorded at the top of `block_system_design.md`, which record the owner's direction at
> the time rather than a frozen commitment.
>
> **The binding specs are elsewhere:** `docs/design/PLAN.md`, `docs/design/roadmap.json`,
> `docs/design/integration.json`, `docs/design/DEFERRED.md` and the hardened subsystem designs in
> `docs/design/`. Where this document and a binding spec disagree, **the binding spec wins** until an
> investigation says otherwise.
>
> **What this document IS good for:** the prior art it collected, the arithmetic it did, the failure
> modes it found, the one-way doors it named, and the questions it framed. Reuse those. Re-derive the
> conclusions.
>
> See `docs/investigation/README.md`.

# Block provenance and structural collapse — the three-state ruling

**Status:** settled design, owed by owner ruling **R6**. Binding on §2.5.2, §2.5.3, §2.7, §3.5.1 and
§3.9.3 of `docs/investigation/block_system_design.md`; those sections are stale where they disagree with this file.
**Date:** 2026-08-03.
**Phases:** the encoding lands at **P4** (before the first world is written); the collapse machinery lands
at **P6** with block edits; the ship arm is proven at **P6**, not deferred to P8.

---

## 1. What the owner asked for

R6, verbatim:

> *"terrain should not break under weight of gravity, but ideally biom and built things might."*

The ruling row that records it:

> **R6 — Provenance becomes THREE-state, not two: terrain / biome feature / player-placed.** Base terrain
> never collapses; **biome features** (a tree, a boulder, a rock arch, an ice shelf) collapse like built
> structures when cut. This spends one of R1's reserved bits — the first deliberate use of the slack, and
> exactly what it was reserved for.

### What changes as a result

| # | Change | Where it lands |
|---|---|---|
| 1 | The one-bit natural-versus-built marker in the durable eight-byte record becomes a **two-bit** field with three defined values and one rejected value. Reserved bits go **15 → 14**. | §2.7.1, §3.9.2, §8.6 item 1 |
| 2 | §2.5.2's *"generator-authored blocks are unconditionally anchored, and within thirty-two blocks of an edit one loses its anchor if it lost a generator-authored neighbour"* is **retracted outright**. Terrain is unconditionally anchored, everywhere, permanently, with no radius and no exception. | §2.5.2 |
| 3 | A **feature group** becomes a first-class concept: an authored, seed-derived, bounded set of cells that is **rigid while intact**. It is what lets an arch or an ice shelf exist at all. | new, this file |
| 4 | The support flood becomes **anisotropic** — free along the gravity axis toward the anchor, attenuating sideways. Without this, R6 caps every tree, tower, mast and antenna in the game at thirty-two blocks tall. | §2.5.2, §2.2 |
| 5 | `MAX_SUPPORT_RADIUS` **keeps** its §2.2 job (the lateral cantilever cap) and **loses** its §2.5.2 job (bounding the flood). The flood is bounded by the existing carry-over cell budget. | §2.5.1, §2.5.2 |
| 6 | §3.5.1's seven-stage generation pipeline gains an owed **stage 8 — feature placement**, running after carvers, water and ores, with an attachment validator. | §3.5.1 |
| 7 | `SupportModel::{Anchored, GridRigid}` **collapses into one algorithm** whose load term is proportional to the realm's declared gravity magnitude. At zero gravity it degenerates, with no branch, into Space Engineers grid rigidity. | §2.5.3 |
| 8 | A falling group is a **server-authored scripted topple with a closed-form final pose**, not a rapier rigid body. | §2.5.2 |
| 9 | §2.3.7's harvest-tier drop gate is amended: it gates **tool-carrying sources only**. | §2.3.7 |
| 10 | The chunk record gains a **prune-on-equality rule** that compares provenance. This is what lets R7/R8 regrowth reclaim its storage, and simultaneously what blocks a break-and-replace laundering exploit. | §3.9.3 |

---

## 2. The ruling in one page

*(No technical names on this page. This is the page to read.)*

Every block in the world now carries a small mark saying where it came from. There are three marks, and
they mean three different things about gravity.

**The ground never falls.** The crust, the mountain, the seabed, the cave wall — anything the world
generator laid down as landscape — is held up by the planet itself and is never subject to weight. You can
tunnel through a mountain, hollow out a hillside, dig a room the size of a cathedral, and nothing above it
will ever come down. Mining is permanently safe. There are no cave-ins.

**Things that grow or sit on the ground do fall.** A tree, a boulder, a rock arch, an ice shelf, a coral
stack, a hanging vine — anything the generator placed *on* the landscape rather than *as* the landscape —
is a single recognised object. While it is whole, it is rigid: it stands on whatever it is attached to,
and it stands as one piece, which is why an arch can span a gap that a loose pile of the same stone never
could. It stops being whole the moment you cut through the part that holds it together.

**Two different things can happen to it, and you can tell which by what you did.**

*Cut the trunk.* The tree stops being one object. Everything above your cut has nothing under it, so it
comes away as a single piece, topples over, and lands lying on the ground. The branches and leaves burst
apart on impact and you collect them; the trunk stays as logs you can pick up. What lands is exactly what
came down — nothing is created and nothing quietly disappears.

**Mine under the boulder.** You never touched the boulder itself, so it is still one whole object — but
the ground it was standing on is gone. It drops, as one piece, into the hole you dug. If it lands on
something breakable, it breaks it.

**Everything a player builds behaves the same way as a biome feature that has been cut**, block by block:
a beam can only reach so far sideways before it gives way, but a tower can be as tall as you like, because
weight travels straight down through a column without limit. That is how real structures behave, it is how
the best building games behave, and it is the difference between a game where you can build a lighthouse
and one where you cannot.

**Three consequences the owner should see now rather than discover later.**

First, because the ground never falls, you can mine out an entire hillside and leave a single square metre
of earth under a tree, and the tree will stand on it forever. It will look silly. It is the direct and
unavoidable price of "terrain should not break under weight of gravity".

Second, felling a tree is a physical mechanic, not a shortcut. Cutting one block at the base and then
picking up eleven logs off the floor is only about twice as fast as harvesting the standing tree
block-by-block. Felling looks and feels right; it does not make wood cheap. If wood *should* get cheaper,
that is a separate decision about how a felled tree is processed, and it is in the register below.

Third, clearing forest is by far the most expensive thing a player can do to the world's saved storage —
about two kilobytes of permanent record per tree. Clear-cutting a patch a quarter of a thousandth of the
planet's surface would fill the planet's entire storage budget. This is precisely why the owner's own later
ruling — that cleared land recovers slowly by itself — is not a flavour feature but a requirement: when a
forest grows back to what the world originally said was there, the record of it being cleared is deleted
and the storage comes back.

**Nothing about this exists on a ship.** A ship has no landscape, so nothing on a ship carries the first
mark at all; every block on a ship is either something the generator put in a derelict or something a
player welded. A ship in flight has no gravity, so nothing on it has weight and nothing on it ever falls —
but cut a ship in half and the severed piece separates and drifts away, which is the same rule with the
weight set to zero. The same code decides both, which is why the same test passes on a planet and inside a
hull.

---

## 3. The encoding

### 3.1 The eight-byte durable record, exact

R1 fixed the record at eight bytes with fifteen reserved bits. §8.6 item 1's adjudicated synthesis — which
R1 adopted — states the field **order** verbatim: *"eighteen bits of cell index, sixteen of block identity,
six of orientation including the mirror flag, one generated-versus-placed marker, eight of
material-interpreted state, fifteen reserved and rejected on decode if non-zero."* Eighteen plus sixteen
plus six plus one plus eight is forty-nine; sixty-four minus forty-nine is fifteen. The layout is therefore
already determined up to the single bit R6 spends, and the positions below are read off R1's own ruling
rather than invented here.

```
crates/core/src/block/record.rs — BlockRecord(u64). PERSISTED. PERMANENT. ONE-WAY DOOR.

  bits 63..46  cell         (18)  CellIndex within a 62³ chunk, 0..=238_327
  bits 45..30  block_type   (16)  BlockTypeId
  bits 29..24  orient        (6)  24 proper rotations + mirror bit (§4.1.2)
  bits 23..22  provenance    (2)  <-- R6.  WAS: bit 23 = `placed`, 1 bit
  bits 21..14  state         (8)  material-interpreted (§8.6 item 1)
  bits 13.. 0  reserved     (14)  MUST be zero; decode REJECTS non-zero.  WAS 15.
  18 + 16 + 6 + 2 + 8 + 14 = 64.
```

**A consistency check that falls out and is worth recording.** §3.9.2 publishes the record as
`cell 18 | block state 32 | reserved 14` and delegates the state word's internal split to §4. Before R6 the
declared fields fill only thirty-one of those thirty-two bits, which is exactly why R1 counts *fifteen*
reserved rather than fourteen. Spending the provenance bit makes the state word exactly thirty-two bits and
the reserved run exactly fourteen — so **R6 reconciles §3.9.2 and §8.6 item 1, which have differed by one
bit since the record was first written.** The bit R6 spends is the bit §3.9.2 had already accounted for.

**Marginal cost: zero bytes.** The record's width was already fixed at eight. The reserve goes 15 → 14.
**Zero wire cost:** the eight-byte record *is* the delta payload in both of §3.9.3's encodings, so there is
no new field, no `PROTO_MINOR` bump and no new `InterShardFlow` arm. **Zero pyramid cost:** §2.4.3 already
excludes the natural-versus-built marker from `CellSummary` on the grounds that support is tier-0 and
edit-local; widening it to two bits changes none of that, so the ladder's one-way door stays shut.

**The palette-entry projection is one shift and one mask.** Bits 45..22 are exactly twenty-four contiguous
bits of `(block_type | orient | provenance)`, which is precisely §2.4.1's `PaletteEntry`. It follows that
`PaletteEntry` should be a twenty-four-bit-valued `u32` newtype rather than a three-field struct with a
padding byte — palette equality becomes a single integer compare, and the structure is persisted inside
§3.9.3's masked-dense encoding, so removing the pad is a real saving. That is §2.4.1's and §3.9.3's change
to take, not this file's to impose; it is raised as **[USER DECISION R6-14]** below.

### 3.2 The values, and why this numbering

```rust
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Provenance { Terrain = 0, Feature = 1, Placed = 2 }
```

Value **3 is reserved and REJECTED on decode** — never a Default, never a clamp. It is also the coverable
negative branch HR5 wants, and it is the escape hatch that makes a fourth class later (candidates:
`NpcBuilt` for the dormant-world pillar, `Ruin`, `Grown`) an epoch-gated additive change rather than a
record migration. Only a *fifth* class would cost a format change, which is why two bits beats three.

**The numbering is chosen so that a single integer comparison enforces the model's central invariant.**
Order the three values by **anchor privilege**, descending: `Terrain` has an unconditional anchor; `Feature`
has group rigidity; `Placed` has neither. Then:

> **PROVENANCE NEVER GAINS PRIVILEGE.** For every write to an existing cell, `new >= old`. `Terrain` is
> written by a realm's generator and by nothing else.

That is one `>=` in one place — a typed constructor in the block store's write API — rather than a rule six
independent write sites (settle-and-reintegrate, itemise-then-replace, dormant NPC construction, blueprint
paste, admin revert, R7/R8 regrowth) each have to remember. It is the same move `Tier0Key` made when *"only
tier 0 is writable"* stopped being a convention and became a type.

It also closes, structurally rather than by policy, the one exploit that a mutable provenance field would
have opened on day one: **a player cannot launder built matter into unbreakable terrain.** Shove crust into
a wall, bury your base in dirt, break-and-replace a granite cell — none of it can produce `Terrain`, because
no edit path can construct that value.

The rival numbering (`0 Terrain, 1 Placed, 2 Feature`) was argued from backward compatibility with saved
worlds, on the grounds that bit 0 would then be the old one-bit marker. Verified in the worktree:
`crates/core/src/` contains `celestial`, `collections`, `entity_kind`, `fence`, `frame`, `geometry`, `ids`,
`incarnation`, `kinematics`, `pose`, `realm_coord`, `realm_path`, `rng`, `taxonomy`, `tlv`, `worldgen` — there
is no `block/`, no `grid/`, no `voxel/`, no `chunk/` anywhere in the workspace, and no world format has ever
been written. There are no old records. The argument buys nothing; the monotonicity property is real and
permanent.

### 3.3 The decode rule

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown block provenance {0}")]
pub struct UnknownProvenance(pub u8);

impl Provenance {
    pub const ALL: [Provenance; 3] = [Self::Terrain, Self::Feature, Self::Placed];
    pub fn from_bits(v: u8) -> Result<Self, UnknownProvenance> {
        match v { 0 => Ok(Self::Terrain), 1 => Ok(Self::Feature),
                  2 => Ok(Self::Placed),  other => Err(UnknownProvenance(other)) }
    }
}
```

Monomorphic, exhaustive match, no bounds-checked array index (the uncoverable panic path HR5 punishes). The
`ALL` loop covers three `Ok` arms; one `expect_err` covers the `Err` arm. Three placements, mirroring
§2.8.2/§2.8.3 exactly:

| Site | Behaviour |
|---|---|
| **Wire** | an unknown provenance in a delta is a protocol violation and **tears the connection** |
| **Persistence** | an unknown provenance in a WAL frame or a `ChunkRecord` makes the store **refuse to open** — it does not skip the record, because a silently dropped edit is a lost player build and P6's DoD requires edits to survive `kill -9` and a re-shard |
| **Generation** | structurally impossible: the generator constructs values and never decodes them |

Note the deliberate contrast with `entity_kind.rs`'s `continuity_of`/`durability_of`, which *do* resolve an
unknown tag to a conservative default because they run on a possibly-corrupt `EntityId` in a hot path where
a panic is worse. **There is no conservative default here.** "Some provenance" cannot decide whether a
mountain falls.

### 3.4 Two canonicalisation rules, both required for determinism

1. **An `Air` record carries provenance 0 and state 0.** Any other value is rejected on decode. A break is
   an edit to `BlockTypeId 0` (§2.8.3); air has no provenance, so without this rule two shards could write
   different bytes for the same break and P6's *"deterministic WAL byte output (sorted drain)"* scenario
   would fail. It is safe because §3.9.3's masked-dense authored-cell bitmask already records *which* air
   was dug, so nothing needs provenance to distinguish player-dug air from natural air.

2. **`ChunkRecord` prunes on equality, and the equality compares provenance.** An authored cell is deleted
   from the record iff its full eight-byte state — type, orientation, **provenance** and state — equals what
   the generator produces at that address. This is the same principle §2.4.3 already applies to the pyramid
   (*"an entry is deleted iff it equals `generate_summary` of its own address"*), extended to the delta,
   and it is required twice over:
   - **R7/R8 depend on it.** The owner's ruling that natural recovery *"drifts toward the generated baseline
     and prunes its own delta as it succeeds"* is only true if this rule exists. §3.9.6's current delta
     lifecycle is budget/revert/reap and never deletes a record; §3.9.3 has no prune rule at all. This adds
     the missing one.
   - **Without the provenance clause it is an exploit.** Break a `Terrain` granite cell, place granite back:
     the type and orientation match the generated cell, the record prunes, and the cell reverts to
     unconditionally anchored terrain. Players would build never-collapsing bases out of restored crust. A
     `Placed` cell can never equal a `Terrain` cell, so it can never prune.

### 3.5 What happens to the existing one-bit marker

`PaletteEntry.placed: bool` (§2.4.1) and the record's `p` bit (§2.7.1) both disappear as booleans.
`PaletteEntry` stays **four bytes** — not one memory figure in §2.4.1's table moves.

**The identifier `placed` must not survive anywhere in the codebase.** A reader who sees a boolean will
assume two states and will silently treat every feature as terrain. Two accessors replace it, named apart on
purpose so the collapse test and the authorship test can never be confused:

- `Provenance::anchors_unconditionally() -> bool` (`== Terrain`) — **the collapse test**, and the only one
  the support kernel may call.
- `Provenance::is_player_authored() -> bool` (`== Placed`) — for drops, claims, salvage and economy.

A grep-level tripwire test enforces the retirement, in the same way the registry drift tripwires already work.

**Provenance must not be overloaded as "anchor."** Anchoring is what a realm's support model *does* with
provenance, not what provenance *means*: a station foundation, a claim beacon and a ship's grid root are all
anchors without being terrain. `AnchorSource::{TerrainProvenance, GridRoot}` stays a capability-selected data
row (§2.5.3's existing idiom) so that nobody later adds an `Anchor` provenance value and fuses two orthogonal
concepts inside a persisted field.

### 3.6 Ships and stations — the degeneration

The **format does not change** and no code branches on shard kind (HR3/HR4). What narrows is the **legal
domain**, derived in `ShardProfile::build`'s existing validated lattice in the same idiom as
`signal_graph = req.signal_graph || req.functional_blocks`:

```rust
terrain_admissible = req.terrain_generator.is_some()
```

Deliberately keyed on *"does this realm have a seed-driven terrain generator"*, **not** on
`VoxelGeometry::{Spherical, Cartesian}`: a Cartesian asteroid realm with a generator legitimately admits
terrain, and a station built on a planet's surface is *in* the planet's realm where terrain is admissible.

Where `terrain_admissible` is false, a write of `Provenance::Terrain` is **rejected — a covered branch, the
HR5 negative test** — rather than being unrepresentable in the format, which would make the record differ by
realm kind and be a straight HR3 violation.

Provenance earns its keep on a hull realm three ways, in descending order of certainty:

1. **P8 ship interiors are persisted voxel realms**, and procedurally generated derelicts and wrecks are
   generator-authored. They must be distinguishable from player work for loot, claim, salvage and
   *"did I build this or find it"* rules. They are `Feature`.
2. **A spin station runs the anchored support model.** Provenance still decides feature-versus-placed for
   drops and for the composite lane — a potted tree inside a ship should fall when its planter is cut — even
   though the anchor source is the grid root rather than terrain.
3. **HR4.** The identical fixture must pass on a Spherical planet profile and a Cartesian ship profile, and
   a format that varies by realm kind cannot pass it.

Where it is inert, it is free: a ship's records are effectively all `Placed`, a constant, and a constant
compresses to one palette entry per chunk.

---

## 4. The collapse semantics

### 4.1 The three predicates, stated exactly

There are **two** independent conditions on a feature, not one, and this is the finding on which the whole
design turns.

```
anchored(cell) :=
    provenance(cell) == Terrain
 || cell ∈ realm.anchor_cells                            // a ship's grid root, a station foundation
 || reached by the support flood from an anchor with remaining load capacity > 0

group_state(g) :=
    Intact  iff  every structural cell of g still holds a block whose substance is in g's declared set
    Broken  otherwise                                    // permanently: deltas are permanent

anchored(g : Intact) :=
    ∃ a ∈ g.attachment_cells :  anchored(neighbour_of(a, g.attachment_dir(a)))

// A Broken group is not a group. Its cells become ordinary matter and each is re-tested
// individually against anchored(cell) above.
```

| Class | While intact | When a structural cell is removed | When the ground under it is removed |
|---|---|---|---|
| **Terrain** | anchored, unconditionally, everywhere, permanently | nothing — its neighbours are still terrain | nothing |
| **Feature** | rigid: one node in the support graph, anchored via its own attachment cells | **group dissolves**; every survivor re-tested per cell → the part above the cut falls | still intact, now unfooted → **the whole group falls as one** |
| **Placed** | ordinary per-cell support, load-attenuated | the cut cell's dependents are re-tested | its dependents are re-tested |

The owner's two sentences are exactly the second row's two right-hand columns. **Chop the trunk and the tree
falls** is the *broken* case; **mine under the boulder and it drops** is the *intact-but-unfooted* case. They
produce visibly different results — a cut arch partially collapses, an undermined boulder topples whole — and
both are wanted.

**Why the group is semantically necessary and not an optimisation.** §2.2's derived table gives water ice a
cantilever of **6** blocks and granite **12**. A twenty-metre rock arch and any ice shelf worth the name both
exceed their material's cantilever by a wide margin. Under a pure per-cell rule — *"anchored equals terrain,
then flood with per-material attenuation"* — an arch and a shelf **cannot exist**: they stand only until the
first edit within seeding distance, then collapse. The group's entire job is to let an *authored* feature be
rigid and so exceed its material's cantilever, in exactly the way Space Engineers' grid rigidity lets a
player's ship exceed its material's cantilever. Two of the four features the owner named by name depend on it.

**Attachment is per instance, never per kind.** A per-kind policy (`Below` for a tree, `Above` for a vine,
`Any` for a shelf) breaks on the first tree growing horizontally out of a cliff face: it has air below and
rock beside, so it drops off the wall the first time anything near it is edited. The same failure hits a coral
stack on a vertical reef, a bracket fungus, and any instance the generator rotated onto a slope. Because a
feature instance is already re-derivable from the seed, the feature stage records the instance's own
attachment cells (1..=9 of them, each with its own direction) as part of the instance. Cost: zero persisted
bytes.

**Intactness is a content test, never a delta-existence test.** *"The structural footprint has an empty
intersection with the chunk's authored-cell mask"* is cheaper and wrong: a player who breaks one trunk log and
places a **stronger** block back in the same cell has created an authored cell inside the footprint, so the
group would break, dissolve, and the canopy — with leaf material at effectively zero cantilever — would
detonate. Repairing a tree would kill it. The correct test reads the cells' current contents. The fast path
survives unchanged: **a chunk with an empty authored-cell mask has every feature intact, by one comparison**,
because an empty mask implies no cell differs from what the generator produced.

**Cycles are impossible by construction.** Anchoredness is the least fixpoint of a forward flood from the
anchor set — a node is anchored iff it was *reached* — never a recursive per-node query. §2.5.1's existing
increase/decrease queue pair is exactly that kernel. Feature A footed on feature B footed on feature A simply
never gets reached.

### 4.2 The support flood, amended — and the defect R6 activates

§2.2 derives `cantilever_blocks` from flexural strength as the classic **cantilever length of a
one-metre-deep beam failing in bending** — a *horizontal* span. §2.5.2 then spends that number as an
**isotropic** flood attenuation bounded by `MAX_SUPPORT_RADIUS = 32`.

Under the old binary rule this never bit, because generator blocks were unconditionally anchored and only
player builds conducted. R6 makes it bite immediately and on generated content: **a forty-metre pine, a
hundred-metre redwood, a sixty-metre mast and every player tower above thirty-two blocks are unsupported at
the top and shed their crowns while standing, unedited.** Physically a column carries axial load
indefinitely; only lateral reach is bending-limited.

> **AMENDMENT.** The flood's per-step cost is **0** when the step moves the cell closer to its anchor along
> the gravity axis, and **1/`cantilever_blocks`** otherwise.

One sign test against a vector the mechanic already reads. It is what §2.2's derivation already *means*, and
it is what shipped structural-integrity systems do: 7 Days to Die's model is *"vertical stability is
infinite — you can stack 250 stone on a single building block"* against *"a beam longer than eight blocks
will partially collapse; it is not possible to build a beam longer than fifteen"*. Unlimited vertical,
limited horizontal, is also what players expect.

> **AMENDMENT.** `MAX_SUPPORT_RADIUS = 32` **keeps** its §2.2 role as the lateral cantilever cap and
> **loses** its §2.5.2 role as the flood's bound. The flood is bounded by §2.5.1's existing
> `budget_cells_per_tick`, which carries over across ticks.

A radius cap on a **decrease** flood does not make a wrong answer slow; it makes it wrong. An eight-thousand
block tower or a fifty-thousand block hull is one connected component reaching far past thirty-two blocks
from any edit, and truncating the decrease flood leaves cells beyond the cap holding a **stale** support
level — floating blocks, which will be chased as a rendering bug.

### 4.3 The cost, re-derived — R6 makes support dramatically cheaper

**Terrain is a source, not a conductor.** The flood may not propagate *through* an unconditionally anchored
cell, so the visited set is the touched non-terrain connected component plus its boundary.

| Operation | Cells visited | Former bound |
|---|---|---|
| Mining one cell of plain rock (six terrain neighbours) | **≈ 7** | ≤ 45,825 |
| Cutting the reference oak (162 cells) | ≈ 162 interior + ≈ 500 boundary = **≈ 662** | ≤ 45,825 |
| Collapsing a 20×20×20 player tower | 8,000, paced at `collapse_frontier_cells_per_tick` = 256 → **32 ticks = 1.6 s** | ≤ 45,825 |

The 45,825 figure is §2.5.1's taxicab ball at radius 32, reproduced exactly: `(2·32+1)(2·32² + 2·32 + 3)/3
= 65 × 705 = 45,825`. **The commonest operation in the game gets 6,500× cheaper.** The 1.6-second tower is
§2.5.2's own *"a cathedral comes down over two seconds"* target, now derived rather than asserted.

**A broken feature seeds over the group's own cells, not the radius ball.** A forty-metre redwood cut at the
base has cells thirty-two blocks up the trunk that a radius ball would miss, leaving a floating crown. Since
a group is bounded at 1,024 cells, seeding over the group's own cell set is both **correct** and **45×
cheaper** than the ball. Rule: a structural break seeds `(the group's cells) ∪ (the six-neighbourhood of the
removed cell)`, and only the second half is bounded by distance. This also decouples maximum feature height
from `MAX_SUPPORT_RADIUS`, which would otherwise silently cap how tall a tree the art team may author.

### 4.4 The generation-time hazard, and the stage that does not exist yet

§3.5.1's pipeline is **body definition → height field → strata → biomes → water → carvers → ores**, and it
places no tree, boulder, arch, shelf or coral. **R6 owes stage 8.**

> **Stage 8 — feature placement.** Per feature region (one 62-block tangential cell of the direction sphere,
> the same idiom the carvers already use), `SplitMix64(seed, region)` draws `0..n` instances, each a
> `{origin_cell, kind, size_class, rotation, attachment_cells}`. Runs **after** carvers, **after** water and
> **after** ores, and tests each instance's attachment cells against the **post-carver, post-water** field.

Order is not a preference. Run it before the carvers and a cave carved under a tree leaves it floating —
permanently, in generated output, in every world produced by that seed.

**Three deterministic rejections inside the stage**, each producing a relocation rather than a bad instance:

1. **Unanchored attachment** — the instance's attachment neighbour is air or water in the post-carver field.
2. **Cube-face seam crossing** — §2.7's rule is *"a chunk never straddles a cube face"* and orientation is
   expressed in the owning face's triad. A *feature* could straddle one, and its footprint arithmetic would
   then cross two bases. Rejected and relocated.
3. **Size cap violation** — asserted at content build per size class, so this arm is unreachable at runtime
   and exists as a belt to the braces.

**Who validates, and who does not.** The generator's own deterministic test suite validates; the runtime
never does. A runtime check would be a per-spin-up world scan, which is the single cost this whole design
exists to avoid.

- `assert_features_anchored(seed, region)` — a Tier-A property test sweeping ≥ 1,000 regions per biome
  against pure generator output with an empty delta. Zero unanchored instances.
- `vdctl world verify-features <realm> --regions N` — the whole-body sweep, run in the harness after every
  generation scenario.

**Prior art confirms the shape.** Vintage Story generates a layer of hardened rock around naturally generated
caves and overhangs specifically so that generated geometry is not inherently unstable under its cave-in
rule, and it caps an unstable block's instability at the stability of the block below so a player cannot be
killed by blocks he cannot see. The first is our attachment validator; the second is reserved below.

**The honest limitation, stated so it is seen rather than discovered: provenance is an *authoring* fact, so
the generation stage decides what collapses, not the visual category.** A rock arch that the *cave carver*
happens to cut out of a ridge is terrain and will never fall; an arch the *feature stage* authored is a
feature and will. The alternative — deriving the class at generation time from a connectivity/extent test —
is rejected on measured cost and on correctness: the test is not chunk-local (a body spans chunks, so it needs
at least a one-chunk apron ≈ 1 M extra cell visits against Appendix A's measured **0.885 ms per 62³ chunk**,
a 2–3× generation cost) and it gives the wrong answer for genuinely large features such as a two-hundred-metre
mesa. Keep the authoring rule; the feature stage's catalogue is the knob.

**A generator-version rule, free now and expensive later.** Because §3.9.3 stores only authored cells, the
provenance of ~99.99 % of the world is re-derived from the seed on **every** load. Changing which cells the
feature stage authors therefore retroactively changes what collapses *under things players have already
built* — a house whose foundation rests on a rock arch loses its foundation when a content patch reclassifies
the arch.

> **Across a `gen_schema_version` bump, provenance may only be RELAXED (`Feature → Terrain`), never tightened
> (`Terrain → Feature`).** Relaxing can only make things stop falling. Enforced as a content-build assertion
> against the previous catalogue.

**Fold the feature catalogue and its provenance assignment into `BlockRegistryHash`**, alongside `coarsen`
and `COARSE_SHAPE`. Otherwise a client and a server can generate byte-identical terrain and disagree about
which cells are features — and therefore about what falls — a divergence the byte-identity gate cannot catch,
because the block ids match.

### 4.5 What a falling group becomes

> **A falling group is a server-authored scripted topple with a closed-form final pose, followed by an
> integer settle. It is not a rapier rigid body.**

This overturns §2.5.2's *"merge contiguous groups into ONE rigid body (SE grid-split)"* for the
gravity-bearing case, and keeps it verbatim for the gravity-free one (§4.9).

**The topple.** The group rotates about its hinge edge on a closed-form integer schedule:
`topple_ticks = TOPPLE_K · isqrt(height_mm · 1000 / g_mm_per_s²) / 1000`, calibrated so a twelve-metre tree
falls in **50 ticks = 2.5 s** at one standard gravity — which is what a real tree of that height takes. The
angle is quantised to 1/1024 radian. The server ships the group's pose on the existing entity lane each tick;
the client renders it. No physics body, no collider, no float.

**The final pose is closed-form.** The topple azimuth is a deterministic draw from
`(realm_seed, group_origin_cell, the fence of the edit that broke it)`, quantised to the four cardinal
directions of the chunk basis — so the settled rotation is one of four of the twenty-four proper rotations,
and the translation is the integer march along the gravity direction to first contact. **The final pose does
not depend on the path.**

That single property is the decisive argument for the scripted topple, and it dissolves a problem all three
candidate designs accepted as unavoidable:

> **The live path and the dormant path write IDENTICAL BYTES.** A realm with a player in it animates the
> rotation; a dormant or coarse-resident realm jumps straight to the same final pose. There is no
> observed-versus-unobserved world divergence, and a fixture can assert exactly where a tree lands.

A rapier body cannot have this property. Quantising its settle to the lattice makes the *output* stable but
not *reproducible* — near a snap boundary a one-ulp difference flips the rotation — and Category C says
physics state is host-specific and never re-simulated cross-host. Every design that reached for a rigid body
also had to reach for a separate analytic settle for the dormant case and then accept that the two produce
different worlds.

**What the scripted topple costs.** A boulder undermined on a slope translates down and settles rather than
rolling fifty metres. That is a quality loss, bounded and legible. The rigid-body path is **reserved as a
data-selected quality tier** (`FallMotion::{ScriptedTopple, RigidBody}` on the substance row, `RigidBody`
unused in v1) and is explicitly gated on **[USER DECISION 2-G]** — the parry sparse `Voxels` collider
question. **This design removes a second, independent argument from that spike**, which can now be decided on
terrain collision alone.

**The settle, and why matter cannot be created.** The settled pose is one of the twenty-four proper rotations
composed with an integer lattice translation. That map is a lattice automorphism composed with a translation,
therefore **injective on cells**, therefore `|writes| + |drops| == |group cells|` exactly. Cells whose image
is occupied or outside the realm become dropped stacks rather than being discarded, so it cannot silently
destroy matter either. `settle(group, pose) -> (Vec<GridWrite>, Vec<DropStack>)` is a pure integer Tier-A
function and conservation is a unit assertion, not a harness property.

> **The obvious wrong implementation, rejected by name so nobody writes it: re-voxelising the rotated body by
> sampling its volume.** Rotating a 1×1×20 trunk ninety degrees and re-sampling yields 19, 20 or 21 blocks
> depending on the epsilon; it needs a transcendental for the rotation, which the brief lists as *the*
> determinism killer; and it is a matter faucet and a matter sink at once.

**The landing outcome is resolved PER CELL from a new authored column**, `fall_outcome: FallOutcome` on the
substance row — **not** a reinterpretation of `shatter_frac_q12`. This matters: §2.3.3's family table gives
Metal/Organic/Granular `shatter_frac_q12 = 4096`, which reads as *"never shatters — it just breaks"*, so wood
cannot shatter under the **damage** model. Landing is not a damage outcome; it is its own policy, and giving
it its own column lets leaves shatter on landing while staying Organic for damage, with no contradiction and
no repurposing.

| Outcome | Applies to | Behaviour |
|---|---|---|
| `Reintegrate` | logs, hull plate, stone blocks; groups ≥ `reintegrate_min_cells` = 4 | written back as ordinary edits at the settled pose, keeping their own provenance |
| `Shatter` | leaves, ice, glass | destroyed on landing, yielding per the drop table at the declared yield |
| `Itemise` | groups below the minimum; substances flagged portable or granular | coalesced into `DroppedBlock` stacks under `debris_merge_radius_blocks` and `max_debris_entities` |

Resolving per cell is what lets one falling tree reintegrate its trunk and shatter its canopy — which is both
what every game that does this well ships and, as §5 shows, a 34 % reduction in permanent delta.

### 4.6 Landing damage

The landing delivers an ordinary `Hit` with `HitKind::Impulse` to the cells the group lands **on** — never to
the group's own cells, which re-insert or shatter per their own outcome row. That keeps conservation
structural rather than hostage to the drop gate, and it means there is no second damage model.

```
E_kJ        = Σ(cell mass_g) · g_mm_per_s² · drop_mm / 10^12       // integer throughout
matter_dp   = min(E_kJ · FALL_DP_PER_KJ / contact_cells, fall_impact_max_dp)
```

**`FALL_DP_PER_KJ = 5` — the fifteenth `GameScale` constant**, derived from one sentence a player would
recognise, exactly as the other fourteen are:

> *A one-cubic-metre granite boulder dropped ten metres onto a granite floor breaks the cell it lands on, and
> not the one beside it.*

Granite: 2,700 kg, `m·g·h` = 264.9 kJ, one contact cell, integrity 1,198 dp. `1198 / 264.9 = 4.52` → **5**.
264.9 × 5 = 1,325 dp ≥ 1,198, and ≥ the shatter threshold of 359, so the floor cell **shatters** and yields
granite's `shatter_drop_q12` of 1024/4096 = 25 %. Dropping boulders on things destroys most of the material —
a legible trade, and the same shape as §2.3.7's existing sledgehammer-versus-drill trade.

**Worked the other way**, so the constant is visible from both ends. The reference oak (§5) masses
11 × 700 + 150 × 150 = **30,200 kg**; its centroid falls from ≈ 7 m to ≈ 0.5 m, so `E` = 30,200 × 9.81 × 6.5
= 1,926 kJ; spread over the ≈ 50-cell swept footprint that is **193 dp per cell**. Against an oak plank
(1/8 volume, integrity 1,498) that is 13 % — a dent. Against a glass pane (integrity 23) or packed soil
(169) it goes straight through. **In this game's physics a felled tree does not destroy a wooden roof**,
because oak's work of fracture makes it ten times tougher than granite; it destroys a glass one. That is a
consequence of §2.2's material model, not of this constant, and it is self-consistent.

Overflow is carried downward by the **existing** spall path, bounded at `spall_max_depth` = 3. Falling matter
also damages actors through the same `Hit` object — Minecraft's anvil does 2 damage per block fallen and
dripstone 6, both capped at 40, which is the right order.

### 4.7 Chain reactions — bounded by content, not by a counter

Two mechanisms, one of which is desirable:

1. **Impact chain.** A falling tree's landing `Hit` breaks a cell of a second tree; that break is an ordinary
   edit, so it seeds the ordinary queue and the second tree falls a tick or two later. This is Valheim's
   shipped domino (*"with enough weight and momentum falling trees can take down other trees with them"*),
   it is wanted, and it is rate-limited for free because it rides the edit path.
2. **Support chain.** Removing the matter under a stack unanchors it, bounded by the flood.

**The real bound is subcritical branching, and it should be a content assertion rather than a runtime
counter.** §5/§6 give the forest density twice as **one tree per 100 m²**. A felled twelve-metre tree sweeps
roughly **12 m²** of ground. The expected number of second-generation structural hits per fall is therefore
**0.12** — a branching factor eight times below criticality, so a chain dies in one generation. Criticality
would need one tree per 12 m². Ruling:

- **Content build asserts `feature_density × fall_footprint < 1` per biome.** That is the thing an art-pack
  update could actually break, and a runtime counter would hide it rather than fail it.
- **`collapse_chain_depth_max = 4`** survives as belt-and-braces. At a branching factor of 0.12 the chance of
  reaching generation four is 1.7 × 10⁻³ per fell, so the cap is unreachable in normal play and exists only
  for an adversarial or content-broken case.

### 4.8 Every bound, with its number

| Field (in `CollapseTuning`, never a literal) | Value | Why that number |
|---|---|---|
| `feature_max_cells` | **1,024** | reference oak 162; redwood ≈ 460; caps the intactness sweep and the entity blob |
| `feature_max_structural_cells` | **64** | bounds the intactness test to 64 content reads |
| `feature_max_extent_blocks` | **62** | one chunk edge; a group touches ≤ 8 chunks, all tier-0-resident when an edit seeds it; never spans a realm (HR1) |
| `feature_group_region_draws` | **≤ 9** | a 62-block extent reaches at most one region away tangentially; ≈ 50 ns, no allocation |
| `collapse_frontier_cells_per_tick` | **256** | an 8,000-cell tower comes down over 32 ticks = 1.6 s |
| `collapse_budget_us_per_tick` | **2,000 µs** | 4 % of a 50 ms tick |
| `collapse_latency_max_ticks` | **4** | 200 ms from seed to first visible motion — reads as immediate |
| `collapse_edits_per_tick_per_realm` | **4,096** | bounds the WAL write rate; `max_edits_per_second_per_session` bounds clicks, not collapse output |
| `max_toppling_groups_per_realm` | **64** | groups beyond the cap resolve by the analytic settle in the same tick — the degradation is "in a big collapse things drop straight down instead of toppling", which is legible and bounded by the edit budget rather than by body concurrency |
| `collapse_chain_depth_max` | **4** | see §4.7 |
| `topple_ticks` (12 m tree, 1 g) | **50 ticks = 2.5 s** | `TOPPLE_K` calibrated to a real rod's fall time |
| `fall_settle_ticks` | **10 (0.5 s)** | quiescence before the settle commits |
| `falling_group_max_lifetime_s` | **30 s (600 ticks)** | Minecraft's own falling-block timeout, verified; a group that never lands itemises rather than persisting |
| `reintegrate_min_cells` | **4** | below it, itemise |
| `fall_impact_max_cells` | **64** | a 512-cell body cannot deliver 512 hits in one tick |
| `fall_impact_max_dp` | **65,535** | bounds the accumulation into a `u16` damage field; the real bounds are the cell cap and the chain depth |
| `FALL_DP_PER_KJ` (in `GameScale`, → 15 constants) | **5** | §4.6's derivation |
| `collapse_queue_max_chunks` | **1,024** (= 8 KB) | past the cap, coalesce to a single whole-realm re-evaluation flag: the queue never refuses an edit and never drops a seed |
| `falling_inflight_max_bytes` | **512 KiB** | 64 groups × 1,024 cells × 8 B — the durable in-flight table of §5.3 |
| `fall_group_stream_max_cells` | **512** | ≈ 3.0 KB worst case per group, sent **once** at detach; the reference oak is ≈ 1 KB |

### 4.9 Ships, stations, and the support model that stops being two models

§2.5.3 currently selects a `SupportModel` data row from the gravity capability — `Anchored { anchor_depth }`
against `GridRigid { root }`. That is **two behaviours**, so the identical HR4 fixture proves nothing on one
side. R6 lets them become one:

> **Support is a path to an anchor with sufficient remaining load capacity, where the load a cell must carry
> is proportional to |g|.** At |g| = 0 every path has infinite remaining capacity, so support degenerates to
> pure connectivity — which **is** Space Engineers' grid rigidity. No branch, no second row. The
> `SupportModel` enum is deleted.

The anchor predicate is one line, in which the first term is everything on a planet and empty on a station,
and the second is the reverse:

```rust
is_anchor(cell) = provenance(cell) == Terrain || realm.anchor_cells.contains(cell)
```

A connected component containing no anchor becomes a detached group — so **a free-flying ship is already one
such component, and cutting it produces two**, with the whole grid-split behaviour falling out for free.

**Two consumers of gravity, deliberately separated.** This resolves the landed-ship bug and the P9 hazard at
once:

| Consumer | Reads | Consequence |
|---|---|---|
| **Structural support** | the realm's **declared, static** gravity model (`None`/`Radial`/`Uniform`/`Spin`), realm metadata, fixed for the realm's life | a hull never re-floods; a P9 artificial-gravity generator toggling can never force a support recomputation over 50,000 cells |
| **Loose matter** — a detached group, a dropped item, an actor | the **live** gravity direction and magnitude at its pose, which for a docked or landed ship is the parent realm's field | a block cut loose in a parked ship's cargo bay **falls**; the same cut in free flight makes the piece **drift** |

Both are data values read from `gravity_dir`, which PLAN.md already places **above** the `FrameSpace` seam as
shared write-once. Neither is a kind discriminant (HR3-clean). **Spin gravity varies across a station** (it
points outward with magnitude ω²r), so the load term reads the direction **per cell**, not per realm —
written from the start, because retrofitting it means every support computation on every station was wrong.

**The P6 fixture pair** — this is P6's owed `assert_feature_anywhere`, not a P8 deferral:

- **Positive:** `cut_the_strut_and_the_component_separates`, identical on a Spherical planet profile and a
  Cartesian hull profile.
- **Negative:** `ship_hull_does_not_collapse` — a 50,000-block hull at |g| = 0 for 1,000 ticks; assert zero
  falling groups and zero WAL writes.
- **Feature half:** a Cartesian profile has no generator and therefore no features, so "identical results on
  both arms" is unsatisfiable as the profiles stand. **Resolution: give the Cartesian test profile a
  test-only generator emitting three boulders** — legitimate, since `roadmap.json` already commits that *"the
  chunk math is shard-agnostic (same code path a ship-interior realm will use)"*. This is
  **[USER DECISION R6-12]**.

One boundary so nobody tries to make it a support edge: §2.5.1's `flood_terminates_at_realm_boundary`
(infinite attenuation at a docking interface) already forbids a station's support flood entering a docked
ship. A docked ship is held by the P8 coupling constraint, never *supported by* the station.

### 4.10 Dormancy

Collapse is **catch-up class 1** in §2.5.2's taxonomy — a pure function of the block field, with no rate, no
accumulator and no history. **There is nothing to integrate over a month of sleep.** What must survive is
**pending work**.

- **The persisted work list.** A set of dirty chunk keys, ≤ 1,024 × 8 B = **8 KB**, empty in the steady state,
  written inside the WAL's own redb transaction so it is crash-consistent for free and needs no recovery path.
  Seeds coalesce to chunk granularity under pressure and, past the cap, to a single whole-realm
  re-evaluation flag — so the queue **never refuses an edit and never silently drops a seed**.
- **A realm may not spin down with a group in flight.** Spin-down force-settles every active group (≤ 64 ×
  ≤ 1,024 cells, one transaction, a few milliseconds), converting a transient quantity into grid state, which
  is the project's existing checkpoint discipline.
- **Coarse-observation onset drains the queue.** §2.5.5 rule 2 already requires a catch-up-to-now pass before
  a realm publishes, or a distant player watches a tower jump when the realm wakes. Because the topple's
  final pose is closed-form (§4.5), the drain writes the **same bytes** the live path would have written.

**One retraction.** The tempting invariant *"no edit ⇒ no collapse, so a dormant realm cannot spontaneously
collapse"* is **false**. §2.5.2 marks fire, fluids, growth, weathering, freezing/melting and explosions as
running dormant with capped or closed-form catch-up, and §2.5.5's table states that fire, fluids,
freezing/melting, growth and collapse all change solidity. Every one of those produces block changes through
the ordinary edit path in a realm nobody has visited — a lightning fire burns through a trunk, an ice shelf
melts, a flood undercuts a bank. A dormant realm's collapse queue is **not** empty.

**The salvageable half is about spin-up cost, and it is real.** §2.5.2 classes support as *"a pure function of
the block field: recompute on spin-up, free"*. That was true when the flood had nothing to traverse. With
feature cells conductive, a forested surface chunk holds `62²/100 × 162 = 6,227` feature cells (2.6 % of the
chunk) and the 505-column tier-0 disc holds **3.14 M** of them — ≈ **31 ms per disc spin-up** at 10 ns per
visit, paid on every spin-up and every coarse catch-up pass. It is free again under one rule:

> **Support is computed only for chunks whose authored-cell mask is non-empty.** A chunk with an empty mask
> has, by the generator's own validator, all its features anchored.

**And the HR1 half nobody should have to discover:** support and collapse run **only on the realm owner**. A
ghost or overlap shard must never run the flood, or two shards disagree about what is standing.

---

## 5. Conservation of matter, traced end to end for one tree

The economy depends on this, so it is traced cell by cell with the arithmetic visible.

**The reference instance.** The v1 feature catalogue's *mature oak, size class 2*: **162 cells = 12 oak trunk
cells (structural) + 150 leaf cells (non-structural)**. Oak: density 700 kg/m³, integrity 11,985 dp,
`fall_outcome = Reintegrate`. Leaf (a new substance row R6 requires — the v1 table has no leaf material):
density 150 kg/m³, work of fracture 50 J/m² → toughness 846 dp/m³ → integrity 846 dp,
`fall_outcome = Shatter`.

### 5.1 The cut

An axe at 150 W delivers `150 × TOOL_DP_PER_WATT(4)` = 600 dp/s gross, split Cut 3686 / Blunt 410 of 4096.
Oak passes the axe's 2,000 MPa cut gate at full scale, so the effective rate is `600 × 3686/4096` = **540
dp/s**. One oak block is `11,985 / 540` = **22.2 s**.

### 5.2 The ledger

| Step | Cells | Records written | Bytes |
|---|---|---|---|
| Player breaks the base trunk cell | 1 | 1 removal | 8 |
| Group breaks (a structural cell is gone) → 161 cells above the cut are unanchored | — | 0 | 0 |
| Detach: the 161 cells leave the grid | 161 | 161 removals | 1,288 |
| Topple, 50 ticks | — | 0 | 0 |
| Settle: 11 trunk cells reintegrate as horizontal logs, keeping `Feature` | 11 | 11 placements | 88 |
| Settle: 150 leaf cells shatter → drops, coalesced into `DroppedBlock` stacks | 150 | 0 | 0 |
| **WAL total** | **162 in, 11 back** | **173 records** | **1,384 B** |
| Edit pyramid at §2.4.3's **2-D / surface** row (0.333 entries per edit × 8 B) | | 58 entries | **461 B** |
| **Permanent delta per felled tree** | | | **1,845 B ≈ 1.80 KB** |

**Conservation, checked at every arrow.** 1 (cut) + 161 (detached) = 162 cells left the grid. 11 returned as
blocks; 150 became item stacks; 1 became the player's drop. `11 + 150 + 1 = 162`. **Nothing was created and
nothing vanished**, and the settle map's injectivity (§4.5) is what makes that a unit assertion rather than a
hope.

**The one hole this exposes in already-written §2.3.7.** As written, `Broken` drops the full table *"if
`tool_tier ≥ harvest_tier`; otherwise nothing"*, and `Shattered` is *"subject to the same tier gate"*. A
landing, an explosion, a fire and a collapse carry **no tool**, so under the literal rule every block they
break yields **nothing** — the 150 leaf cells above evaporate, and so does the wood in every tree a forest
fire fells and the granite in every wall a shell blasts. In a purely player-driven economy where relative
labour cost *is* the price, that is a very large unpriced matter sink. **Amendment:**

> `harvest_tier` gates **only sources that carry a tool**. Physics sources (landing, explosion, fire,
> collapse, fluid) drop at the material's declared `shatter_drop_q12` yield with **no tier gate**, and every
> unit not dropped is posted as a typed `MatterDestroyed { reason, substance, volume }` fact on the game's
> own journal.

That makes the dormant-world design's identity `Σ mint − Σ burn − Σ DECLARED_loss == Σ positions` repairable
for physics losses, which today it is not. Raised as **[USER DECISION R6-7]** because it amends a written
section.

### 5.3 Crash safety — both ends, because only one end was ever specified

**Detach** mirrors §2.4.2's existing break-then-delete ordering: the removal edits enter `block_wal`
**first**, the group is created **second**. A `kill -9` between the two loses a tree and never duplicates one.
Loss is the correct failure direction.

**Landing is the actual duplication window, and nobody specified it.** If the write-back records commit and
the group's destruction does not, a retry writes the cells twice — matter created on an ordinary retry.
**Rule: destroy-group and write-cells commit in ONE redb transaction carrying a `settle_id`, which makes the
retry a no-op.**

**A durable in-flight table, because a routine re-shard would otherwise destroy matter.** A falling group is a
new `EntityKind` (§7) in the Transient band, and a Transient kind's loss on a realm re-shard is *within
declared budget* — so a re-shard during a bombing run would legally destroy thousands of cells that have
already been removed from the grid. The cure is cheap: a small durable table `falling_inflight`, written in
the same transaction as the detach, holding each group's cells at 8 B each. **64 groups × 1,024 cells × 8 B =
512 KiB worst case.** Spin-up, re-shard and `kill -9` all resolve to an analytic settle from that table
rather than to a matter sink.

**One rule about the debris cap.** `max_debris_entities` bounds **entities**, never **matter**. Coalescing
merges by stack count; hitting the cap must never refuse a drop, or the cap itself becomes the largest sink
in the game.

### 5.4 Deforestation at scale — the number that makes R7/R8 a storage requirement

| Quantity | Arithmetic | Result |
|---|---|---|
| Trees in the tier-0 disc | `π × 786² m² ÷ 100 m²` | **19,406** |
| Permanent delta for a clear-cut disc | `19,406 × 1,845 B` | **35.8 MB** |
| Discs to exhaust §3.9.3's ~1.5 GB heavily-played-planet budget | `1.5 GB ÷ 35.8 MB` | **41.9** |
| Area that represents | `41.9 × 1.941 km²` | **81.3 km²** — a square 9.0 km on a side |
| As a fraction of the starter body's surface | `81.3 ÷ 328,444 km²` (at R = 161,671 m) | **0.025 %** |

**A quarter of a thousandth of one planet's surface, clear-cut, fills the planet's entire delta budget.**
§3.9.6's response to exhaustion is a typed refusal of all further edits plus an operational alarm, so the
first organised logging operation would brick editing on the planet.

**This is what makes R7/R8 a storage requirement rather than a flavour feature**, and it is discharged by
§3.4's prune rule: when natural recovery restores a cell to what the generator always said was there, the
record is deleted at every rung and the bytes come back. The provenance clause is what keeps that safe.

**Per-substance landing outcome is worth 46 %.** Reintegrating the whole canopy instead of shattering it
would be `162 + 161 = 323` records = 2,584 B of WAL plus `323 × 0.333 × 8` = 862 B of pyramid = **3,446 B per
tree**, against 1,845 B — a **1.87×** saving, or 46 % of the total. (The claim circulating in review that
trunk-only reintegration is *13× smaller* counts only write-backs and ignores that the 162 removals are
unavoidable either way.)

---

## 6. The composite seam (owner ruling R4)

R4's multi-block composite subsystem is being designed concurrently. This design does not design it. What
follows is the **interface only**.

### 6.1 The load-bearing property that keeps the two designs independent

> **THE COLLAPSE SYSTEM NEEDS NOTHING FROM COMPOSITES.** Provenance is per-cell, total, and complete on its
> own; feature groups are seed-derived. Every rule above must be correct with `composite_of(cell)` returning
> `None` for every cell in the world.

### 6.2 What collapse may take from composites (all optional, all read-only)

```rust
fn composite_of(cell: BlockAddr) -> Option<CompositeId>;         // total, cheap, deterministic
fn cells_of(id: CompositeId) -> impl Iterator<Item = CellIndex>; // may be empty
fn falls_as_a_unit(id: CompositeId) -> bool;                     // + a designated root cell set
fn on_cells_removed(id: CompositeId, removed: &[CellIndex]);     // notification; return value never branched on
```

Used by the unsupported-set merger as a **grouping hint** beside its existing contiguity merge, never as a
precondition. Composites contract to one node in the support graph exactly as feature groups do — which is
the convergence the brief noticed, and the whole of it.

### 6.3 What collapse is BANNED from assuming

- That a composite exists. A lone feature log must still fall.
- That a composite's cells are 6-connected, convex, or inside one chunk.
- That membership is stable within a tick — recognition may be deferred and budgeted.
- That a composite's cells share one provenance value.
- **That the collider comes from the composite.** A falling group's collider — if the reserved rigid-body
  tier is ever enabled — comes from the **tier-0 voxel occupancy of the detached cells**, on the ship-grid
  path. R4 says an art-pack tree is bigger than its blocks and *"all of it must collide"*; taking a collider
  from that would put **meshes on the server**, which §2.5.2 already refuses on identical grounds when it
  rules sound occlusion server-side without meshes. A composite's oversized collider is a client and
  gameplay concern that collapse never consults.

### 6.4 Two one-sentence requirements on R4

1. **A cell belongs to at most one group** (feature or composite). Without it, "the group is one node" is
   ill-defined.
2. **Recognition must consume orientation.** If a composite is recognised from a block *pattern*, a felled
   tree lying on the ground is a fresh pattern of tree blocks, and a naive recogniser will promote it back
   into a standing tree. A horizontal log run is not a vertical trunk.

Neither constrains R4's implementation.

### 6.5 The invariant both sides assert

> **Composite membership NEVER writes provenance, and provenance NEVER gates composite recognition.** A
> durable field with two writers is banned.

### 6.6 Both branches — and whether this design differs

**The encoding is identical in both branches.** That is why this file can land before R4 does.

| | **Branch A — composites are recognised block groups over the same grid** | **Branch B — composites are entities that occupy cells** |
|---|---|---|
| `composite_of` returns | a `CompositeId` | an `EntityId` |
| The collapse **unit** | unchanged: a contracted node in the support graph | the **entity**; the merger hands the group to the entity system rather than assembling one from cells |
| Group intactness | composites answer it themselves (a player-placed composite's cells are all delta cells, so this design's authored-mask fast path does not apply to them) | the entity owns its own integrity |
| Everything else in this file | unchanged | unchanged |

**The one thing that materially differs is the granularity of the falling unit**, and it is one function —
`component_of()` becomes `contracted_component_of()` — at one call site, with zero record change and zero
wire change. **Build collapse for Branch B** (pure connectivity over non-terrain cells plus feature groups)
and ask the composite run to hold nothing for it beyond the four accessors above.

**One agreement the two runs must make explicitly:** generated features keep their **seed-derived** identity
even if player composites carry a **persisted** one. Everything cheap about this design — zero persisted group
bytes, one-comparison intactness, a resting forest costing nothing — flows from that.

---

## 7. The consequences

### 7.1 Gameplay

- **Mining is permanently safe.** No tunnel collapse, no unstable rock, no cave-in, ever, at any span. That
  is exactly what R6 asks for, and it is a simplification and a real cost saving — but it also gives up a
  mechanic that Vintage Story and Deep Rock Galactic both build content on. Confirmed as
  **[USER DECISION R6-6]**, with the seam reserved for free.
- **A tree standing forever on a single remaining terrain cell** after a player mines out a hillside is
  direct and unavoidable. It will look silly in a screenshot.
- **One anchored contact is enough** for an intact feature group. That is Teardown's shipped compromise and
  it carries Teardown's shipped complaint (*"massive buildings held up by a single strand of voxels"*). For a
  bounded ≤ 1,024-cell authored feature it is defensible; for player builds the ordinary cantilever test is
  what carries the load and is strictly stricter. The two rules meet at a treehouse, and the result — correct
  under the fixpoint — may read as arbitrary to a player who cannot see which rule applies.
- **Collapse damage bypasses edit admission**, by construction: felling onto a protected build or undercutting
  a boulder above a base arrives as a *landing*, not as a player edit, so `max_reach_m`, the per-session edit
  rate cap and every future claim check are skipped. **Rule: a landing IS an edit and is subject to the same
  admission check; a group whose landing footprint intersects a region the actor may not edit is REFUSED and
  shatters into drops rather than overwriting protected blocks.** (**[USER DECISION R6-9]**.)

### 7.2 Economy

**Felling saves less labour than it looks, and the arithmetic matters** because it is the difference between
a physics mechanic and an economy event.

| Path | Work | Time |
|---|---|---|
| Harvest the standing tree block by block | 12 trunk × 22.2 s + 150 leaves × 1.57 s | **500 s** |
| Fell it, then break the 11 landed logs | 22.2 s + 11 × 22.2 s | **266 s** |
| **Saving** | | **1.9×** |

The saving comes entirely from the canopy **shattering on landing and yielding its drops automatically**;
the trunk still has to be broken log by log. So the claim that R6 *"cuts wood's labour by up to fifty times
and removes wood from the traded-goods set within a week"* is wrong — it was computed with granite's mining
time rather than oak's. And the opposite claim — that felling saves *nothing* — is also wrong, because it
counts only the trunk.

**R6 is a believability mechanic, not a harvesting shortcut, and the economy consequence is small.** If wood
*should* get cheaper, the lever is a landed-group bulk harvest or a processing step (logs → planks at a
workbench, which is Eco's and Vintage Story's shipped shape and creates a tradeable intermediate and a second
profession). That is **[USER DECISION R6-8]**, and it is the decision that turns R6 from an ornament into an
economy feature — or deliberately does not.

**The unpriced-sink hole is the real economy finding** and it is §5.2's amendment to the drop gate.

### 7.3 The detail ladder

**R6 spends zero of the pyramid record's fourteen reserved bits and adds zero wire surface.** §2.4.3 already
excludes the natural-versus-built marker from `CellSummary` on the grounds that support is tier-0 and
edit-local; two bits changes nothing. Detach and landing are ordinary tier-0 edits that reach every coarse
rung through the ordinary walk-up (5.85 µs at `tier_depth` 13, ≈ 0.5 µs with the early exit), so **[USER
DECISION 2-J]**'s fluid-rung option stays intact.

**Provenance never crosses to the client.** The obvious wrong instinct is *"the client needs to know which
blocks can fall"*. It does not: provenance changes no geometry, no material, no shape, no overlay. It is a
server-side policy input to a server-side flood, and the server does all the math. Shipping it would be two
bits × every block on every chunk delta for zero rendering value.

**Feature collapse must not ship before the edit pyramid.** A deforested hillside would be invisible at 4 km
and would pop back into existence as an observer approached, because the coarse rung renders from the
generator alone and the generator still says *forest*. That is a seamless-law violation, not a cosmetic one,
and it is the hardest ordering constraint R6 creates.

**One AoI constraint, free now and invisible until D-9's work lands.** A player at 4 km sees a fell at 2 km
through three lanes: the removal (a tier-0 edit, walked up to his rung), the fall (an entity — a 20 m body at
4 km subtends 5.0 mrad ≈ 7.9 px at the reference screen, comfortably above the locked angular-size rule), and
the landing (a second edit). Between the first and the third the tree is absent from the pyramid for the ≈ 2.5
seconds of the topple, and that reads correctly **only** while the entity lane delivers the group at that
range. **Rule: a falling group's AoI threshold derives from the same `px_per_cell_target` as everything else,
never a separate literal, and is never coarser than the rung at which its blocks were visible.**

### 7.4 Ships

Covered in §3.6 and §4.9. In one line: the format is identical, the domain narrows to two values, the support
model becomes one algorithm whose gravity term is zero, and the negative fixture lands at P6.

---

## 8. What is reserved now, and what it costs to retrofit

| # | Reserve | Where | Free now / cost later |
|---|---|---|---|
| 1 | The **two-bit provenance field** at bits 23..22 and the reserve going 15 → 14 | the durable record | Zero bytes today. Later: an offline read-modify-write of every record in every saved world (§3.9.3 sizes a heavily-played planet at ~1.5 GB of deltas, ~1.7 GB with the pyramid), plus an epoch bump, plus a digest change, plus a fleet-wide client refusal. |
| 2 | **Value 3 reserved and rejected on decode** | the same field | The escape slot that makes a fourth class (`NpcBuilt`, `Ruin`, `Grown`) an epoch-gated additive change. Without it, a fourth class is a record migration. |
| 3 | The **numbering** `0 Terrain, 1 Feature, 2 Placed` | the same field | Buys `new >= old` as the monotonicity check in one typed constructor. Renumbering later reinterprets every saved cell. |
| 4 | The **typed write constructor** in the block store's write API, taking provenance and enforcing monotonicity + the generator-only rule for terrain | `crates/core/src/block/` | One newtype and one constructor before the write sites exist, versus auditing six of them afterwards. This is the `Tier0Key` move applied to a second invariant. |
| 5 | **`fall_outcome: FallOutcome`** column on the substance row | `MaterialDef` | One `u8` in a 118-byte row. Reusing `shatter_frac_q12` instead would forbid a shattering canopy outright (§2.3.3 gives Organic 4096 = never shatters) and would couple a landing policy to a damage policy permanently. |
| 6 | **`FEATURE_STRUCTURAL`** flag bit in `BlockTypeDef.flags` | the generated tables | A bit in an existing `u16`. Without it, picking one leaf dissolves the whole canopy. |
| 7 | **`TERRAIN_UNSTABLE`** flag bit on the substance row, **unset on every v1 row** with a coherence assertion, plus a per-realm cave-ins toggle | `MaterialDef`, realm metadata | A spare bit in an existing `u32`. Retrofitting a per-substance terrain-instability policy after worlds are saved is a table migration **plus** a re-balance of every rock row. Reserving is the only way to keep R6-6 reversible. |
| 8 | **`FallMotion::{ScriptedTopple, RigidBody}`** on the substance row, `RigidBody` unused in v1 | `MaterialDef` | Keeps the rigid-body quality tier available without committing to it, and without a second world representation. |
| 9 | **`terrain_admissible`** derived field | `crates/sim/src/capability.rs` (`ShardProfile::build`'s validated lattice) | A derived boolean in an existing lattice, free today. Without it, *"no realm without a generator may hold terrain"* is a comment. |
| 10 | **`gravity: GravityModel`** as **static realm metadata**, and `anchor_cells` / `root_cell` on the realm metadata row | realm store, `ShardProfile` | Free today. Without the static rule, a P9 artificial-gravity signal edge forces a 50,000-cell support re-flood; without `anchor_cells`, a ship has no rigidity root and no successor rule when the root is destroyed. |
| 11 | **The `falling_inflight` durable table** (≤ 512 KiB) and the **`collapse_dirty_chunks` work list** (≤ 8 KB), both written in the WAL's own transaction | the realm's redb file | Both are new durable state, but neither is a one-way door — an empty table is always legal and both can be widened freely. They need the same treatment §2.4.3 gave the pyramid: a recompute path, a `vdctl` command, and a harness assertion that a scenario ends with an empty list and no unanchored cell. |
| 12 | **`EntityKind::FallingGroup = 14`**, Transient, `NeverTransferOnly`, `RealmAnchored`, `LossBudget(1)`, `SchemaId(14)`, `max_state_bytes: 8192` | `crates/core/src/entity_kind.rs` | Verified: the Transient band is 10..20 and tags 10–13 (`Debris`, `DroppedBlock`, `Projectile`, `Rocket`) are taken, leaving **six** slots — and R4's composites may want one. The band is a persisted registry with dense append-only tags. `DROPPED_BLOCK_DEF`'s `max_state_bytes: 64` holds eight cells and cannot be reused. 8,192 B is exactly `feature_max_cells` × 8 B. |
| 13 | **The prune-on-equality rule for `ChunkRecord`, comparing provenance** | §3.9.3 | Free before the compactor is written. It is the only mechanism that returns deforestation's bytes, and the provenance clause is the only thing that stops it becoming a laundering exploit. |
| 14 | **`MatterDestroyed { reason, substance, volume }`** as a typed fact on the game journal | the game journal | An enum, free now, and the thing that keeps the economy's conservation identity repairable for physics losses. |
| 15 | **`FALL_DP_PER_KJ`** in `GameScale` (14 → 15 constants) and **`CollapseTuning`** as a sibling of `BlockStoreTuning` / `TransferTuning` | the one tuning config struct family | No magic numbers. Every bound in §4.8 is a named field. |
| 16 | Folding the **feature catalogue and its provenance assignment into `BlockRegistryHash`** | `crates/core/src/block/` | Free before the hash is pinned. Without it a client and server generate identical terrain and disagree about what falls — a divergence the byte-identity gate cannot catch. |

**Zero new `InterShardFlow` arms.** Detach and landing are ordinary edits on the already-reserved `BlockEdit`
arm (`SideEffecting { FencedKey }`, `ProducerLessReliable`). The group is a realm-anchored transient on the
existing snapshot lane; its cell list rides the existing `BulkKind` on the D-4 lane, once, at detach. The
reserved names `BlockEdit` (P6) and `Signal` (P9) are untouched.

---

## 9. The gates this creates

Every one with the number it must hold.

| # | Gate | Must hold |
|---|---|---|
| G1 | `provenance_round_trip` | 3 `Ok` arms via `ALL` + 1 `Err` arm via `expect_err`; **100 % region+branch** on the decode |
| G2 | `air_canonical_form` | an `Air` record with non-zero provenance or state is **rejected**: 1 accept + 1 reject branch, both covered |
| G3 | `terrain_write_rejected` | a write of `Terrain` through any edit path, and any write on a realm with `terrain_admissible == false`, is **refused** with a typed error: 1 accept + 1 reject, both covered |
| G4 | `provenance_monotone` | over 10,000 random write sequences through the typed constructor, provenance **never decreases**; 0 violations |
| G5 | `assert_features_anchored` | ≥ **1,000 regions per biome**, pure generator output, empty delta → **0** unanchored instances |
| G6 | `feature_size_caps` | content build asserts every size class ≤ **1,024** cells, ≤ **64** structural cells, ≤ **62** blocks extent, and **0** instances crossing a cube-face seam |
| G7 | `feature_density_subcritical` | content build asserts `density × fall_footprint < 1` per biome; the reference forest gives **0.12** |
| G8 | `collapse_conserves_matter` | over 10,000 random groups: `writes + drops == group cells`, **exactly**, 0 discrepancies |
| G9 | `topple_pose_is_closed_form` | for 10,000 groups, the live topple's settled pose is **byte-equal** to the analytic settle's pose for the same group and fence |
| G10 | `ship_hull_does_not_collapse` | 50,000-block hull, \|g\| = 0, 1,000 ticks → **0** falling groups, **0** WAL writes |
| G11 | `assert_feature_anywhere(collapse)` | one identical fixture passes on a **Spherical planet** profile and a **Cartesian** profile with the three-boulder test generator |
| G12 | `collapse_survives_kill9` | SIGKILL at ≥ 20 points across detach/topple/settle → **0** duplicated cells; every lost group recovered from `falling_inflight` |
| G13 | `support_flood_cost` (benchmark) | mining one cell of plain rock visits **≤ 8** cells; a 20×20×20 tower fully collapses within **32 ticks** |
| G14 | `spinup_support_is_free` (benchmark) | a 505-column tier-0 disc with an empty authored mask spins up visiting **0** support cells |
| G15 | `deforestation_delta_budget` (load) | fell 1,000 reference oaks → **1,845 B ± 5 %** of WAL + pyramid per tree |
| G16 | `palette_width_under_provenance` (load) | generated forest chunks stay at **≤ 16** distinct palette entries (4-bit indices) at the 95th percentile — the step to 5 bits costs **116.4 → 145.5 KiB per chunk, +25 %** |
| G17 | `collapse_mass_event` (load) | 3,704 groups (§9 note) drain within **200 ticks** with **0** unanchored cells remaining and **0** groups dropped |

> **Note on G17's 3,704.** §2.5.4's reference event is an HE shell breaking 500 blocks; a hundred charges is
> 50,000 broken cells, and structural trunk cells are 12/162 = 7.4 % of a tree, so ≈ 3,704 groups break at
> once. This is the stressor, not fire: §2.3.6 gives fire 132 dp/s against oak's 11,985 dp = 90.8 s per block,
> and §2.5.2 prices a 500-block active fire set as the reference load, so trunk burn-throughs arrive at
> ≈ 0.46/s — 10,000 trees is six to eighty **hours** of burning. **Fire is not a collapse stressor and no
> budget should be sized for it.** Explosives are, which is why groups beyond `max_toppling_groups_per_realm`
> resolve analytically in the same tick rather than queueing: at 4 groups per tick, 3,704 groups would leave
> tree-tops hanging in the air for one to two minutes.

---

## 10. The decision register for this ruling

Each row states a **default** that ships if the owner says nothing, so nothing is blocked.

| # | Decision | Options | Recommendation / default | One-way door | Cost of deferring |
|---|---|---|---|---|---|
| **R6-1** | **The numbering.** | (a) `0 Terrain, 1 Feature, 2 Placed`, value 3 rejected — monotone, so "provenance never gains privilege" is one `>=` in one constructor. (b) `0 Terrain, 1 Placed, 2 Feature` — argued from backward compatibility with saved worlds. | **(a).** There are no saved worlds; verified, the workspace has no block/grid/voxel/chunk module at all. | **YES**, the instant the first planet is saved | Renumbering later reinterprets every saved cell |
| **R6-2** | **Two bits or three.** | (a) Two — three values plus one rejected escape; a fourth class later is an epoch-gated additive change. (b) Three — eight classes, reserve drops 15 → 12. | **(a).** Only a *fifth* class would cost a format change. | **YES** | None; the escape slot keeps it open |
| **R6-3** | **Is `Terrain` ever writable?** A player filling a hole in a mountain writes `Placed`, so his fill is collapsible while the visually identical rock beside it is not. | (a) Representable but **not writable** at P6 — the write path rejects it as a covered branch a future terraforming tool can relax. (b) Writable through a designated terraform path. (c) Writable when fill becomes fully enclosed. | **(a).** (c) is a laundering vector: bury your base in dirt and it becomes indestructible. | No — (a) → (b) is a validation change, not a format change | None |
| **R6-4** | **Scripted topple or rigid body.** | (a) Server-authored scripted topple + closed-form final pose + integer settle: fully deterministic, replayable in a fixture, **no collider**, no transfer envelope, no re-shard matter sink, and the live and dormant paths write identical bytes. Cost: a fall is a scripted rotation, so a boulder cannot roll off a cliff. (b) A rapier rigid body: prettier in the rare case; drags in up to 32,768 voxel colliders at the 64-group cap, a non-reproducible landing, and a second world representation. | **(a)** for v1, with (b) reserved as a data-selected quality tier for player-built collapses. | No, but (a) **removes** an argument from **[USER DECISION 2-G]** and (b) **adds** one | Take it before 2-G's spike is scoped, or 2-G is answered on incomplete evidence |
| **R6-5** | **The axial/lateral support amendment.** Today `cantilever_blocks` is derived as a horizontal beam span and then spent as an isotropic flood radius capped at 32, which caps every tree, tower, mast and antenna at 32 blocks tall. | (a) Amend: zero attenuation along −gravity toward the anchor, `1/cantilever_blocks` laterally. (b) Leave it and accept a hard 32 m vertical limit on everything. | **(a).** Physically correct, one sign test, and it is what 7 Days to Die ships (*infinite vertical stability, an eight-block horizontal beam limit*). | No | It changes how building **feels** — unlimited vertical, limited horizontal — so it is a design call, not a silent fix |
| **R6-6** | **Cave-ins are now impossible, by implication rather than by decision.** Mining is perfectly safe forever. | (a) Confirm, and reserve the seam: one `TERRAIN_UNSTABLE` substance flag (unset on every v1 row, coherence-asserted) plus a per-realm toggle. (b) Confirm without reserving. (c) Design terrain instability now. | **(a).** Reserving is a spare bit in an existing `u32`; retrofitting a per-substance terrain-instability policy after worlds are saved is a table migration plus a re-balance of every rock row. | No if reserved; effectively yes if not | Vintage Story and Deep Rock Galactic both build content on cave-ins; giving it up by implication is worth confirming |
| **R6-7** | **The physics-source drop gate** (amends written §2.3.7). As written, every tool-less break yields **nothing**, so landings, explosions, fire and collapse destroy all matter they touch. | (a) Keep it. (b) Gate only tool-carrying sources; physics sources drop at `shatter_drop_q12` with no tier gate, and every unit not dropped is posted as a typed destroyed-matter fact. | **(b).** Note that §2.3.3 already forbids Organic materials from shattering at all, so wood always breaks whole — this decision mostly governs rock, ice and glass. | No | A very large unpriced sink in a purely player-driven economy, and the conservation identity is unverifiable exactly where the physics is |
| **R6-8** | **What does a player DO with a landed group?** With the shipped numbers felling saves **1.9×**, not 50×. | (a) Nothing special — felling is a believability mechanic. (b) A landed group is bulk-harvestable in one action → a genuine 10–20× labour cut on wood. (c) A second production step (logs → planks at a workbench), Eco's and Vintage Story's shipped shape, creating a tradeable intermediate and a second profession. | **(a) for v1, (c) as the eventual answer.** (b) is the one that actually moves the economy and should be taken deliberately. | No | Decides whether R6 is a feature or an ornament; must be settled before the landing outcome is coded |
| **R6-9** | **Does falling matter damage player builds, and does a landing pass edit admission?** | (a) Landings damage, and a landing **is** an edit subject to `max_reach_m`, the rate cap and every future claim check; a refused landing shatters into drops. (b) Landings damage with no admission check (the grief vector). (c) Landings never damage structures — Valheim's shipped choice for rolling logs. | **(a).** It interacts with build protection, which is not designed, so it should be taken consciously rather than inherited. | No | Without it, felling onto a protected build bypasses every permission check by construction |
| **R6-10** | **Are grown and planted things features?** (R7/R8 make planting a profession and store a growth stage.) | (a) `Feature`, regardless of who planted the seed — for collapse purposes a grown tree and a natural tree are the same object; who planted it is R8's growth record's business. (b) `Placed`. | **(a).** It also keeps the re-fell exploit closed: a planted tree that has grown is a real feature with a real structural footprint, and once cut the delta is permanent. | No | Affects whether a player-grown forest falls the way a natural one does |
| **R6-11** | **`FALL_DP_PER_KJ = 5`** — the fifteenth `GameScale` constant, derived from *"a cubic metre of granite dropped ten metres breaks the floor cell it lands on and not the one beside it"*. | Accept, or re-derive from a different sentence. | **Accept.** It is a balance number in a system whose discipline is that tuned numbers are few, global, named and in one place; it should be signed off in the same session as the other fourteen. | No | None |
| **R6-12** | **The HR4 feature fixture.** A Cartesian profile has no generator and therefore no features, so the feature half of the identical fixture is unsatisfiable as the profiles stand. | (a) Give the Cartesian test profile a **test-only generator emitting three boulders**, so the fixture is genuinely identical. (b) Declare features a `ShardProfile` capability and weaken the gate to *"identical on ≥ 2 profiles that declare features"*. | **(a).** Legitimate, since `roadmap.json` already commits that the chunk math is shard-agnostic. (b) leaves the project's strongest hard rule partially unenforced on a new subsystem. | No | The gate lands weaker and is hard to strengthen later |
| **R6-13** | **What provenance does gravity-moved matter carry?** | (a) It keeps what it fell with — a felled tree's logs stay `Feature`. (b) Everything that moves becomes `Placed`. (c) Terrain that moves becomes `Feature`; feature and placed keep theirs. | **(a) = (c)**, since terrain never moves under this design. It is monotone under R6-1, it does not launder natural wood into player wood at the moment it drops, and it does not launder shoved crust into terrain. | No | Affects drops, claims, salvage, and whether a composite recogniser still sees a natural rock |
| **R6-14** | **Consequential and cheap: should `PaletteEntry` become a 24-bit-valued `u32` newtype** (exactly bits 45..22 of the record) instead of a three-field struct? | (a) Yes. (b) No. | **(a).** One shift and one mask for the record→palette projection, a single integer compare for palette equality, and one padding byte removed from a structure that **is** persisted inside the masked-dense chunk encoding. This is §2.4.1's and §3.9.3's change to take, not this file's to impose. | No | A struct-shape change after the compactor is written |

---

## 11. Adjudicated objections

**Where the adversarial review was right, and decisive:**

- *A per-cell "anchored equals terrain, then flood" model cannot represent an arch or an ice shelf* — correct
  and decisive; §2.2 gives ice a cantilever of 6 and granite 12, so two of the four features the owner named
  by name would collapse on first touch. The feature group is semantically necessary, not an optimisation.
- *A per-kind attachment policy drops a tree growing out of a cliff face* — correct; attachment is per
  instance, recorded by the feature stage, at zero persisted cost.
- *An intactness test on delta **existence** kills a tree when a player **repairs** it* — correct; the test
  reads cell contents, and the empty-mask fast path survives unchanged.
- *"No edit implies no collapse" is false for dormancy* — correct; fire, fluids, melting and growth all
  produce edits in unvisited realms. The salvageable half is about spin-up cost and is adopted verbatim.
- *Using a horizontal beam span as an isotropic flood radius caps every tower at 32 blocks* — correct, and
  independently confirmed by 7 Days to Die's shipped model of infinite vertical and finite horizontal support.
- *A radius cap on a **decrease** flood leaves floating structures rather than slow ones* — correct; the bound
  is the carry-over cell budget, and `MAX_SUPPORT_RADIUS` keeps only its cantilever-cap role.
- *The **landing** transaction, not the detach, is the duplication window* — correct, and nobody had specified
  it; one transaction with a settle identifier closes it.
- *A falling group as a Transient entity is legally droppable on a re-shard* — correct, and the reason for the
  durable in-flight table.
- *The domino branching factor is 0.12 and therefore subcritical by eight times* — correct, and it replaces
  both designs' chain-depth reasoning with a content assertion.
- *The prune-on-equality rule regrowth needs does not exist, and a naive version is a laundering exploit* —
  correct, and it is the strongest single argument in the whole review for putting provenance in the
  per-instance record.
- *The fifteen reserved bits are reproducible by exactly one layout* — correct, and independently confirmed:
  §8.6 item 1's adjudicated synthesis states the field order verbatim.

**Where the adversarial review was wrong:**

- **The bit positions are not a cross-section land grab.** §8.6 item 1 — which R1 adopted — states the field
  order explicitly, so the positions follow from a ruling already taken rather than from this design.
- **The 0.0104 % delta-budget figure is wrong by 2.4×.** It reintegrates ~100 canopy cells that the
  per-substance landing outcome shatters, and it applies §2.4.3's 1.1 mixed-edit pyramid factor to what is
  plainly a 2-D surface edit set, which §2.4.3 prices at 0.333. The corrected figure is **0.025 %** — still
  decisive, and still what makes R7/R8 a storage requirement.
- **"Felling saves NO labour" counts only the trunk.** The canopy shatters on landing and yields its drops
  automatically, which is 235 s of hand-tool work removed. The saving is **1.9×**, not 0× — and not the 50×
  claimed on the other side, which used granite's mining time instead of oak's.
- **"Organic materials cannot shatter, so a shattering canopy is illegal"** is true of §2.3.3's *damage*
  outcome and irrelevant as an objection: the landing outcome is its own authored column, not a
  reinterpretation of the shatter column. Repurposing it would have been the actual defect.
- **The observed-versus-dormant world divergence is dissolved, not accepted.** With a scripted topple the
  final pose is closed-form in `(group, fence)`, so the live path and the dormant path write identical bytes
  and a fixture can assert exactly where a tree lands. Every design that reached for a rigid body had to
  accept the divergence; none noticed it was optional.
- **Retiring `MAX_SUPPORT_RADIUS` outright overreaches.** It must keep its §2.2 job as the lateral cantilever
  cap; only its §2.5.2 job as a flood bound is retired.
- **The in-flight table is 512 KiB, not 45 KB** — 64 groups × 1,024 cells × 8 B. The cell contents are not
  recoverable from the WAL, because dedup-to-final stores the new state (air), never the old.
- **Fire is not the collapse stressor.** §2.3.6's own numbers give oak 90.8 s per block under fire, so trunk
  burn-throughs arrive at ≈ 0.46/s and ten thousand trees is six to eighty hours of burning. Sizing any budget
  for a forest fire would be sizing for the wrong event; explosives are the stressor.

**Where the three candidate designs were wrong:**

- *"R6 makes the support rule strictly simpler — one integer compare"* — wrong; that rule deletes the arch and
  the ice shelf the owner named.
- *"A composite may supply the falling group's collider"* — wrong; it would put art meshes on the server,
  which §2.5.2 already refuses on identical grounds for sound occlusion.
- *"Reintegrating trunk-only is 13× smaller"* — wrong by counting only write-backs; the 162 removals are
  unavoidable either way, and the real saving is 1.87×.
- *"Regrowth returns ~60 MB per disc through pruning"* — right in direction, wrong in mechanism: §3.9.3 has no
  prune rule at all today, so as written regrowth recovers only the pyramid half. The missing rule is added
  here, with the provenance clause that keeps it safe.

---

**Sources for the prior art cited above:**
[Vintage Story cave-in and hardened rock](https://mods.vintagestory.at/caveinfix) ·
[Minecraft falling-block entity](https://minecraft.wiki/w/Falling_Block) ·
[Valheim falling-tree domino](https://www.pcgamer.com/in-valheim-the-trees-punch-back/) ·
[7 Days to Die structural integrity](https://7daystodie.wiki.gg/wiki/Structural_Integrity)
