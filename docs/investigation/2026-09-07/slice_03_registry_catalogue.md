# Slice 3 — The registry and the shape catalogue

What is built, why, how it is tested, and what the owner decides. Sources: `04_block_record_registry.md`
§1, §3, §4 and `02_smooth_terrain.md` §4 (both refuted twice and revised), ruling V6 Part B, and the
registry idiom the code already uses (`crates/core/src/entity_kind.rs`). Simplified Technical English,
examples from the game.

---

## 1. What this slice decides

The registry is the catalogue of everything a cell can be. A saved record names a kind by a number;
the registry says what that number means: what it is made of, what shape it has, what it does,
how much it weighs and how it breaks. If the meaning of a number ever changed,
every saved plate in the world would change with it. So the registry is append-only, and the part of
it that gives a saved record its meaning is fenced by a digest.

Nothing is placed, meshed or collided in this slice. It lands the tables, the shape catalogue, the
rotation table, the digest, and the gates that keep them honest. The first placement is slice 10.

---

## 2. The five tables

```text
   substance          form (shape)         function
   +-----------+      +--------------+     +--------------+
   | basalt    |      | cube         |     | none         |
   | dirt      |      | slab         |     | (thruster)   |   <- later, P9
   | steel     |      | wedge        |     | (seat)       |
   | titanium  |      | corner-in    |     | ...          |
   | ...       |      | ...          |     +--------------+
   +-----------+      +--------------+
          \                 |                  /
           \                |                 /
            +---------------+----------------+
                            |
                     block kind (one row = one number a record names)
   +----------------------------------------------------------------------+
   | id | substance | form | function | flags | variants | sub_scales |
   +----------------------------------------------------------------------+
   |  0 | EMPTY     | -    | -        | ...   | 1        | -          |   <- the removal kind
   |  1 | basalt    | cube | none     | ...   | 1        | 0000       |
   |  2 | steel     | cube | none     | ...   | 3        | 1111       |
   | ...                                                                    |
   +----------------------------------------------------------------------+

   attachment kind (a second typed table under the same mechanism)
   +--------------------------------------------------------------------+
   | id | name  | makes_body | legal_faces | slots | params_max |
   +--------------------------------------------------------------------+
   |  0 | hud   | no         | all         | 4     | 64         |
   |  1 | hinge | yes        | all         | 1     | 16         |
   +--------------------------------------------------------------------+
```

Every table follows the idiom the entity kinds already use: dense explicit numbers, an `ALL` list, a
`from_number` that errors on an unknown number, and a `def` that is an exhaustive match. A number is
never reused. A retired kind keeps its row, marked retired and pointing at its replacement, so a saved
plate of a retired kind still decodes and still draws, and new placements of it are refused.

Sixteen bits per id: 65,536 rows per table. A thousand building substances times twenty forms is 20,000 kinds; terrain substances take one kind each.

**Example.** A titanium hull plate is kind 2: substance titanium, form cube, function none, three
styles, every small-block scale allowed. A saved record says "kind 2, style 1". The
registry says the rest, and never differently.

---

## 3. What is per kind and what is per cell

Ruling B-13 and your 15.4 fixed this:

| Fact | Where it lives | Why |
|---|---|---|
| mass | derived per kind: density of the substance × volume of the form | two titanium plates never weigh differently |
| durability | derived per kind from the substance's work of fracture | what varies per cell is DAMAGE, a sparse side table |
| liquidity, melting point, freezing point, flammability | columns of the substance row, added over time | a new behaviour is a new column, never a record change |
| style | the six-bit variant in the record | the one thing a player chooses per placement that changes nothing physical |
| the tree's shape byte, the terrain gap | the record | the server reads them |

A hull's mass is a running sum over its placed cells, kept by the hull's shard and stated on change:
one addition per weld, never a re-sum of a fifty-thousand-block ship.

**Example.** A pilot breaks one titanium plate and welds a half-scale titanium wedge. The plate was a
quarter of a cell of titanium, 1,107 kg. The wedge is half a cell at one eighth of its volume, 277 kg.
The hull's stated mass drops by 830 kg, once, on the change lane.

---

## 4. The identity digest: what may change and what may not

One test decides which column is in the digest: does this column change what a SAVED record means?

```text
   IN the digest (a saved record's meaning)    OUT of the digest (what a player may place NEXT)
   ------------------------------------------  -------------------------------------------------
   the kind's (substance, form, function)      the kind's variant count
   the form's geometry: volume, faces,         the kind's small-block scale mask
     collider, octants, mirror twin            an attachment kind's slots and legal faces
   the form's OBJECT flag and what the         any damage, thermal or power column
     object byte means for it                  any float, any render row
   the attachment kind's identity
   the density byte's resolution and sign
```

The digest is a running SHA-256 over the in-digest columns of rows 0 to n, in table order. It is a
PREFIX digest: a build that knows more rows than a store still matches the store's prefix.

Two checks in two places, both from ruling B-12 and B-13:

- **The store open.** The block store's own header row carries the registry length and the prefix
  digest at that length. This build opens the store if it knows at least that many rows and its
  prefix digest at that length matches. Otherwise it refuses, naming both digests. It never opens with
  a best-effort decode.
- **The client handshake.** A prefix digest too, not equality, plus a per-chunk refusal when a chunk
  names a kind the client does not hold. A content update therefore refuses only the pilots who fly to
  the chunks that hold the new kinds, and names the kind they lack.

What passes without a migration: appending a kind, a substance, a form, a function, an attachment
kind, a variant; raising a variant or slot count; widening a scale mask; retuning any
physical or render column. What refuses, loudly: changing an allocated row's triple, renumbering,
reusing a retired slot, changing a form's geometry, lowering a count, narrowing a mask, changing the
density convention.

**Example.** The team ships a fluted style for every hull plate. Every plate kind's variant count goes
from 3 to 4. The count is outside the digest, so every store on every shard opens, and every unpatched
client still connects. It cannot place a fluted plate, and it draws a received one with the fallback
the render registry names. In the same release the team also ships a carbon composite substance. That
appends rows. Every store still opens. An unpatched client is refused only at the first chunk that
holds a carbon plate, and the refusal names the kind.

---

## 5. Themes — dropped by the owner

There is no theme column, no theme in the digest and no placement allow-list by theme. A kind is
placeable anywhere any kind is placeable. Content packs are rows appended over time. The cap on kinds is
the record's sixteen-bit id, 65,536 registered rows, and nothing else.

**Example.** A player finds a carved-wood kind on a verdant planet, carries it home and welds it onto
a steel hull. Nothing refuses it, and both clients draw both, because there is one registry.

---

## 6. The shape catalogue

The investigation base has 23 shapes. With 1/8 m small blocks approved, three of them are plain small
boxes the sub-grid already expresses in one or two records: the half-cube, the quarter post and the
post. Keeping them as shapes would be two mechanisms for one thing, which is the fork the hard rules
exist to prevent. Dropping them leaves exactly twenty:

```text
   full        thin plates          slopes                  corners
   +------+    +------+ slab (1/2)  /|  wedge                /\  corner-out   \/  corner-in
   | cube |    +------+ plate (1/4) / |  ramp-low            /  \ ramp-low-out    ramp-low-in
   +------+    +------+ panel (1/8) /  |  ramp-high         /    \ ramp-high-out  ramp-high-in

   tetrahedra          stairs         connect families (their look derives from neighbours)
   /\ tetra            _|             wall   fence   railing   conduit
   \/ tetra-in       _|
                   _|
```

Every shape is achiral today. A future shape with a mirror image gets its twin as a row, so the baked
collider and the winding never branch. Rotations are an integer table of the 24 proper rotations of a
cube. The record's orientation field is six bits with the sixth bit zero and reserved, by your ruling
B-4. The thin plates stay as shapes because a shape is one record where a sub-grid of small blocks is
many.

Collision per shape is declared now and built later: the cube's box; one convex hull for thirteen
convex shapes; a compound of convex parts for the nine that are not convex; an empty collider class
kept for a harvestable bush if one is ever a cell.

In this slice every shape but the cube is declared unplaceable. The catalogue is data; the placement of
shapes is slice 12.

**Example.** A shipwright places a wedge on a hull's bow at rotation code 17. The registry knows the
wedge's volume, its face mask and its convex hull. The rotation table turns the hull's hull into the
bow's hull exactly, with integers. The client draws the same wedge from the same row.

---

## 7. What is built

```text
  crates/core/src/registry/
    mod.rs          the five tables behind one mechanism; the retired rule
    substance.rs    SubstanceId, the substance rows (density, work of fracture, the behaviour columns)
    form.rs         ShapeId, the twenty shapes: volume units, face masks, octants, collider class, OBJECT flag
    function.rs     FunctionId: none today; the P9 rows are additive
    kind.rs         BlockKindId rows: (substance, form, function, flags, variants, sub_scales)
    attachment.rs   AttachmentKindId rows: hud, hinge and the rail and piston reservations
    rotation.rs     the 24 proper rotations as integer matrices; the orientation code; the closure test
    digest.rs       the identity prefix digest over the in-digest columns, in table order
```

The first content, by the owner's word, is every substance the home planet's generator can emit plus
what a builder needs, about fifty rows, append-only and growing without a migration:

- **Bedrock:** granite, basalt, gabbro, andesite, limestone, sandstone, shale, slate, marble, quartzite, obsidian.
- **Loose ground:** dirt, loam, clay, sand, gravel, silt, mud, peat, permafrost.
- **Surface and water:** snow, ice, packed ice, water, salt water, lava, salt, ash.
- **Ores and minerals** (kinds in the registry; placed by live state, as the seed ruling says): iron ore,
  copper ore, tin ore, coal, bauxite, gold ore, silver ore, uranium ore, sulfur, quartz.
- **Grown and organic, as placed blocks:** oak wood, pine wood, birch wood, planks, bark, moss, fungus.
- **Building:** brick, concrete, mortar, glass, aluminium alloy, structural steel, stainless steel,
  titanium alloy, copper, carbon composite, ceramic, polymer, rubber.
- **Atmosphere:** air — the substance the medium is, never the removal kind. The Empty kind is the removal.

Every substance's density and work of fracture come from cited physical tables, so mass and durability
derive per kind and no number is typed by hand. The twenty forms and the function "none" complete the
first kinds. Slice 5's geology tables append what the strata need.

Dependencies: none new. The digest uses the SHA-256 crate already in the workspace.

---

## 8. The gates

| Gate | What it proves |
|---|---|
| The drift tripwire, all five tables | every `ALL` list, `from_number` and `def` agree; a hand-edited row fails the build |
| The digest-column test | append a kind and raise a variant count from 3 to 4: the stored-length prefix digest is UNCHANGED; change one row's triple: the store refuses, naming both digests |
| The rotation closure | the 24 rotations form a group; each shape's mirror, where one exists, is a row; every rotation of every face mask is exact |
| The volume rule | every shape's volume units are a multiple of 512, so a small block at every scale keeps a non-zero volume and weight |
| The decode rule | an unknown kind, a variant at or above the count, a rotation outside a shape's legal set, a retired kind at placement: each is a typed refusal, never a default |

**Measurement.** The digest's cost at a store open and at a handshake, over about 23 KiB of identity
columns: pass under 200 microseconds each. A number, never the word "negligible".

---

## 9. Laws, performance, seamlessness

- **HR3.** Features never match on a kind. They read registry data. The one exhaustive match per table
  is the registry's own.
- **HR5.** Tier-A at 100 percent; the tables are data and the tests walk every row.
- **SL5.** One registry on every host. No test-only catalogue.
- **SL10.** The in-digest columns are what the world identity binds: a client that knows fewer rows
  still matches the prefix, and a client that holds a different meaning for a row is refused.
- **SL8.** A content update refuses only the chunks that name unknown kinds, and names them. A style
  update refuses nothing anywhere.
- **Performance.** Every lookup is an index into a table. The digest runs once per open and once per
  handshake.

---

## 10. What you decide

| # | Question | Recommendation |
|---|---|---|
| S3-1 | Drop the half-cube, the quarter post and the post from the catalogue, so the sub-grid owns every small box | Yes: twenty shapes exactly |
| S3-2 | The first substances | ANSWERED: the fifty-row list of §7, every substance the home planet can emit plus the builder's; strike or add before it is frozen |
| S3-3 | A retired kind keeps its row, marked, with a replacement pointer | Yes |
| S3-4 | The digest algorithm | SHA-256 over the in-digest columns, already a dependency |
| S3-5 | The unknown-variant fallback on the client | the render registry names one per kind; the server never learns of it |

---

## 11. How it is built

I write it myself, table by table, with the tests beside each. After it is green, one Opus 5 refuter
attacks the digest's column set, the rotation table and the volume rule, and I answer every finding
before the slice is called done. No new design run is needed. Estimated size: about 1,200 lines of code
and 900 lines of tests.

---

## 12. Results (2026-09-07, after the refuter)

**What landed**, under `crates/core/src/registry/` (uncommitted until the owner's word):

```text
  mod.rs          GAP_STEPS_PER_CELL 128 + GAP_CONVENTION_TAG 1; GameScale {surface 200, base 268, SquareRoot}
  substance.rs    60 rows: void, air, 11 bedrock, 9 loose, 8 surface/water, 10 ores, 7 grown, 13 building;
                  KEY (identity) beside NAME (display); provenance Cited (9) / Provisional / Sentinel (void)
  form.rs         24 rows: void, terrain, sub-grid, object, the twenty shapes; class + collider numbered
                  (repr u8); legal-orientation mask (out of the digest); provisional marks named
  function.rs     "none"
  attachment.rs   hud, hinge, piston, rail clamp — key + makes_body in the digest, the rest out
  rotation.rs     the 24 proper rotations; code() refuses a reflection (None); 24 × 64 mask sweep
  kind.rs         523 kinds = Empty + 40 terrain + 24 building substances × 20 shapes + 2 holders,
                  ONE frozen recipe + APPENDED list, pinned by a recipe digest; PLACEABLE by rule
  digest.rs       SHA-256 prefix digest over KEYS (never numbers or names) + form geometry + gap convention
```

**Deviations from §7, each from a refuter finding:**

- The mason builds with the bedrock rows themselves. There is no "cut stone" substance: granite,
  basalt, sandstone, limestone, marble and slate are building substances too, in every shape.
- Every table row has a `key` (identity, in the digest) beside a `name` (display, out of it). The
  digest folds keys, never substance numbers, so a renumbering cannot collide with a swap.
- The gap convention folds its tag (radial, centre-sampled, negative inside solid), not the step count
  alone.
- The kind numbering is one ordered recipe. The frozen batch is built from three groups written today
  and never edited again; a later kind goes into the appended list at the end. A pinned digest of every
  row's triple fails the build on a reorder.
- Placeability is a stated rule, never a wholesale grant: Empty; a terrain cell of solid or loose
  matter that is not an ore; the cube shape of every building substance. Liquids, gases, ores, the
  holders and the nineteen other shapes are not placeable in this build.
- Integrity is computed with the kilojoule divide inside the root, at 2^20 fixed point, so a soft
  substance never floors to zero. The design's worked table reproduces exactly: packed soil 169, ice
  207, glass 379, granite 1 198, concrete 1 312, basalt 1 366, oak 11 985, steel 37 900, titanium 41 518.
- The scale mask is three bits (half, quarter, eighth). A scale above 3 is refused by name. The
  volume rule's divisor (512) is derived from that widest scale in the test, never typed.
- An orientation code is checked against the form's legal mask: a terrain cell, a holder and the cube
  accept only code 0; every other shape accepts every code until the mesh bake (slice 12) states each
  shape's canonical set (the masks are provisional and out of the digest).
- The four connect families carry one nominal volume marked provisional; the octant mask is not a
  column yet (it changes the digest, which is free before the first world is saved).
- The ore convention is one for all ten: the ore-bearing rock at a typical grade. Lava freezes at
  1 400 K; packed ice is at least as dense as ice; dry peat is 300 kg/m³.

**Measured:**

| What | Result |
|---|---|
| Registry tests | 24 pass; `cargo clippy --all-targets -D warnings` clean on core and bins |
| Store open (stamp + accept), release | 73.5 µs, gate 200 µs, the example asserts it |
| Handshake (accept), release | 35.9 µs, gate 200 µs |
| Recipe pin | `2_884_287_356_012_083_886` over 523 triples |
| Rotation pin | `5_946_405_317_151_576_421` |
| Tier-A coverage (`just coverage`, clean first) | PASS at 100 %; io-prod 95.12 / 95.33 % against the floor of 94 |

The digest cost rose from 21.7 / 10.9 µs before the refuter, because it now folds 523 rows with their
key strings instead of 423 rows of numbers. Both stay far inside the budget.

**The refuter.** `verdicts/slice_03_refutation.md`: 35 findings (13 WRONG, 3 BREAKS_LAW, 9 MISSING,
3 UNMEASURED_AS_FACT, 7 STANDS). Every one is answered above or in the code; none is deferred.
