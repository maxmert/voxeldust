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

# Storage and streaming — columnar formats, surface-only representations, and player megastructures

**Status:** ruling, awaiting owner decisions on the seventeen rows in §12. Not yet binding.
**Date:** 2026-08-03.
**Answers:** the owner's storage question of 2026-08-03, quoted verbatim below.
**Amends:** `docs/investigation/block_system_design.md` §2.4, §2.7, §3.9, §3.10, §5.1 — one defect, four gaps, no
structural change. **Does not amend** the owner rulings table, which wins over every section it touches.

---

## 1. The owner's question

> *"Should we discuss the way of storage? I thought if it would be better to apply it in Capacitor Format
> (similar to google big query). Or that will be bad, as we are mostly interested in the flat terrains,
> but not about what's inside. How we can optimize it to the maximum, so that we even can stream
> enormously big constructions, like cities, build by players? In the capacitor format, is there a way to
> get and render only blocks that contact with air / water (within certain deepness) / other transparent
> materials, so in the end it should be fast queries for each row. Or that will be too expensive to do?"*

Four separate questions live in that paragraph and each gets its own answer:

| # | The question | Answered in |
|---|---|---|
| **Q1** | Is a **columnar** storage format — BigQuery's Capacitor being the named example — the right shape for voxel data? | §4 |
| **Q2** | Is a **surface-only** representation viable — store and send only the blocks touching air, water or something transparent — and to what depth? | §5 |
| **Q3** | How do we stream a **player-built megastructure**, a city, where nothing is derivable from the world's recipe and every cell is a stored difference? | §6 |
| **Q4** | Is the "**fast query per row**" model affordable, or too expensive? | §5.4 |

---

## 2. The answer in one page

**Read only this page if you read nothing else.**

**The short version: the design you already approved is close to optimal, and the two ideas that make the
format you named fast are already in it under different names.** You do not need a redesign. What you
need is one bug fixed, three rules written down that nobody wrote down, and one optional addition worth
about two hundred times its own weight. Everything else on this page is confirmation.

**Your first question — should storage be organised the way the big-data format organises it?** The good
parts of that idea are already in the design. That format is fast for two reasons. The first is that it
writes down each distinct value once and then refers to it by a short number instead of repeating it —
and that is exactly what the design already does inside every box of blocks: a box that contains only
sixteen distinct kinds of block stores four bits per block instead of sixteen, which is four times
smaller, and a box that is entirely air or entirely rock stores *nothing at all*. The second reason is
that it stores each property in its own separate place, so a question about one property never has to
read the others — and the design already keeps a block's identity, its damage, its temperature, its
fluid, its wiring and its screen overlay in physically separate tables, so drawing the world reads only
the identity and never touches any of the rest.

The part of that format that is genuinely distinctive is something we cannot use, and not because it is
expensive — because it is **meaningless here**. In a spreadsheet the rows are an unordered bag, so that
format shuffles them into whatever order compresses best, and shuffling costs nothing because nobody
cares which row came first. In our world the row *is the position*. Shuffling the rows would not
re-encode the world, it would build a different world. So there is nothing to gain there, and everything
to lose: the only thing that keeps the drawing code fast is that a block's position tells you exactly
where its bytes are, without looking anything up.

There is one genuine hole, and it is small: nothing squeezes the saved bytes at the very end, the way a
zip file does. That is worth taking, on disk only — but it means adding an outside library, so it is
yours to decide, and the costed choices are in the decision list at the end.

**Your second question — could we store and send only the blocks that touch air, water or glass?** You
have already got that, and in a stronger form than you were asking for. Nothing about the natural
landscape is ever saved or ever sent — not the surface, not the interior, nothing. The whole landscape is
recomputed from the world's recipe wherever it is needed, and the only thing on disk is what a player
changed. Zero is not a number a shell can beat. And on the drawing side, the exact question you asked —
*which blocks touch air, water or something see-through* — is already computed, and it costs less than a
ten-thousandth of a second for a whole box of a quarter of a million blocks, including telling water and
glass apart. It is cheaper to work that out afresh every time than to remember the answer, because
remembering it would cost more disk writing than the calculation costs, and one dug block would spoil the
remembered answer for eight boxes at once.

**"Within a certain depth" is where the idea dies, and it dies for a reason worth knowing.** There is no
depth that works. When the world is drawn from far away it is drawn in bigger blocks — two metres, then
four, then eight, up to a hundred and twenty-eight. For a hollow shell of a mountain still to look like a
mountain at the hundred-and-twenty-eight-metre size, the shell would have to be a hundred and twenty-eight
metres thick. That is not a shell, that is the mountain. And for digging there is no depth at all, because
you can always dig one block deeper than whatever we chose.

**Your third question — can we stream a player-built city?** Yes, comfortably, and the numbers are
better than you would expect. **A ten-million-block city costs about four bytes per block, all in — about
forty-one megabytes.** A planet's whole budget holds about forty such cities. Seen from a hundred
kilometres the entire city is five kilobytes; from six kilometres it is half a megabyte; standing in the
middle of it, twenty-three megabytes, which is under two seconds on a normal connection — and the
skyline is already drawn from the five kilobytes that arrived in the first fraction of a second. The one
case that genuinely does not fit is flying *fast* and *low* over continuously built ground: at the design's
own top speed at head height that is about two hundred megabits a second. Flying higher fixes it
automatically, because the world is drawn in bigger blocks up there.

**And a comparison worth having: clear-cutting a forest costs slightly more per square kilometre than
building a city on it.** Logging is the bigger threat to a planet's storage budget than construction is.

**Your fourth question — is asking that question per row affordable?** Yes. It already runs, it is
already measured, and it is about a third of a nanosecond per block.

**Where your instinct found something real.** Three places. First, you are right that we mostly care
about the surface — and the design agrees so completely that it stores nothing at all for the interior,
which is why storing a shell would make things *worse*, not better. Second, you are right that repetition
is where the wins are, and the strongest single improvement available anywhere in this analysis comes
from exactly that instinct: describing a wall as *one rectangular box of stone* instead of as three
thousand individual stone blocks. On a typical hollow building that is **two hundred and forty-four times
smaller**, it needs no new library, it is about two hundred and fifty lines of whole-number arithmetic,
and it can never lose because we compute both forms and keep the smaller one. Third, you are right to ask
about squeezing bytes — there is a real hole there, it is just smaller than the box idea.

**What is actually wrong.** Four things, none of which is what you asked about:

1. **A one-line bug in the loop that keeps the far-away view up to date.** As written, it never stops
   early. It is supposed to notice "nothing changed above this point" and stop; the check it makes can
   never be true in the ordinary case, so every single block a player places walks all the way to the
   top and announces that ten large regions changed when none of them did. Free to fix now, before it
   exists.
2. **The cost of scattered edits is understated by five and a half times.** The design plans for about
   one summary entry per edit. That is right when players build in one place. When ten thousand players
   each leave ten thousand marks scattered across a whole planet — which is what exploring looks like —
   it is six entries per edit, and the planet's budget is exceeded three times over. No compression idea
   in this whole analysis helps with that, because a scattered edit is already a single block.
3. **The widest record in the whole design has no rule that stops it growing.** It is the sunlight
   record, one entry per column of the world, and depending on a policy nobody has written it is
   somewhere between half a gigabyte and seven and a half gigabytes. Every other saved thing has a rule
   that deletes it when it matches what the recipe says; this one does not. Free to fix, permanent if not.
4. **The same problem for snow and grass.** One snowfall over one player's neighbourhood writes twenty-one
   megabytes, for no player action at all, and never deletes it.

**In order of what I would do:** fix the loop; write the deletion rule for sunlight, snow and grass; add
the box description; add a row to the cost table for scattered edits and re-derive the planet budget from
it; reserve the label for a squeezer even if we never use one. Do **not** build the big-data format, do
**not** build a hollow-shell world, do **not** build the shared-shape graph that the research literature
recommends for static scenery.

---

## 3. What the design already has — the inventory, with sizes

This is the committed design as written, established before any recommendation, because a recommendation
that re-invents what exists is worse than useless. Every figure below was re-derived rather than copied.

### 3.1 In memory: the palette-indirected chunk container

A chunk is 62³ = **238,328 cells**. The container is a per-chunk palette of 4-byte `PaletteEntry` rows
(`block_type: u16`, `orientation: 5 bits`, `placed: 1 bit`) plus a bit-packed index array at
`ceil(log2(palette_len))` bits, floor 4, with a **zero-bit single-value fast path**.

| Distinct `(type, orient, placed)` triples | Index bits | Index array | + palette | Total | Against dense u16 |
|---|---|---|---|---|---|
| 1 (all air, all granite) | 0 | **0 B** | 4 B | **4 B** | unbounded |
| 16 (terrain) | 4 | 116.4 KiB | 64 B | **116.4 KiB** | **4.00×** |
| 256 (a built chunk) | 8 | 232.7 KiB | 1.0 KiB | 233.7 KiB | 1.99× |
| 4,096 | 12 | 349.1 KiB | 16.0 KiB | 365.1 KiB | 0.78× (worse) |
| 24,000 (1,000 types × 24 orientations) | 15 | 436.4 KiB | 93.8 KiB | 530.1 KiB | 0.44× (worse) |
| — dense `u16` reference — | — | — | — | 465.5 KiB | 1.00× |

The 4.00× at sixteen triples costs **0.054 %** in dictionary overhead. That is textbook dictionary
encoding, arrived at independently, and the decisive property is the one usually undersold: most of a
planet is a single-value chunk at zero index bytes.

### 3.2 On disk: the durable record and the two chunk-record encodings

**The durable placement record is eight bytes** (owner ruling R1, which supersedes §2.7.1's five-byte
layout — that table row is stale): `cell 18 | state 32 | reserved 14`, reserved bits **rejected on decode
if non-zero**, of which two were spent by R6 on three-state provenance, leaving fourteen.

**The semantic rule is the load-bearing one:** a chunk record is *always* the set of player-authored
cells and **never** embeds generator output. There is no encoding in the format that can hold an
unedited cell.

| Encoding | Bytes | Wins when |
|---|---|---|
| Sparse — ascending `(cell, state)` list | `8n` | few edits |
| Masked-dense — 238,328-bit authored-cell bitmask + palette + packed states | `29,791 + palette + n·bits/8` | many edits |

The mask is **29,791 bytes exactly** (238,328 ÷ 8). The compactor computes both and takes the smaller.
The published crossover is `8n ≥ 29,791 + n` ⇒ **n ≥ 4,256 cells (1.786 % of a chunk)**; including the
palette term that appears in the same formula it is **4,403** with a 256-entry palette — a 3.5 %
correction, immaterial. **The pyramid's crossover is a different number entirely and §9.7 corrects it.**

### 3.3 Identity separated from mutable state — six tables, six record widths

| Table | Record | Bytes |
|---|---|---|
| `block_wal` | `(cell 18 \| state 32 \| reserved 14)` | 8 |
| `chunk_delta` | TLV-framed `ChunkRecord`, two encodings | variable |
| `chunk_pyramid` | `(cell 18 \| substance 16 \| occ_mask 8 \| fill_256 8 \| reserved 14)` | 8/entry |
| `block_state` | `(u32 local, u16 damage_dp, u16 function_dp, u16 state_bits, u8 growth)` | 11 |
| `block_thermal` | `(u32 local, u16 temp_dk)` | 6 |
| `block_fluid` | `(u32 local, u8 fluid)` | 5 |
| `block_config` | `(u32 local, u32 config_ref)` → TLV blob | 8 |
| `block_cover` | `(u32 local, u8 face, u32 cover_ref)` | 9 |
| `chunk_meta` | `(u64 chunk_key, u64 last_ticked_universe_tick, u32 registry_len, …)` | 20+ |
| **`column_sky`** | **`[u16; 62²]` per chunk column** | **7,688** |

Six dense mutable fields at 62³ would be **1.705 MiB per chunk** of almost entirely zeros. The sparse
in-memory form is two parallel sorted vectors (`keys: Vec<u32>`, `vals: Vec<T>`) rather than
`Vec<(u32, T)>`, because the tuple pads to 8 bytes for a `u16` payload and the split form costs 6 — a
**1.33×** saving, and deterministic iteration for free. Dense promotion is computed exactly: a `u16`
payload crosses at 79,443 entries (**33.3 %**), a `u8` payload at 47,666 (**20.0 %**).

### 3.4 The edit pyramid, and the rule that keeps it small

Only tier 0 is writable; every coarser rung is derived, enforced by the `Tier0Key` newtype rather than by
convention. An entry is 8 bytes and exists **iff its summary differs from what the generator would
produce at the same address**; it is deleted the moment it matches. Storage blocks are one per
`(tier, chunk)` pair, in the same two encodings as the delta.

| Edit shape | Series | Entries per edit | Bytes per edit |
|---|---|---|---|
| Compact 3-D cluster | Σ 8⁻ᴸ | 0.143 n | 1.14 |
| Surface / wall / floor (2-D) | Σ 4⁻ᴸ | 0.333 n | 2.67 |
| Line / pipe (1-D) | Σ 2⁻ᴸ | ≈ 1.000 n | 8.00 |
| Isolated (stated hard bound) | `tier_depth − 1` | 12 at the starter body | 96 |

Planning figure: ~1.1 entries per edit mixed, **~880 MB of pyramid at 100 M edits**, against ~800 MB of
sparse delta plus a few hundred MB of masked-dense chunks — **under 1.7 GB per heavily-played planet**.
§9.2 shows that this table is missing the row that matters at scale.

### 3.5 The wire, and what the client holds

- `BulkMsg::ChunkDelta { realm, chunk: ChunkKey, epoch, encoding: DeltaEncoding, payload: TlvBytes }`
  on the reliable-paced bulk carrier. The tier rides inside the key; the pyramid is TLV tag 4.
- `BulkMsg::DeltaManifest { realm, cell, keys: Vec<ChunkKey>, digest }` — **8 bytes per key**, one
  manifest covering every rung of an area-of-interest cell.
- Residency: meshes for the whole tiered set (~6,800 chunks at a 100 km view); **voxels only for tier-0
  chunks inside the edit/collider radius and for any chunk carrying a delta**; everything else dropped
  after meshing and regenerated on demand. Coarse-first ordering is a rule, not a heuristic.
- Rung *L* is used out to **785.7 × 2ᴸ metres**: 786 m, 1.6 km, 3.1 km, 6.3 km, 12.6 km, 25.2 km,
  50.3 km, 100.6 km. Tier 0 is ~505 chunk columns; every rung above it is ~379, **independent of the
  planet's radius**.

### 3.6 The mesher's surface query — already the owner's Q2 and Q4, already measured

Two occupancy bitmasks — `occ_ne` (opaque ∪ cutout ∪ translucent) and `occ_solid` (opaque only) — with
`col & !(col << 1)` applied to each and OR-ed, over `u64` bit columns on a 62-cell axis with a one-cell
apron each side: **62 + 2 = 64**, exactly one word, no spill, at every rung. Measured on the reference
implementation: **65 µs opaque, 90 µs with the dual transparency masks**, per 238,328-cell chunk —
**272.7 and 377.6 picoseconds per cell**. Our own mesher is booked to rise to ~95 µs once the twelve face
masks land, so the honest band for *our* future code is **~95–120 µs**, and that is the number to quote.

---

## 4. Columnar formats — Q1

### 4.1 What they optimise for

Capacitor is a scan-optimised analytics format. Its published techniques are run-length encoding,
dictionary encoding, bit-vector encoding and frame-of-reference encoding, plus Dremel-style definition
and repetition levels so a query reads only the columns it asked for. Its *distinctive* technique is
**approximate-optimal row reordering**: it permutes rows to lengthen runs, solving an NP-complete problem
heuristically, weighted by which columns queries select and filter on. Parquet and ORC share the family
shape — immutable files, a smallest decodable unit of a page or row group of thousands of rows, updates
arriving as new files merged by compaction. Arrow, the in-memory member of the family, is the one that
states the decisive property directly: every layout offers constant-time random access **except**
run-end encoding, which requires a logarithmic binary search.

Every one of those design inputs presupposes a large analytical scan over a subset of columns. None
presupposes a point lookup, and none of these formats has a point-update path at all.

### 4.2 The eleven techniques, scored against the committed design

| # | Technique | In the design? | Where |
|---|---|---|---|
| 1 | **Dictionary encoding** | ✅ **yes, in the stronger joint form** | the per-chunk palette |
| 2 | **Column separation** | 🟨 **partial** | five tables, but `block_state` is four columns in one 11-byte row (§9.6) |
| 3 | **Validity bitmap + packed present-values** | ✅ yes, exactly Arrow's layout | the masked-dense chunk record |
| 4 | **Per-page encoding selection** | ✅ yes, and **better than Parquet** | the compactor computes both sizes exactly |
| 5 | **Zone maps / min-max statistics** | ✅ yes, and better positioned | the relief envelope, derived and never persisted |
| 6 | **Immutable base + delta merge** | ✅ yes | `block_wal` append-only + three-step compaction |
| 7 | **Encoded-at-rest / dense-in-memory split** | ✅ yes, three forms | `ChunkRecord` / `Chunk` / the 64³ resolved buffer |
| 8 | **Hierarchical summaries** | ✅ yes — **no analytics format has this at all** | the edit pyramid |
| 9 | **Run-length encoding** | 🟨 **only the degenerate single-run case** | the zero-bit single-value chunk (§4.4) |
| 10 | **Row reordering** | ⛔ **inapplicable by type, not by cost** | §4.3 |
| 11 | **A general block codec** | ❌ **absent — the one genuine gap** | §9.5 |

**Seven fully present, two partial, one inapplicable, one absent.** A claim circulating in review that
nine are fully present overstates rows 2 and 9; the correction does not change the verdict.

### 4.3 The decisive point: row reordering is not expensive here, it is undefined

In a table, rows are an unordered bag: permuting them is semantics-preserving, so any run-length gain is
free. In a chunk, `CellIndex` is defined as `(c*62 + b)*62 + a` with `a` fastest, `0..238_327` — it is
not a surrogate key attached to a position, **it is the position**. Permuting cells does not re-encode
the data; it produces a different world.

And the index is already frozen into three permanent formats:

1. the 8-byte WAL record, bits 63..46;
2. `PyramidEntry`, bits 63..46;
3. the bit order of the masked-dense 238,328-bit authored-cell mask — whose ascending order is also what
   makes the WAL's sorted drain deterministic.

The escape hatch — keep a permutation vector and restore order on read — is fatal for a fourth reason:
the mesher packs a 62-cell axis plus a one-cell apron each side into exactly one `u64`, and an
indirection layer destroys that outright.

The only surviving legitimate form of the idea is **choosing the fixed linearisation**, and the design
has already taken it. That branch is also defensible on run-length grounds, which is not obvious: with
`a` fastest and the radial axis outermost, a surface chunk's roughly 52 uniform horizontal levels each
collapse to one run and only the ~10 straddling levels fragment; with the radial axis fastest, every one
of 3,844 columns crosses the surface and yields at least two runs. Roughly **2× fewer runs**, and better
for buildings too, whose floors and ceilings are horizontal slabs. Morton order is rejected on
arithmetic: 62 is not a power of two, so a Morton code needs 6 bits per axis = 262,144 codes, wasting
23,816 of them, breaking the `0..=238_327` range check that decode rejects on, and destroying the
contiguous-axis property the whole binary mesher rests on.

### 4.4 Why voxel access is different — the six patterns, and the one that decides it

| # | Pattern | Shape | Columnar helps? |
|---|---|---|---|
| 1 | **Meshing** | full 62³ + apron, resolved into a 64³ `u16` buffer (512 KiB/worker), swept as `u64` bit columns — 3,844 columns × 3 axes = 11,532 word ops against 238,328 scalar iterations, a ~20× saving that exists **only** under dense fixed-stride addressing | only via column *projection*, already banked |
| 2 | **Collision** | neighbourhood point queries, tier 0 only, always resident | ❌ hurts |
| 3 | **Raycast / block targeting** | one point query per cell stepped, per player per frame | ❌ hurts |
| 4 | **Flood fills** (light, sky, support, heat, radiation, signal — one shared kernel) | queue-ordered walk over the 6-neighbour graph; **read order is set by the queue, not by storage order** | ❌ hurts most |
| 5 | **The edit path** | point writes; under run encoding a mid-run write splits one run into three — an O(n) memmove that can *grow* the record | ❌ hurts |
| 6 | **Pyramid walk-up** | 8 sibling lookups per rung, always inside one storage block because 62 is even | neutral |

**Exactly one of the six is scan-like, and even that one is a `SELECT *` over one tiny partition — the
query shape columnar is worst at relative to dense row storage.**

**The cost of getting it wrong, in one number.** Placing a torch runs a light flood of radius 15, which
touches (2·15+1)³ = **29,791 cells**. Against the dense bit-packed container the lookup is
`word = (i·bits) >> 6`, one or two loads, ~4 ns hot: **119 µs**, comfortably inside the ~1 ms edit
budget. Against a run-encoded column the lookup is Arrow's documented O(log n) binary search; at ~62 runs
per column that is ~6 probes, and because a flood's walk order is queue-driven most probes miss cache. At
20 ns per probe that is **3.6 ms**; at a full DRAM miss of 80 ns, **14.3 ms**. Either end is a **3.6× to
14× overrun of the entire edit budget for one torch**, on a path six mechanics share. That is the
concrete, arithmetic reason columnar encoding cannot live in the live container.

### 4.5 Joint dictionary beats per-field columns, provably

The palette dictionaries the **joint** `(block_type, orientation, placed)` triple. Splitting it into
three columns is strictly worse, because the fields are strongly correlated:

| Chunk | Joint palette | Split into three columns | Verdict |
|---|---|---|---|
| Player-built, ~200 observed triples of 1,440 possible | 8 bits = 232.7 KiB | 5 + 5 + 1 = 11 bits = 319.9 KiB | **joint 37 % better** |
| Terrain, orientation always 0 | 4 bits = 116.4 KiB | 11 bits = 319.9 KiB | **joint 2.75× better** |

This is `H(A,B,C) ≤ H(A)+H(B)+H(C)` with equality only under independence, and voxel identity fields are
never independent. So the design has not merely matched columnar's dictionary idea; it took the strictly
stronger form of it.

### 4.6 The one place a columnar format still earns its keep

**The economy's trade ledger, and the run telemetry — never blocks, and only ever as a one-way export.**

A trade ledger is append-only, immutable, wide (tick, buyer, seller, item, quantity, price, fee, market,
realm), never point-updated, and queried analytically: *median iron price in this system over thirty
days*, *which materials are net-imported*. That is Capacitor's workload verbatim, and it is the only
dataset in the project with that shape. The standing law that the economy is a decoupled overlay with a
one-way dependency is precisely what makes a columnar file safe there: it cannot appear on the
realm/physics/lifecycle path. The same applies to the harness's run manifests, wire traces and chaos
output, which are scanned across runs.

Two guardrails, both non-negotiable:

- **HR1.** A shard's redb is private. A columnar file must be produced by an **export** from the owner,
  never be the source of truth, and nothing in `sim` or `node` may read it — otherwise determinism breaks
  on a file whose contents depend on when the export ran.
- **Dependency hygiene.** `arrow` and `parquet` are large trees (flatbuffers; thrift plus zstd/snap
  respectively). They belong in an analytics-only crate outside `bins → node → sim → wire → core`, never
  in a Tier-A crate that must reach 100 % region+branch.

The cold-tier/backup case is different again and does **not** want columnar: a spun-down realm's redb
file is a scan-once, restore-whole object, so it wants a codec, not a format.

---

## 5. Surface-only representations — Q2 and Q4

### 5.1 The design already commits to something strictly stronger than a shell

`ChunkRecord` semantics: *"always, semantically, the set of player-authored cells; it never embeds
generator output."* An unedited region costs **zero bytes at every rung**, and the wire carries no
unedited cell in any encoding.

Pre-generating the starter world's full band would be 6 × 4096² × 272 = 2.74 × 10¹⁰ chunks = **2.9 PiB**
at 4 bits/cell, and 1,551× that on an Earth-sized body. What is actually stored is **under 1.7 GB**,
proportional to edits and independent of the body's radius.

**A one-cell shell of a mountain is ≈ 6n² cells of NEW stored data replacing 0 bytes.** No surface-only
scheme can beat zero. For generated terrain, the design did not merely arrive at the owner's idea; it
arrived at the strictly stronger version — store neither the shell nor the interior.

The generation side has the same idea again, in the **generator band**: a conservative per-(tier, chunk
column) min/max solid-radius pair answers "all air / all water / all deep rock" with zero noise
evaluations, and on an Earth-like relief profile a 272-chunk vertical column has **1–3 straddling
chunks** — a ~100× reduction in the chunks needing any generation at all. Derived, cached in a bounded
LRU, never persisted.

### 5.2 What genuinely needs the interior

Seven consumers, not the two usually assumed:

| Needs only the surface | Needs the interior |
|---|---|
| rendering (an all-solid chunk emits **zero** quads because every face is culled) | digging |
| collision contact (parry's `Voxels` classifies Empty/Vertex/Edge/Face/**Interior** so interior voxels never produce contacts) | mass and centre of mass (summed over every cell's density × volume) |
| raycasting from outside | `realm_pcu` — the demand scheduler's per-realm cost sum |
| pressurisation (solid is a barrier; what is behind it is irrelevant) | structural support (load transmits *through* solid blocks) |
| light and sky exposure | explosion ray-march (~23,000 queries per blast, marching through material) |
| fluids, weathering, growth, sound occlusion | fire eating into a wooden mass |
| the whole decoration layer | heat diffusion |

Every one of the seven is satisfied for **terrain** by the recipe at a measured 3.71 ns/cell. **None** is
satisfied for a player-built structure by anything at all. So the interior question is entirely a
player-built question — which routes it to §7, where boxes beat shells outright.

### 5.3 "Within a certain deepness" has no finite answer

| Consumer | Required depth |
|---|---|
| Face emission | 1 |
| Mesher apron | 1, with cross-chunk completeness |
| Collider classification | 2 (else the shell's inward faces are spurious) |
| Spall | 3 (`spall_max_depth` default) |
| Support flood | 32 (`MAX_SUPPORT_RADIUS`) |
| Explosion | until blast resistance exhausts — unbounded |
| **The detail ladder** | **2ᴸ — 128 cells at rung 7, 4,096 at the starter body's top rung** |
| Digging | whatever the player built — no finite value exists |

The ladder is what kills it. A tier-*L* pyramid cell covers 8ᴸ tier-0 cells and its summary is a function
of its eight children **alone**. A coarse cell wholly inside a mountain contains no shell cell, so it
coarsens to empty and the mountain renders hollow. Repairing that needs an enclosure test, which is a
flood over the realm — **not** a function of eight children. That converts a local eight-sibling lookup
inside one storage block into a global query per edit, and destroys the order-independence the pyramid's
whole correctness argument rests on. Pruning breaks too: *"delete iff the entry equals the generator's
own summary of its own address"* has no stable comparand for an enclosure-derived value.

**This is why a shell cannot be adopted "just for storage" and left out of the render path: the pyramid
is where storage and render meet.**

### 5.4 The arithmetic that settles it — and answers Q4

For a solid n³ mass, a one-cell shell holds ≈ 6n² cells; a run encoding holds exactly n² runs; a box
encoding holds **one box**.

| n | Cells | Shell (8 B/cell) | 1-D runs (8 B/run) | 3-D boxes (16 B/box) | Shell ÷ runs |
|---|---|---|---|---|---|
| 10 | 1,000 | 3,904 B | 800 B | **16 B** | 4.88 |
| 20 | 8,000 | 17,344 B | 3,200 B | **16 B** | 5.42 |
| 62 | 238,328 | 178,624 B | 30,752 B | **16 B** | 5.81 |
| 100 | 1,000,000 | 470,464 B | 80,000 B | **16 B** | 5.88 |
| 200 | 8,000,000 | 1,900,864 B | 320,000 B | **16 B** | 5.94 |

**A shell is ~6× larger than a run encoding at every size, and thousands of times larger than a box
encoding.** A run encoding is already a surface representation — it records where each row *enters* and
*leaves* the solid, a shell in one dimension instead of three, at a sixth of the cost, with no interior
reconstruction, no enclosure oracle and no change to the pyramid. And a box encoding beats both.

**Q4 — the per-row query.** It already exists, it is already measured, and it is **cheaper to recompute
than to store**:

- **Recompute:** 65 µs opaque / 90 µs including the water and glass distinction per chunk (reference
  implementation); ~95–120 µs for our own future mesher. Extending it to *depth N* is a bitwise dilation
  — `m |= (m << 1) | (m >> 1)` per column per axis, ~8 extra ops per column at N = 4, ~25–35 µs on top,
  so "every block within four cells of air, water or glass" lands at **~120–155 µs per chunk**.
- **Store:** one bit per cell = **29,791 B per chunk**, roughly 60 µs of pure NVMe write bandwidth before
  any fsync — and it must be invalidated across up to 8 chunks per edit by the mesher's apron rule, so it
  is dirtied more often than it is read.

**The rule to write into the mesher's doc comment: the surface set is a computed view, never a stored
table.** This is the same conclusion the design already reached from a different direction when it
deleted the content-keyed mesh cache.

### 5.5 One more reason the prize is small: the client's memory is already surface-dominated

| At a 100 km view | Bytes |
|---|---|
| Mesh, measured natural terrain, packed vertex | **120 MB** |
| Mesh, conservative allowance, packed vertex | **273 MB** |
| Resident block field on unbuilt ground (27 chunks in the collider radius at 4 bits) | **3.07 MiB** |

**The surface representation is the large one and the volumetric representation is the small one**,
which inverts the intuition the proposal rests on. Amdahl's law then bounds the entire prize: a shell
scheme that made the block field free would save one to three percent of client memory, and would cost
the pyramid's closure property to do it.

### 5.6 Verdict

**Reject surface-only as a storage or wire format. Keep the three surface-only mechanisms the design
already ships — the generator band, the mesher's face masks, and the pyramid's coarse summaries — and
rule the fourth out explicitly (§12, D-3) so it does not return every time someone reads the storage
numbers.**

---

## 6. The megastructure — Q3

### 6.1 The model, stated so it can be challenged

Ten million player-placed blocks over a **2.9 km²** footprint — a square about 1.7 km on a side, roughly
2,400 buildings of 30 × 20 × 12 m plus authored pavement. At 62 m per chunk edge that is **754 tier-0
chunk columns**, **13,255 authored cells per chunk** — 5.6 % of a chunk, 3.1× past the crossover, so
every city chunk goes masked-dense.

### 6.2 The city in bytes, with the encoding rule applied at *every* rung

Three of the five investigations charged the pyramid a flat 8 bytes per entry at every rung. That is
wrong: a coarse city chunk holds thousands of entries and flips masked-dense exactly as tier 0 does. The
corrected figures:

| Rung | Cell size | Entries | Chunks | Per chunk | Encoding | Rung total | Flat-8 B would be |
|---|---|---|---|---|---|---|---|
| **0** | 1 m | 10,000,000 | 754 | 13,255 | masked-dense 39,989 B | **30.17 MB** | 80.00 MB sparse |
| 1 | 2 m | 2,500,000 | 189 | 13,255 | masked-dense 44,070 B | 8.31 MB | 20.00 MB |
| 2 | 4 m | 625,000 | 47 | 13,255 | masked-dense 44,070 B | 2.08 MB | 5.00 MB |
| 3 | 8 m | 156,250 | 12 | 13,255 | masked-dense 44,070 B | 0.519 MB | 1.25 MB |
| 4 | 16 m | 39,062 | 3 | 13,255 | masked-dense 44,070 B | 0.130 MB | 0.312 MB |
| 5 | 32 m | 9,766 | 1 | 9,766 | masked-dense 40,580 B | 0.041 MB | 0.078 MB |
| 6 | 64 m | 2,441 | 1 | 2,441 | sparse 19,528 B | 0.020 MB | 0.020 MB |
| 7 | 128 m | 610 | 1 | 610 | sparse 4,880 B | **0.005 MB** | 0.005 MB |
| 8–11 | 256 m–2 km | 203 total | 4 | — | sparse | 0.002 MB | 0.002 MB |
| | | | | | **Pyramid total** | **11.11 MB** | 26.67 MB |

> **A ten-million-block city is 30.17 MB of delta + 11.11 MB of pyramid = 41.3 MB, or 4.13 bytes per
> authored block, all in.** A planet's ~1.7 GB budget holds **41 such cities**.

Two things fall out that nobody had quoted. First, **the design's own budget is conservative on the
pyramid by 2.4×**, because it charges 8 B/entry flat where the encoding rule already halves it. Second,
against a naive baseline of 8 B sparse + 2.67 B pyramid = 10.67 B/cell, **the committed design is
already 2.6× ahead with no new machinery.**

**Cross-check.** Four independently-built city models from the investigation pool normalise to 4.32,
4.82, 5.75 and 5.90 B/cell against my 4.13. Every one within 1.5×, despite wildly different building
geometry. The number is robust; the individual byte counts are not.

### 6.3 What the detail ladder already removes — and the honest correction

| Observer distance | Rung | Bytes for the whole city | Attenuation |
|---|---|---|---|
| standing inside | 0 | 30.17 MB | 1× |
| 1.6 km | 1 | 8.31 MB | **3.6×** |
| 3.1 km | 2 | 2.08 MB | 14.5× |
| 6.3 km | 3 | 0.519 MB | 58× |
| 12.6 km | 4 | 0.130 MB | 232× |
| 25.2 km | 5 | 0.041 MB | 736× |
| 50.3 km | 6 | 0.020 MB | 1,509× |
| 100.6 km | 7 | **0.005 MB** | **6,034×** |

**The ladder's attenuation is real but BACK-LOADED, and the headline "64× to 1,024×" figures circulating
in review are inflated 2.6× by comparing against the *sparse* tier-0 figure the compactor would never
choose.** The first rung buys only 3.6× — and the first rung is the one a player crosses when he walks
out of the city. The reason is arithmetic rather than accidental: entry count falls 4× per rung, but the
fixed 29,791-byte mask does not fall at all until the entry count drops below the crossover, which
happens around rung 5. **Anyone budgeting the ladder as "8× per rung" is wrong by a factor of two at
exactly the rung where the bytes are largest.**

### 6.4 What is actually left — the three quantities that matter

| Quantity | Bytes | Time at 100 Mbit/s |
|---|---|---|
| **Know the city exists** — every rung's keys at 8 B, plus the rung-6 record that puts it on screen as a shape | **8.1 KB + 20 KB** | < 0.01 s |
| **Collide with it** — tier 0 only, over the physics radius floor (`max_tool_reach + one chunk edge` ≈ 70 m = 4 columns) | **160 KB** | 0.01 s |
| **Draw it, standing in the middle** — 505 tier-0 columns inside the 786 m disc + rung 1 for the rest | **~23 MB** | **1.8 s** |

**The bytes that must land before the player can take a step are under two hundred kilobytes.** The
twenty-three megabytes is the city *sharpening* behind him, not a gate on his movement — and by the
coarse-first rule the skyline is already drawn from bytes that landed in the first fraction of a second.
That is exactly the shape the seamless law wants, and it is a property of the existing residency rule,
not something that needs adding.

### 6.5 Streaming while moving

New tier-0 columns per second is `ρ₀ = 1.284·v`, and each new column in a city is one delta-carrying
chunk at 39,989 B.

| Motion | Columns/s | Tier 0 alone | With coarse rungs (low altitude) |
|---|---|---|---|
| Walking, 5 m/s | 6.4 | 2.05 Mbit/s | ~4 Mbit/s |
| Ground vehicle, 30 m/s | 38.5 | 12.3 Mbit/s | ~25 Mbit/s |
| Low flight, 100 m/s | 128.4 | 41.1 Mbit/s | **~82 Mbit/s** |
| **Fast low flight, 528 m/s** (the design's own derived `v_max`) | 678 | **217 Mbit/s** | **~434 Mbit/s** |
| 528 m/s at 5 km altitude (`L_min = 3`) | — | — | ~57 Mbit/s |
| 528 m/s at 10 km altitude (`L_min = 4`) | — | — | **~28 Mbit/s** |

**Walking and driving are trivial. Fast low flight over continuously built ground is the one case that
genuinely breaches, and altitude is the automatic escape**, because at altitude *h* the finest resident
rung is `⌈log₂(h/786)⌉` and everything finer is simply not requested. The dangerous case is not the
spacecraft; it is a hovering gunship — which is precisely the case the design already names as expensive
for *natural* terrain, made about 2.3× worse by the city.

None of this touches the datagram budget: chunk deltas ride the reliable-paced bulk carrier, never
datagrams, and the measured latency gate already carried ~180 MB/s of bulk on one loopback connection
while 20 Hz snapshot datagrams held p99 at 678–728 µs against a 15 ms budget. The city's worst sustained
demand is 16 % of that. The honest qualifier: 180 MB/s is loopback, so what that proves is that the
server-side lane is not the constraint. The constraint is the player's downlink.

### 6.6 Cities against forests — the comparison nobody made

| | Permanent delta per km² |
|---|---|
| A player-built city (this model) | **14.2 MB/km²** |
| A clear-cut forest (19,406 trees × 1,845 B over a 1.941 km² disc) | **18.4 MB/km²** |

**Clear-cutting costs 30 % more per square kilometre than building on the same ground.** 41.9 clear-cut
discs — 81.3 km², a square 9.0 km on a side, **0.025 % of the starter body's surface** — exhaust the
whole delta budget, against 41 cities. So the owner's Q3 is aimed at the *cheaper* of the two hazards:
an organised logging operation bricks editing on a planet before an organised construction project does.
That risk is already ledgered and already discharged by the prune rule; no storage format touches it,
because a felled tree is a surface edit already.

### 6.7 Should a city be its own realm?

**Not now, and nothing needs reserving to keep the door open.**

- **The ground case is currently forbidden by construction.** Promoting a ground-built structure to its
  own realm *"would be a re-home of a voxel volume across a grid-mapping boundary (Spherical → Identity),
  which `satisfies()`'s exact geometry equality forbids by construction. Ground-built structures are part
  of the planet grid, permanently."*
- **The floating case is already free.** A construction anchor mints a Cartesian construction realm with
  its own store, its own physics authority and its own demand-driven spin-up, today, at zero cost.
- **A ground city would need a new object** — an `Area` realm (the arm already exists in `RealmKindTag`)
  owning a *sub-range* of the parent body's spherical chunk-key space rather than an identity grid. Its
  three real costs: a realm seam across visible ground with no crack and no double-collision; constant
  re-home flapping for players walking the perimeter (the exact failure already root-caused in the
  moving-frame crossing work); and an ambiguous containment answer for a block straddling the boundary.
- **The pressing half of the motivation needs none of it.** Budget isolation is obtainable by making the
  per-realm delta budget hierarchical with per-claim sub-budgets — no realm-model change, no new grid.
- **The door stays open for free.** Because `ChunkKey` orders sector → tier → a → b → c, extracting a
  spatial district later is one contiguous range scan per `a` value per rung — about 28 scans per rung
  for a 1.7 km city. Promotion stays a **store split**, never a format migration.

### 6.8 The megastructure re-escalates the vertex-format row

The detail-ladder addendum de-escalated the vertex format out of the blocking set on **natural-terrain**
arithmetic. A city changes that arithmetic. A city chunk presents ~26,500 exposed faces; greedy merging
on flat walls and floor slabs is very effective (~8× on a 30 × 12 m wall), but the material-seam and
shaped-neighbour peels fire heavily where stone, wood, glass and doors meet. Call it ~8,000 quads/chunk,
1.6× the design's conservative allowance:

| Vertex format | 505 city chunks in the tier-0 disc | Design's budget for an entire 100 km view |
|---|---|---|
| Today's 24-byte `MeshPrim` (V1, 88 B/quad) | **355 MB** | 3.27 GB |
| Packed 8-byte-per-quad (V2) | **32 MB** | 273 MB conservative / 120 MB measured |

**At today's format a city's near field alone is 1.3× the whole conservative whole-view budget.** The
city, not the terrain, is what decides this row, and it must be settled **before a city exists**, because
it is a mesh-encoder plus upload-path change, not a data migration.

---

## 7. Repetition and instancing, ranked by benefit per unit of complexity

### 7.1 The ranking

| # | Technique | Benefit | Complexity | New dependency? | Verdict |
|---|---|---|---|---|---|
| **1** | **Greedy 3-D box encoding** as a fourth `DeltaEncoding` arm | **244× on a hollow building, 200× on a solid cube, 25,000× on a mined-out room** | ~250 lines of integer code + a canonicality proptest | none | **ADOPT** |
| 2 | **Blueprint stamps** — a durable 64-byte reference instead of the cells | 244× on the first copy, unbounded on repeats; the same on the wire | a new durable table + a third level in the read path + a collapse rule | none | **worth a decision** |
| 3 | **Lattice-snapping** stamps to 2ᵏ metres | multiplies (2) by removing 98 % of the pyramid | a build-UI constraint | none | only with (2) |
| 4 | **1-D run encoding** | **dominated 92× by (1)** | same as (1) | none | **SKIP** |
| 5 | **A byte codec** (deflate/zstd) over the encoded record | 2–3× on disk, ~1.05× on packed indices | a library adoption | **yes** | **§9.5 / owner's call** |
| 6 | **Shared-subtree graph (SVDAG)** | up to ~32× in the literature — on **occupancy only** | a research problem | probably | **REJECT for the live world** |

### 7.2 Boxes — the top recommendation, and why runs lose

**Record:** 16 bytes — one `u64` holding `anchor: CellIndex(18) | extent_u(6) | extent_v(6) |
extent_w(6) | reserved(28)`, plus the existing 8-byte state word **verbatim** as the second word. A
proper new type reusing the frozen payload, not a reinterpretation of reserved bits — which is what the
standing "never hack/repurpose, add proper new types" law requires, and it means R1's fourteen reserved
bits stay untouched.

**Crossover, exact and trivially stated:** boxes beat sparse when `16b < 8n`, i.e. **whenever the mean
box volume exceeds 2.0 cells**. Every axis-aligned player build satisfies that by 30–250×.

**The head-to-head, on the same hollow building** — 20 × 20 × 30 m, one-cell shell, 2,928 authored cells:

| Encoding | Bytes | Against sparse |
|---|---|---|
| Sparse `(cell, state)` | 23,424 | 1× |
| Masked-dense (16-entry palette) | 31,319 | 0.75× (loses — 2,928 is far below the crossover) |
| **1-D runs** along the fastest axis | **8,832** (1,104 runs) | 2.65× |
| **3-D boxes** | **96** (6 boxes) | **244×** |

**Boxes beat runs 92× on this shape, and 200× on a solid 20³ cube (16 B against 3,200 B).** The reason is
structural: half of any axis-aligned building's walls are perpendicular to whichever single axis a run
encoding picks, and those walls degenerate to 1,008 length-one runs. **Ship boxes; do not ship runs as a
stepping stone.**

Four further properties:

- **It can never lose.** The compactor already computes every candidate's size and takes the smallest, so
  a fourth arm is a strict improvement with a proven floor. Worst case — a 3-D checkerboard — is 2× sparse
  and the compactor simply does not pick it.
- **It survives chunk straddling with no cliff.** A building split across four chunks becomes ~10 boxes
  instead of 6, a 1.7× penalty, against masked-dense's 4 × 29,791 B fixed-mask penalty.
- **It applies unchanged to the pyramid**, where a uniform building interior gives every rung-1 parent
  the identical summary, so coarse city records box as well as tier-0 ones do.
- **It gets exactly zero on the scattered-edit case of §9.2**, which is the largest storage number in
  this whole analysis. A scattered edit is a 1 × 1 × 1 box.

**Determinism is the one thing that must be asserted rather than assumed.** A greedy 3-D cover has many
valid outputs for the same input — sweeping x-then-z-then-y differs from z-then-x-then-y. Fix the axis
priority, fix the sweep order (ascending cell index), grow each box maximally in the priority order,
emit in ascending anchor order; then prove it with a decode/re-encode byte-equality proptest **and** an
edit-order-permutation proptest. Without that, two hosts produce different bytes for the identical world
and every content hash and byte-identity gate downstream silently diverges.

### 7.3 Blueprint stamps

A **blueprint** is an immutable, content-addressed, TLV-framed record holding a bounding box, an origin
cell and the authored-cell set in the *same* encodings a chunk record uses — zero new format machinery,
because the TLV envelope already guarantees canonical ascending-tag bytes its own doc comment calls
content-hash-dedup safe. A **stamp** is a durable row: `{ blueprint: BlueprintId, anchor: BlockAddr,
orient: u8, fence: u64, epoch }` ≈ 64 bytes. The read path becomes recipe → stamps → per-cell overrides.

| Case | As cells | As stamps | Ratio |
|---|---|---|---|
| One hollow building | 23,424 B | 64 B + one blueprint (96 B boxed) | 244× / unbounded on repeats |
| 2,400 identical buildings | 56.2 MB | 153.6 KB + one blueprint | **366×** |
| A shipyard of 500 identical hulls (50,000 cells each) | 200 MB | 32 KB + one blueprint (400 KB sparse) | **463×** (5,000× if the blueprint is itself boxed) |

**Divergence is bounded, monotone and cheap.** Per-cell overrides take precedence at the ordinary 8 B
each, so a building with 20 changed cells is 64 + 160 = 224 B against 23,424 B. The adversarial case — a
war levels the city and every cell is overridden — degrades to exactly the non-instanced cost **plus 64
bytes per stamp**, an overhead of **0.27 %**. That monotone bound is what makes instancing safe to adopt:
it cannot lose. Two rules must be stated as requirements, not left to the implementer: stamps are ordered by
`(fence, stamp_id)` with later winning, and per-cell overrides always beat stamps.

**Instancing is worth more for ships and stations than for planets**, because they have no generated
baseline at all — every cell of a ship is delta today, with nothing prunable.

**Lattice snapping is the half nobody would think of.** A blueprint can carry its own precomputed
pyramid, because coarsening is a pure function of eight children and the contents are fixed — but its
internal groupings only line up with the world's coarse lattice when the anchor is congruent to 0 mod 2ᴸ.
Snapping stamps to 2ᵏ metres at build time makes rungs 1..k free:

| Snap lattice | Pyramid stored | Of the unsnapped total |
|---|---|---|
| 1 m (no snapping) | 100 % | — |
| 2 m | 25.0 % | |
| 4 m | 6.2 % | |
| **8 m (recommended)** | **1.6 %** | |
| 16 m | 0.4 % | |

Without snapping, instancing fixes the delta and leaves the pyramid as the entire remaining cost.
Snapping is relaxable later — an unsnapped stamp simply stores its own rungs — so it is **not** a one-way
door; *tightening* it later would be.

**The composite recogniser must not become a storage idea.** There is a one-sentence rule that keeps the
two lanes apart: **a stored reference may record what the PLAYER DID; it may never record what the SERVER
INFERRED.** A stamp is durable because it records an act, immutable by construction, with its expansion
fixed forever by a content hash. A recognised composite (owner ruling R4) is a *conclusion* about stored
cells, and storing the conclusion fails three ways: the recogniser's rule becomes part of the durable
format, so improving it is a migration over every saved world; breaking one block under a recognised form
has no representation; and re-deriving the cells means running the recogniser. The legitimate convergence
is on the read and draw side and should be reserved now — a stamp and a composite are both "one id + one
anchor + one orientation → many cells", so make the per-chunk instance index generic over
`InstanceKind { Stamp, Composite }` in the same commit, or R4 will build a second one and that is an HR3
violation.

### 7.4 SVDAG — verdict, with the reason that is not the obvious one

**Superb for static archival, structurally wrong for this live world — and the decisive reason is not the
edit cost.**

A shared-subtree graph hash-conses identical subtrees into one node, so a node is referenced from
everywhere in the world that contains the same shape. That is the source of both its compression and its
unsuitability. **You cannot ship "the delta for chunk K" without the shared nodes it references, and
those nodes are shared with chunks the receiver does not have** — so the transmissible unit becomes the
graph from the root, i.e. the whole city, which makes streaming *worse*. Everything the design builds on
per-chunk independence breaks with it: the delta manifest at 8 bytes per key, coarse-first ordering, the
per-(tier, chunk) storage block, and the key ordering that makes "every tier-3 record on face 2" one
contiguous range scan. There is also no analogue of the prune rule, because a graph node has no address
whose generator baseline you can compare against.

Two further incompatibilities are pure arithmetic. 62³ is not a power of two, so an octree inside the
committed chunk needs padding to 64³ — a **10.0 %** cell overhead before any node overhead. And the tree
over a cube face is deep: 62 × 4,096 = 253,952 cells per face edge, so a point lookup is **18 dependent
pointer chases against exactly one array index** in the current container.

**What the literature actually reports, and why it does not transfer.** The founding result (Kämpe,
Sintorn & Assarsson, SIGGRAPH 2013) reports up to roughly 32× against an optimally encoded sparse voxel
octree, strongly content-dependent — architectural and repetitive scenes near the top, organic geometry
near 1.2–2×; symmetry-aware matching (Villanueva et al., I3D 2016) adds roughly 1.5–3× more. The caveat
the literature repeats is the one that decides our case: **the graph compresses occupancy only.**
Per-voxel attributes do not deduplicate, because two subtrees of identical shape almost never carry
identical materials — which is why a whole follow-up line of work exists solely to compress voxel colour
separately. Our cell is not one bit; it is an 8-byte record. And editing such a graph is a research
problem in its own right, whose published answer is a GPU hash table with copy-on-write plus periodic
garbage collection and re-deduplication — the worst possible fit for a determinism requirement where
allocation and iteration order must never leak into output bytes.

> **Confidence note on the multipliers:** the 32× and 1.5–3× figures are recalled from the literature and
> were **not** re-verified in this session. The structural conclusions do not depend on them; the numbers
> should be re-checked before they appear in any owner-facing comparison.

**Keep it on the shelf for exactly one job:** the at-rest encoding of a *blueprint's* contents, where the
data is immutable, written once and read many times. That composes with stamps rather than competing with
them, and it is entirely contained behind the blueprint's own encoding tag. It must **never** be
considered for the chunk delta or the pyramid.

---

## 8. Combined effect

| Configuration | 10 M-block city, delta | Pyramid | Total | B/cell |
|---|---|---|---|---|
| Naive sparse, flat-8 B pyramid | 80.0 MB | 26.7 MB | 106.7 MB | 10.67 |
| **The committed design as written** | 30.2 MB | 11.1 MB | **41.3 MB** | **4.13** |
| + box encoding | ~3–12 MB | ~2–5 MB | **~5–17 MB** | 0.5–1.7 |
| + stamps, 8 m lattice (2,400 repeated buildings) | ~0.2 MB | ~0.2 MB | **~0.4 MB** | 0.04 |

**The committed design is already 2.6× ahead of naive. Boxes take it to 6–20× for two hundred and fifty
lines and no dependency. Stamps take a *repetitive* city to 100×+ but cost a new durable concept.**

---

## 9. What is genuinely missing from the committed design

### 9.1 A one-line defect in the pyramid walk-up — fix it now, it is free

The published loop reads:

```
stored = chunk_pyramid.get(parent).map(PyramidEntry::summary)
if Some(next) == stored { continue }                   // EXIT ON SUMMARY EQUALITY
if next == generate_summary(parent) { chunk_pyramid.remove(parent) }   // PRUNE
else { chunk_pyramid.put(parent, PyramidEntry::new(parent.cell, next)) }
dirty_coarse_chunk(parent.chunk, L)
mark parent.ancestor(L+1) dirty
```

**For an unedited parent, `stored` is `None`, so `Some(next) == None` is always false and the exit can
never fire.** Control falls through, calls `remove` on a key that does not exist, calls
`dirty_coarse_chunk`, and marks the ancestor dirty — **at every rung, on every edit.**

Traced concretely. A player mines one granite cell deep inside rock:

| Rung | Summary | Generator's summary | Should exit? | Does exit? |
|---|---|---|---|---|
| 1 | occ 0xFE, fill 223 | occ 0xFF, fill 255 | no — write | no ✓ |
| 2 | occ 0xFF, fill 251 | occ 0xFF, fill 255 | no — write | no ✓ |
| **3** | occ 0xFF, **fill 255** | occ 0xFF, fill 255 | **YES** | **NO ✗** |
| 4–12 | identical to the generator at every rung | | **YES** | **NO ✗** |

The design's own claim that *"the common single edit inside solid rock terminates at rung 1 or 2
(~0.5 µs)"* is **false as written**. The common case runs the full walk and marks a coarse chunk dirty at
**ten of the twelve rungs** at the starter body, **eighteen of twenty** at Earth scale.

**The CPU cost of the walk itself is small** — a no-op remove plus a set insert. **The cost that matters
is downstream:** a coarse chunk is announced as changed when it did not change, so every observer
resident at that rung re-fetches and re-meshes a byte-identical record. At ~65–95 µs per coarse mesh and
ten spurious rungs, that is up to **~0.9 ms of client mesher work per edit** on distant clients, plus the
fan-out messages to carry it — and in a city at 200 edits/s that is a real fraction of a core producing
byte-identical meshes. *(The exact downstream cost depends on what `dirty_coarse_chunk` is wired to,
which the design does not state; the loop's behaviour itself is verified from the text.)*

**The fix is one line, and it uses a function the design already defines two paragraphs earlier:**

```
gen     = generate_summary(parent)
current = chunk_pyramid.get(parent).map(PyramidEntry::summary).unwrap_or(gen)   // == coarse_summary
if next == current { continue }                        // the EFFECTIVE summary is unchanged
if next == gen { chunk_pyramid.remove(parent) } else { chunk_pyramid.put(parent, …) }
dirty_coarse_chunk(parent.chunk, L)
mark parent.ancestor(L+1) dirty
```

The exactness argument survives verbatim: what the read path returns for the parent is unchanged, so
every ancestor is unchanged. The prune-when-present case is preserved. **Free before the code exists.**

### 9.2 The pyramid's cost table has no row for planetary scatter, and that row is 5.5× the plan

Entries per edit is not a constant. For *n* edits spread over surface area *A*, it is

> `entries = Σ_{L=1}^{tier_depth−1} min(n, A / 4^L)` — an edit is isolated at every rung whose cell is
> smaller than the mean edit spacing, and clustered above it.

The starter body's surface is 4πR² at R = 161,671 m = **3.285 × 10¹¹ m² = 328,454 km²**:

| Edits | Mean spacing | Entries | Per edit | Pyramid | Delta |
|---|---|---|---|---|---|
| 1 M | 573 m | 9,411,125 | **9.41** | 75 MB | 8 MB |
| 10 M | 181 m | 76,675,881 | **7.67** | 613 MB | 80 MB |
| **100 M** | **57 m** | **606,911,985** | **6.07** | **4.86 GB** | **0.80 GB** |

**4.86 GB against a planned 880 MB — 5.5× — and 2.9× the entire stated 1.7 GB budget.** The spacing that
produces it is one edit per 57 metres, which is not adversarial: it is ten thousand players each leaving
ten thousand marks, torches, sample digs and prospect holes across a world they are exploring.

The four-row table has no row for this geography, and the "isolated" row it *does* have is not a scaling
threat at all: to be isolated at every rung an edit must be 4,096 m from its nearest neighbour, and only
**19,577** such sites exist on the whole starter body — **1.9 MB in total**. **The design bounds the case
that cannot happen and omits the case that will.**

Two mitigations, and they are partial rather than complete. Fixing §9.1 helps directly, because the fill
fold `(Σ + 4) >> 3` means a single 1 m cube contributes 32 at rung 1, 4 at rung 2, 1 at rung 3 and **0 at
rung 4** — so a block placed *on solid ground* stops differing from the generator by rung 4 and prunes,
giving ~4 entries/edit ≈ 3.2 GB. A block placed in mid-air changes the octant mask at every rung and gets
the full twelve. And **no compression idea in this document touches it**, because a scattered edit is
already a single box and a single run.

### 9.3 `column_sky` — the widest record, with no rule that bounds it

`[u16; 62²]` = **7,688 bytes per chunk column**, reserved with the justification *"every catch-up
integrator; sky exposure spans 323 chunks vertically"*, and never mentioned again.

Every other persisted structure has an explicit rule that bounds it: a chunk delta holds only authored
cells; a pyramid entry is deleted the moment it equals the generator's own summary; the relief envelope
was deliberately *struck* as a persisted table and made a pure function of seed plus pyramid.
**`column_sky` has no sparsity rule, no prune rule, no derivation clause, and no place in the per-realm
byte budget**, which counts the delta and the pyramid and nothing else.

| Materialisation | Columns | Bytes |
|---|---|---|
| The whole starter body (6 × 4096²) | 100,663,296 | **774 GB** |
| One percent of the surface ever visited | 1,006,633 | **7.74 GB** (4.5× the whole budget) |
| Only the ~60,000 columns under 200,000 edited chunks | 60,000 | **461 MB** (27 % of the budget) |

**The fix is free and already precedented:** sky exposure is a pure function of the block field, so the
pyramid's own rule applies verbatim — *an entry exists iff it differs from what the generator would
produce*. Somebody just has to write it, and it is delta data, therefore permanent, therefore before the
format freezes.

### 9.4 Persisted weather bits — the same hole, written by the weather rather than by players

`snow_depth` and `grass_cover` are marked persisted. Unlike edits, which are bounded by the per-session
rate cap, these are authored **over whole surfaces by the weather**:

> One snowfall across one player's 786 m tier-0 residency disc sets state on π × 786² = **1,940,595
> surface cells at 11 B = 21.4 MB**, on a checkpoint-carried table flushed at ~1 Hz — **21 MB/s while
> snow is falling, for zero player actions.**

There is no upper bound stated and no prune rule, even though the design itself already names the
baseline that would make one trivial: the snow *quantity* takes its coarse rung *"from the coarse
weather/biome field the generator already evaluates at that tier."* That is exactly a generator baseline
to prune against. The deforestation finding made precisely this argument for trees and it was accepted;
it has not been made for snow.

### 9.5 A byte codec — the one columnar technique genuinely absent

**Ground truth, checked directly against `Cargo.toml` and `Cargo.lock`:**

| Crate | In the lock? | Direct workspace dependency? | Can it compress? |
|---|---|---|---|
| `flate2` 1.1.9 + `miniz_oxide` 0.8.9 | yes | **no** — transitive via `png`/`tiff` under `image` | yes |
| `ruzstd` 0.8.3 | yes | **no** — transitive via `bevy_image`'s KTX2 path | **NO — decoder only** |
| `yazi` | yes | no — transitive via `gltf` | — |
| `blake3` | yes | no — transitive via `bevy_asset`, **not reachable from `vd-core`** | — |
| `zstd`, `lz4`, `lz4_flex`, `snap`, `brotli`, `bitpacking`, `roaring`, `bitvec`, `arrow`, `parquet` | **absent entirely** | — | — |
| `sha2` 0.10 | yes | **yes, direct** | — |

`redb` applies no page compression — it is a plain B-tree — so every byte the design computes is a byte
on disk. **"zstd is already in the lock file" would be a false economy: what is there cannot compress.**

**Where a codec would pay, and where it would not.** The value is concentrated in the fixed
29,791-byte mask, which is a sparse, clustered, tree-shaped bitmap — exactly the input a general codec
crushes — and it is worthless on the packed palette indices beside it:

| Record | Mask share | Expected codec gain |
|---|---|---|
| A clear-cut forest chunk (6,221 authored cells, 31,362 B masked-dense) | **95.0 %** | high — the record is almost entirely mask |
| A city chunk (13,255 authored cells, 39,989 B) | **74.5 %** | 1.8–2.5× net |
| The packed state array beside it (~200 distinct triples, ~7.6 bits of entropy per byte) | — | **~1.05× — worthless** |

**The seam is already planted and the design never cites it.** `crates/core/src/tlv.rs` line 46 pins
`CODEC_FLAGS_V1: u8 = 0`, documented verbatim as *"postcard, no compression. Any other value is from the
future and must be rejected, not guessed at"*, and line 193 hard-rejects it with
`TlvError::UnsupportedCodec`. So adding compression is a **codec value on an envelope that already exists
and already refuses unknown values** — a clean version gate, not a format migration.

**Recommendation: decide after boxes land, not before.** Boxes remove most of the redundancy for free;
what remains is high-entropy packed indices at 1.05×. See §12, D-6.

### 9.6 Column separation is only partially banked

`block_state` is `(u32 local, u16 damage_dp, u16 function_dp, u16 state_bits, u8 growth)` = 11 bytes —
**four logically separate columns in one row**, with nothing in common:

| Field | Changes | Lifetime |
|---|---|---|
| `damage_dp` | every tick a block is being mined | discarded when it breaks |
| `function_dp` | machine state (P9) | per tick |
| `state_bits` | grass / leaf / snow / wetness | driven by the *weather*, over whole surfaces |
| `growth` | a planted thing's stage | days |

Row-orienting them means a damage tick rewrites the growth byte, and a block carrying only a growth stage
pays 11 bytes where 5 would do. Priced on the case that dominates the table — one snowfall over one 786 m
disc — a dedicated `(u32 local, u16 state_bits)` table costs **11.6 MB against 21.4 MB, a 1.84× saving on
a write no player triggered.**

The design is not naive about this: it explicitly splits the **in-memory** form into parallel key and
value vectors for exactly the columnar reason, then states the **persisted** form as a row. That is the
insight applied in memory and dropped on disk, which is backwards, because disk is where the compression
and the read amplification both live.

### 9.7 The pyramid's crossover constant is wrong across a wide band

Both §2.4.3 and §3.9.8 state that the pyramid uses *"the same 4,256-cell crossover"* as the chunk delta.
It does not, and the correct answer is not a single number either — it depends on how many distinct
`(substance, occ_mask, fill)` triples a coarse chunk holds, because the masked-dense form palettises
them:

| Distinct triples in the record | Index bits | Masked-dense first wins at |
|---|---|---|
| 1 (a uniform building interior) | 0 | **n ≥ 3,725** |
| 256 (a mixed built region) | 8 | n ≥ 4,403 |
| 4,096 | 12 | n ≥ 7,104 |
| all distinct (a varied natural surface — `fill_256` takes 256 values) | ~14 | **n ≥ 13,250** |

**Hard-coding 4,256 costs up to 19,683 bytes per record.** At n = 4,256 with an all-distinct palette,
sparse is 34,048 B while masked-dense is 29,791 + 17,024 + 6,916 = **53,731 B — 1.58× WORSE**, on exactly
the varied-surface chunks the encoding exists to help. Two circulating review figures (4,584 and 7,448)
contradict each other and are both special cases; neither should be quoted.

**The design's own prose already prescribes the correct behaviour** — *"the compactor picks whichever is
smaller — computed exactly, not against a guessed threshold"* — so the risk is purely that the constant
`4,256` appears in code. It must not.

---

## 10. The edit path's read and write amplification, traced

### 10.1 Read amplification on chunk load

| Operation | Cost |
|---|---|
| Generation | **0.885 ms** measured per 62³ surface chunk (62 × 62 columns × 8 noise evaluations at 12.19 ns, plus the coarse-lattice cave field) |
| Point reads | 1 delta + 5 side tables + `chunk_meta` = **7** |
| `column_sky` | 1 read of 7,688 B per column |
| **The apron nobody counts** | Meshing needs a one-block apron of *full* records, and a full record is generator ∪ delta — so meshing chunk C must know whether any of its **26** neighbours has an authored cell in the shared boundary plane: **26 additional point reads per chunk meshed** |

**Cold-filling the 1,515 tier-0 chunks of a 786 m disc is therefore ~51,500 point reads plus 1.34 s of
generation on one core (0.17 s on eight).** In steady walking motion the aprons overlap the resident set,
so the marginal cost collapses to the outer ring — which is why this is a cold-start cost, not a hot-path
one. But it is the number that decides how long a warp arrival into a built system takes, and it appears
nowhere in the design.

The related unbounded operation is `vdctl world verify-pyramid <realm>`, which the design requires the
test suite to run after **every** edit scenario. On a realm with 110 M pyramid entries that is
110 × 10⁶ × 8 × 61 ns ≈ **54 s of pure CPU** plus a read of the whole 880 MB pyramid, with no ranged or
incremental form specified. Production is fine — it is bounded per checkpoint. The debug and test path is not.

### 10.2 Write amplification on the edit path

`redb` is a copy-on-write B-tree and the shipped store does whole-value `table.insert` at
`crates/io-prod/src/store.rs:350`, so **changing one 8-byte pyramid entry rewrites its entire per-(tier,
chunk) storage block.**

**And in a built region the record size is scale-invariant up the ladder.** From §6.2: rung 1 is 189
chunks × 13,255 entries, rung 2 is 47 × 13,255 — the entry count and the chunk count *both* fall by four,
so every rung's record stays near **44,070 bytes**.

| Quantity | Value |
|---|---|
| Bytes rewritten for one 8-byte change | **44,062** = **5,508×** |
| One transaction, 100 builders at 2 edits/s, 250 ms fsync interval: 10 tier-0 chunks + 3 changed rungs | 10 × 39,989 + 30 × 44,070 = **1.72 MB** |
| Physical write rate | **6.9 MB/s** |
| Logical player intent | 200 × 8 = **1.6 KB/s** |
| **Aggregate amplification** | **≈ 4,300×** |

Survivable on an SSD, and **the bound comes from the fsync cadence rather than from any deliberate design
decision** — which is the part worth stating, because a cadence change would silently multiply it.

**Three cheap remedies, and the design already contains the argument for two of them:**

1. **Fix §9.1.** Fewer rungs touched per batch, directly.
2. **Move the walk-up out of the WAL's own write transaction into compaction**, with an in-memory pyramid
   absorbing repeated updates. The design puts it inside for crash consistency it explicitly does not
   need: it argues *twice* that the pyramid is 100 % reconstructible from the deltas beside it and that a
   divergence must **not** refuse to open the store.
3. **Make the pyramid's storage-block granularity a tuning field** (`pyramid_storage_block_cells`), so a
   dense chunk splits into sub-blocks. Free now, a persisted-key migration afterwards.

### 10.3 Compaction

Compaction folds WAL frames into the chunk delta, and a chunk record is one redb value, so a chunk that
received *one* edit in the interval has its **entire** record rewritten — 39,989 B for a city chunk.
Compacting an actively-built city dirties ~10 tier-0 chunks and ~20 pyramid blocks per checkpoint
≈ 1.3 MB, sub-millisecond, and even a full rewrite of the city's 41.3 MB is 0.14 s at 300 MB/s. **So
compaction never becomes the constraint at this scale.** The failure mode to watch is *breadth*, not
depth: if a population spreads across thousands of distinct chunks per checkpoint interval the whole-record
rewrite becomes the wall, and how many distinct chunks a real population touches has never been measured.
That should be a load fixture before it is designed around.

Unlike the pyramid, this cannot be fixed by moving the transaction — the delta is authored data that
cannot be recomputed. The standard fix that fits is a log-structured chunk record keyed
`(chunk, generation)` with a small append per checkpoint and a fold only when appended bytes exceed a
fraction of the base, converting O(record) rewrites into O(edits) at the cost of 2–4 point reads on load.

### 10.4 Spin-up, and the one shape that is wrong

**Spin-up is O(1) in city size, and that is already true by construction.** Opening a realm reads the
descriptor, checks the owner fence, and stops; every chunk read after that is demand-driven by the same
predicate as realm spin-up. The one quantity that *would* have been proportional is already reserved
away: `realm_pcu` is persisted in the realm metadata row precisely so *"the demand scheduler wants a
realm's cost before spin-up; without a durable sum, every re-home and spin-up rescans the world."* And
because keys order sector → tier → a → b → c, the arriving player's first draw is one contiguous range
scan of the coarsest rung — 5 KB, one record, and the city is on screen.

**The dormant-advance fold is the exception.** It is O(1) in elapsed *span* but **unbounded in write
volume**. A city's 2.9 km² of cleared ground is 1.49 clear-cut discs ≈ **53 MB of records that a year of
natural recovery would delete** — and deleting 53 MB inside one spin-up write transaction while a player
waits at the door is the wrong shape. `max_advance_span` bounds the span, not the result. The fix is
already reserved and just not stated: `chunk_meta.last_ticked_universe_tick` exists per chunk *"because
every catch-up integrator needs it"*, so the advance can be evaluated **per chunk on first read after
spin-up** rather than realm-wide at spin-up — amortising it over demand at zero extra storage.

The underlying property is good news, and worth saying: a cold city shrinks its own footprint where
nobody maintains it, while `Placed` provenance means the buildings themselves never prune.

### 10.5 The fan-out, which the city makes a hard blocker

D-9 measured 128 sessions in one realm producing 357,248 gateway messages, peaking at 1,920 per tick over
7.6 s, with no within-realm per-entity area-of-interest. The design already rules that the per-cell
reshape is *"a precondition of block edits, not a follow-on."* **The city is what makes it a hard
blocker**, and by message count rather than by bytes:

> 100 builders at 2 blocks/s = 200 edits/s, fanned out to all 100 sessions under whole-realm broadcast =
> **20,000 `ChunkDelta` messages/s = 1,000 per tick at 20 Hz** — squarely in D-9's measured pathological
> band, on an aggregate of only ~800 KB/s.

The uncomfortable part is that per-cell area-of-interest helps *less* here than elsewhere: the radius is
786 m and the city is 1.7 km across, so most builders are inside most other builders' cells and the saving
is nearer 2× than 100×. **The fix the city actually needs is coalescing, and the machinery exists one
layer down** — the WAL is already dedup-to-final, sorted-on-drain per batch — so the fan-out should ship
**one `ChunkDelta` per (chunk, tick) carrying the batch**, not one per edit. That collapses 20,000
messages/s to roughly one per touched chunk per tick per interested session.

---

## 11. Hard-rule conformance — what every proposal here must respect

No recommendation in this document breaks determinism, sealed shards, the one-writer rule, the pure
renderer, the no-magic-numbers rule or the coverage floor. Six specific traps must nonetheless be stated,
because each is a way of breaking one of them by accident.

1. **Determinism of a greedy cover.** A box decomposition has many valid outputs for one input.
   Canonicality must be **asserted** — fixed axis priority, ascending cell-index sweep, maximal growth in
   the priority order, ascending anchor emission — with a decode/re-encode byte-equality proptest **and**
   an edit-order-permutation proptest. The WAL is already required to produce deterministic bytes on a
   sorted drain and chunk-record bytes are required to be content-hash-dedup safe, so this is not
   theoretical.
2. **Compression sits strictly BELOW every content hash.** The TLV envelope guarantees canonical
   ascending-tag bytes so blobs are dedup-safe. Hashing *compressed* bytes would make that guarantee
   depend on a third-party library version, and a routine `cargo update` would invalidate every dedup key
   in the game **with no compile error**. Hash the plaintext; record the codec id and level in the header;
   never let a byte-identity gate see compressed bytes.
3. **The TLV field cap.** `MAX_FIELD_BYTES = 1 << 20` caps one field at 131,072 eight-byte records. A
   fully-authored masked-dense chunk record is 268,119 B and safe; **a 500,000-cell blueprint at 4.0 MB is
   not** and must be split across fields or tags. This is a blueprint-only hazard and will not surface
   until someone saves a large build.
4. **Evolution class.** If stamps land, their tag must sit **above `required_max_tag`** and be refused
   with the existing `MissingRequiredTag`, never registered as an ordinary skip-unknown tag. An older
   client would otherwise render open ground where a city stands and then walk through walls the server
   says are solid — a terrain **fact** divergence, which the ladder's own fact/detail law forbids
   outright. The box arm is safe for free, because `DeltaEncoding` is an explicit enum field and an
   unknown discriminant is already a hard decode failure.
5. **HR5 monomorphisation.** Codec crates are generic over readers and writers, which is the exact
   coverage foot-gun `CLAUDE.md` documents. Any codec call must be wrapped behind a **single monomorphic,
   non-generic function**, or every `?` and every error closure inside the generic wrapper becomes an
   uncovered region per instantiation. The same discipline applies to the encoding-selection function:
   `pick_smallest(sparse, masked_dense, boxes)` must be a branchless shim over monomorphic helpers, and
   every arm must be exercised in every test binary that instantiates it.
6. **Inter-shard bytes.** The design's current claim that *"the ladder adds zero inter-shard bytes"* is
   correct and worth preserving. It breaks the moment a **stamp** can originate in a ghost or overlap
   region, because the forward must then carry the blueprint id, anchor and orientation, and the owner may
   not hold the blueprint — needing a `WantBlueprint` negative acknowledgement and a second round trip.
   Because postcard is positional and the flow taxonomy is closed behind four structural conformance
   gates, **reserving that field is free today and a wire-version event afterwards.** If it is not
   reserved, the alternative is that a player's stamp silently costs thousands of forwarded records
   depending on which side of an invisible boundary he is standing on — exactly the position-dependent
   behaviour the seamless law objects to.

**And one rule that costs nothing and prevents a whole class of mistake:** *a stored reference may record
what the PLAYER DID; it may never record what the SERVER INFERRED* (§7.3). It is what keeps blueprint
stamps durable and keeps the composite recogniser out of the storage format.

**Load and performance gates land with the subsystem**, per the standing rule. The paired fixtures write
themselves: stamp 2,400 buildings and assert store bytes, wire bytes, chunk materialisation time and
walk-up cost against §6.2; stream 100 M scattered edits and assert the pyramid against §9.2's row; run
one snowfall over one residency disc and assert the pruned `block_state` table is empty.

---

## 12. The decision register

Ordered by cost of lateness. Each row states the recommendation, the cost of deferring, and whether it is
a one-way door.

| # | Decision | Recommendation | Cost of deferring | One-way door? |
|---|---|---|---|---|
| **D-1** | **Fix the pyramid walk-up exit condition** (§9.1) — compare against the effective summary (`stored` or generated), not against `Some(next) == stored` | **Do it.** One line. It is barely a decision | Free now. After the loop exists: an audit of a hot path, plus every performance figure taken against the wrong behaviour | No |
| **D-2** | **Prune rule for `column_sky`, `snow_depth`, `grass_cover`** (§9.3, §9.4) — an entry exists iff it differs from what the generator would produce | **Adopt.** The pyramid's rule, applied verbatim | Free now. Afterwards: permanent delta data, so a migration over every saved world — and a store bounded between 461 MB and 7.7 GB by an unwritten policy in the meantime | **YES** |
| **D-3** | **Rule surface-only OUT explicitly** (§5.6) — the three legitimate forms survive by name; "store a shell instead of a volume" is rejected | **Adopt.** State it in the design so it does not return | Free. Without it the idea returns every time someone reads the storage numbers | No |
| **D-4** | **Greedy 3-D box encoding as a fourth `DeltaEncoding` arm** (§7.2) — 16 B/box, applied to the delta, the pyramid and blueprint contents alike. **Do NOT ship 1-D runs** | **Adopt in the P6 edit slice.** ~250 lines + a canonicality proptest; the compactor's "smallest wins" means it can never lose | Free for the *mechanism*. The 16-byte record layout is permanent once a world is saved, and adding the decoder after ship is a `PROTO_MINOR` bump plus a compatibility window | **YES for the record layout** |
| **D-5** | **Reserve the codec tag now** (§9.5) — `crates/core/src/tlv.rs` already pins `CODEC_FLAGS_V1 = 0` and rejects anything else. On the wire, add an orthogonal `codec` **field** beside `encoding` (never extra `DeltaEncoding` arms — codec × encoding is a product), always `Codec::None` today | **Adopt.** Free | Afterwards: postcard is positional, so it is a `PROTO_MINOR` bump **plus** a stored-format migration over every saved world | **YES** |
| **D-6** | **Compression library** (§9.5). (A) **none — recommended for now**; (B) promote `flate2` to a direct dependency — **zero new crates in the tree** (already transitive via `png`/`tiff`), pure-Rust backend, ~2–3× at 60–100 MB/s; (C) `zstd` — best ratio (~3–4×) at ~500 MB/s, **but `zstd-sys` adds a C toolchain to the k3d images, the stable product build AND the pinned coverage nightly**; (D) hand-rolled RLE — no dependency, but new permanent format surface that does the job worse than any of (B)/(C) | **(A) now; re-measure after D-4 lands.** Boxes remove most of the redundancy; what remains is high-entropy packed indices at ~1.05× | Deferring costs nothing **provided D-5 is taken** | No, if D-5 is taken |
| **D-7** | **Add the planetary-scatter row to the pyramid cost table and re-derive the budget** (§9.2) — `log₄(A / n) + ⅓` entries per edit, which is 6.07 at 100 M edits on the starter body. Two sub-questions: (a) is the shipped budget default sized against 1.1 entries/edit or against 4–6? (b) does the budget's failure stay realm-wide (today: a typed refusal of **all** edits on the planet, so one megaproject bricks editing for everyone) or become hierarchical per-claim sub-budgets? | **(a) size against 4. (b) hierarchical** — it needs no realm-model change and no new grid | Free to state now. Discovered later as an outage on a live world | No |
| **D-8** | **Blueprint stamps** (§7.3) — (A) **lazy**: a 64-byte durable reference, cells never written, read path recipe → stamps → overrides; (B) **eager**: stamping writes all n cells, the blueprint is a build-UI convenience only | **Worth taking, but it is genuinely yours.** (A) buys 244–463× and is worth most for ships and stations, which are 100 % delta today; (B) buys nothing and costs nothing. **(B) is not a stepping stone to (A)** — a world stamped eagerly gains nothing retroactively. Reuse the existing `sha2` workspace dependency for the id; do **not** introduce a second hash function (HR3) | The record layout and the id hash are permanent delta data of exactly the pyramid's class | **YES** |
| **D-9** | **If D-8 is taken: the snap lattice** (§7.3) — 1 m (no snapping, 100 % of the pyramid stored), 2 m (25 %), 4 m (6.2 %), **8 m (1.6 %)**, 16 m (0.4 %) | **8 m**, but this is a *taste* question about how building feels and I will not pick it for you | Not a one-way door in the loosening direction; **tightening later strands every existing stamp** | Tightening only |
| **D-10** | **Split `state_bits` out of `block_state`** (§9.6) into its own `(u32 local, u16 state_bits)` table | **Adopt.** 1.84× on the largest writer the table will ever have (the weather), for one table | Free now; a persisted record layout afterwards | **YES** |
| **D-11** | **Pyramid walk-up transaction and granularity** (§10.2) — (a) keep both; (b) move the walk-up into compaction; (c) add `pyramid_storage_block_cells` to `BlockStoreTuning` | **(b) and (c).** The design already argues the pyramid is fully reconstructible, so the crash consistency it pays for is not needed | (c) is a persisted-key decision | **YES for (c)** |
| **D-12** | **Edit fan-out coalescing** (§10.5) — one `ChunkDelta` per (chunk, tick) carrying the already-deduped batch, not one per edit | **Adopt, and land it WITH the D-9/D-39.5 per-cell reshape**, not after | Deferring means 20,000 messages/s in a city of 100 builders | No |
| **D-13** | **Dormant advance shape** (§10.4) — (a) realm-wide at spin-up with a tighter `max_advance_span`; (b) per chunk on first read, using the already-reserved `last_ticked_universe_tick` | **(b).** Zero extra storage, and the stamp is already in the reservation list | Free now | No |
| **D-14** | **Vertex format — register row 21, re-escalated** (§6.8). At today's 24-byte vertex a city's tier-0 disc alone is **355 MB**, against the 273 MB budgeted for an entire 100 km view; packed it is 32 MB | **Packed, chosen before the first city exists** | It is a mesh-encoder plus upload-path change, not a data migration — but retrofitting it after players have built means re-encoding every resident mesh | No, but it must precede content |
| **D-15** | **A columnar export tier for the economy ledger and run telemetry** (§4.6) — a separate analytics-only crate strictly outside `bins → node → sim → wire → core`, producing a **one-way** export no sim or node code may read | **Yes eventually, not before P4.** `arrow`/`parquet` are large trees and must never enter a Tier-A crate | None — it is additive and outside the dependency graph | No |
| **D-16** | **Confirm the `CellIndex` linearisation as a deliberate one-way door** (§4.3) — `(c*62 + b)*62 + a`, `a` fastest, already frozen into the WAL record, the pyramid entry and the mask's bit order | **Confirm, and give it an explicit line in the one-way-door table.** Today it is stated only in a doc comment. The chosen branch is also the better one (≈2× fewer runs than radial-fastest on terrain) | Already spent; the risk is that it is spent *silently* | Already through |
| **D-17** | **Do NOT build**, and rule out explicitly so they do not return: a columnar format for blocks; a shell-only storage representation; a shared-subtree graph for the live world; 1-D run encoding | **Adopt as a written ruling** | Free. Without it all four return | No |

**Cost of deferring the whole register:** rows D-2, D-4, D-5, D-8, D-9, D-10 and D-11(c) are format
freezes — they must land **before the first world is written**, and every one of them is a migration over
every saved planet, station and ship afterwards. D-1 and D-3 are free at any time but free-est now. D-6,
D-12, D-13, D-14 and D-15 can wait, provided D-5 is taken.

---

## 13. Adjudicated objections

**"The design has already independently arrived at nine of the eleven columnar techniques."**
*Partly upheld, corrected to seven full and two partial.* Run-length encoding is present only as the
degenerate single-value chunk, not as general RLE, and column separation is present in memory but only
partially on disk (§9.6). The correction does not change the verdict — the two *load-bearing* techniques,
dictionary encoding and column separation, are genuinely there, and the dictionary is in the strictly
stronger joint form.

**"The detail ladder attenuates a distant observer's city by 64× to 1,024×."**
*Rejected as stated.* Those figures compare a coarse rung against the **sparse** tier-0 encoding, which
the compactor would never choose for a city. Against masked-dense the first rung buys **3.6×**, not 64×
(§6.3). The ladder is excellent at distance and nearly worthless at the first step — and the first step is
the one a player crosses walking out of a city.

**"The pyramid roughly doubles delta storage."**
*Upheld for the clustered case and materially conservative.* Charging the pyramid a flat 8 B/entry
overstates it by **2.4×** once the encoding rule is applied at every rung (§6.2). But the same
conservatism does not survive §9.2's scatter row, which understates it by **5.5×** in the opposite
direction. The two errors do not cancel; they apply to different worlds.

**"Storing a surface index would make the per-row query cheap."**
*Rejected.* Recomputing is 65–120 µs per chunk; storing is 29,791 B per chunk of write plus invalidation
across up to eight chunks per edit, so the index is dirtied more often than it is read (§5.4). This is the
same conclusion the design already reached when it deleted the content-keyed mesh cache.

**"A city should become its own realm so it can be spun up and budgeted independently."**
*Rejected for the ground case, unnecessary for the floating case.* The ground case is structurally
forbidden today; the floating case is already a construction realm at zero cost; and the pressing half of
the motivation — budget isolation — is obtainable by making the delta budget hierarchical, with no realm
change at all (§6.7). The key ordering already makes a later spatial extraction a store split rather than
a format migration, so nothing needs reserving.

**"Add 1-D run encoding as a cheap first step toward better compression."**
*Rejected.* It is dominated 92× by 3-D boxes on a hollow building and 200× on a solid cube, for the same
implementation cost, because half of any axis-aligned building's walls are perpendicular to whichever
axis a run encoding picks (§7.2). Ship boxes; skip runs entirely.

**"A shared-subtree graph gets 32× and should at least be considered for the pyramid."**
*Rejected, and the usual reason is the wrong one.* The decisive objection is not the edit cost — it is
that deduplication deliberately creates references from everywhere to everywhere, which destroys the
ability to ship or store one chunk independently, and per-chunk independence is what the delta manifest,
coarse-first loading, the per-(tier, chunk) storage block and range-scan-by-rung all rest on (§7.4). The
32× also applies to occupancy only, and our cell is an 8-byte record.

**"`zstd` is effectively free because it is already in the lock file."**
*Rejected on fact.* What is in the lock is `ruzstd`, which is a **decoder only** and cannot compress a
single byte. `flate2` is genuinely there transitively and is the only zero-new-crate option; `zstd` is a
new C binding (§9.5).

**"The pyramid's isolated-edit row is the adversarial worst case and is bounded, so scatter is handled."**
*Rejected.* True isolation requires 4,096 m spacing, and only **19,577** such sites exist on the whole
starter body — 1.9 MB in total. The design bounds the case that cannot happen and omits the case that will:
57 m spacing, six entries per edit, 4.86 GB (§9.2).

**"The block field is 51.0 MiB at any view distance."**
*Rejected as stated; needs the qualifier "on unbuilt ground."* Voxels stay resident for **any chunk
carrying a delta**, and in a city that is all 505 tier-0 columns: **115.3 MiB** at an 8-bit palette, and
**261.4 MiB** at the 15-bit palette a determined builder can produce in about forty minutes (§3.1). The
residency governor should be sized against the city figure.

**"A `COARSE_SHAPE` table with no threshold anywhere solves the distant-blob problem."**
*Upheld for the failure it targets; a different one is unexamined.* The table is indexed by
`(occ_mask, fill bucket)` and calibrated against terrain, whose fill distribution is unimodal — a coarse
cell over a hillside genuinely *is* half full and a half-block genuinely *is* right. A city's is bimodal:
a building interior is 255 and a street is 0, so at the 4 m and 8 m rungs a cell straddling a 10 m street
and a 20 m wall lands in the middle buckets while the octant mask reports "mostly solid". The plausible
result is a systematic corrugation across the whole city at exactly the rungs used between 1.6 km and
6.3 km — a *structured* artefact, which reads as intentional and is arguably worse than a blob. **The
remedy is a gate, not a design change:** the acceptance run (*dig a tunnel and build a tower, fly away,
confirm both are visible at every rung boundary*) must include a **built region with streets at rungs
2–4, at walking speed.** This is the only genuine visual risk this analysis found, and it is flagged as
*plausible*, not verified — nothing voxel exists in the workspace to measure it against.

---

## 14. Risks and confidence

- **The planetary-scatter model** (§9.2) is geometric, not measured. Its *shape* is unavoidable — the
  log₄ term is arithmetic — but the constant depends on how real players distribute marks, which nobody
  has measured. It should be a load fixture before the budget default is chosen.
- **The write-amplification arithmetic** (§10.2) assumes redb rewrites a value's pages on update. That is
  true of any copy-on-write B-tree and matches the shipped store's whole-value `insert`, but I did not
  read redb's page allocator. The multiplier could move 2× either way; the conclusion survives.
- **The city geometry model** (§6.1) is a model. Five independently-built models normalise to 4.13–5.90
  bytes per cell, which is reassuringly tight, but a player who builds *solid* rather than hollow roughly
  doubles the cell count per building — and that is precisely the case box encoding absorbs.
- **The mesher's 65 µs / 90 µs figures** are the reference implementation's, on a Ryzen 5 5500, not ours.
  Our own mesher is booked to rise to ~95 µs with the twelve face masks. Quote **~95–120 µs**.
- **The SVDAG multipliers** (§7.4) are recalled from the literature and were not re-verified. Do not
  quote them in an owner-facing comparison without checking.
- **The coarse-shape corrugation risk** (§13) is reasoned, not observed.
- **Scope.** This document covers storage, the edit path and the streaming arithmetic only. It does not
  verify the shape catalogue, the signal bus, the decoration layer or the collision path — and a change
  to the pyramid's storage granularity or transaction placement touches the coarse-rung render path that
  §5 of the block design owns. **Nothing voxel, chunk, block or mesh exists in the workspace today**, so
  every design figure here is a design figure.
