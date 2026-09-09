# CRITIC B — the laws, the cost, the integration and the slice plan

**Date:** 2026-09-08. **Subject:** `docs/investigation/2026-09-08/landforms/00_proposed_landforms.md`
(the synthesis), read against the eight domain documents `01`–`08` of the same date, against the code,
and against the binding rulings.

**Method.** I read the synthesis in full. I skimmed the eight domains for the numbers the synthesis
carries. I opened every file the synthesis cites and checked the citation. I recomputed the largest
numbers myself with `python3`. **I ran no build, no test and no gate**, as instructed.

**Marks.** MEASURED = a program in this repository produced it. COMPUTED = my own arithmetic, shown.
VERIFIED = I opened the file and the text matches. UNMEASURED = nobody knows.

**The verdict in one line.** The diagnosis is sound and it is MEASURED. The architecture is lawful in
its shape. **Four blockers stop the register from being answerable today**: the artifact's true size is
not stated, the per-chunk budget is reported as a pass where the source document reports a fail, the
flagship picture needs three things no slice owns, and HR4 is not answered for one slice of the eight.

---

## THE FINDINGS

### B1 — BLOCKER — The 14.75 MB artifact omits the climate field, and D19 multiplies that field by 6.25

**What the text says.** §3.1 keeps the climate field "with the artifact". D19 rules **ONE lattice**: the
climate runs on the macro lattice, inside the solve's schedule. §1.4, §3.1 and §3.3 all carry
**14.75 MB** as the artifact's size, and §3.3 says that number "replaces 9.83 MB everywhere".

**What I computed.** The 6-byte record holds `Z`, the water level, the receiver-plus-facies byte and the
discharge byte. It holds **no climate**. Document `05` prices the climate at 16 bytes per cell on **its
own** grid of `6 x 256² = 393 216` cells, which is **6.29 MB**, plus a 10-byte anomaly field at
**3.93 MB** (`05:501,1357`). D19 moves that field onto the macro lattice of `6 x 640² = 2 457 600` nodes.

```text
   COMPUTED, the same 16 bytes on the two lattices
   05's own climate grid   6 x 256^2 =   393 216 nodes x 16 B =  6.29 MB
   D19's macro lattice     6 x 640^2 = 2 457 600 nodes x 16 B = 39.32 MB     6.25 x
```

The river polylines and the lake table are also cached "with the artifact", and §3.1 says their
per-node segment bound is **UNMEASURED**.

**Why it is a blocker.** Five things rest on 14.75 MB and every one of them moves:

1. **D1**, derive against ship. §1.4's line *"the 14.75 MB artifact is about one per cent of the mesh
   the same picture holds"* becomes about four per cent, or more.
2. **D3**, the lattice resolution. The 5 140 m option's 37.7 MB becomes about 137 MB.
3. **D31**, the client's residency ceiling, which the owner is asked to state.
4. **B11 below**, the client's per-body memory. `05` already warns that a client visiting every body
   holds up to 180 MB of climate grid at `05`'s own resolution; at D19's it is over 1 GB.
5. **A3**, the SL6 ask that opens the bulk lane if the derive is too slow. The lane's payload is not
   14.75 MB.

**The fix.** State the artifact's TOTAL kept bytes with every field named — `Z`, water, receiver plus
facies, discharge, the climate inputs, the river polylines, the lake table, the pyramid. Or keep the
climate on its own coarser lattice and withdraw D19's "one lattice" as a **memory** claim while keeping
it as a **schedule** claim, which is the half `05` Q3 actually argues for.

---

### B2 — BLOCKER — §1.4 reports the densest chunk under budget; `07` reports the same chunk OVER budget, with no water sheet

**What `07` §7.1 measures and states.** Three named chunks, priced at the two measured marginal
extraction costs:

| Chunk | today | vertices | new, at 2.0x |
|---|---|---|---|
| surface (3, 5) | 3.30 ms | 5 650 | **4.10 to 4.34 ms** |
| cave-dense SEAM (21 223, 7) | 6.11 ms | 8 234 | **7.07 to 7.42 ms** |
| **cave-dense (3, 5)** | 5.22 ms | **23 084** | **7.09 to 8.07 ms — OVER** |

`07`'s own words: *"The document must report a fail, not a pass … the landform work puts the chunk over
the 8 ms budget by 0.07 ms."* No water sheet is in that row.

**What the synthesis carries.** §1.4 lists the surface chunk and the **seam** chunk only, and then
states: *"The water sheet breaks the 8 ms budget, and it is the only term that does."* The cave-dense
(3, 5) chunk — the one with 23 084 vertices, three times the seam chunk's count — is absent.

**Why the omission matters, and it is not a rounding matter.** The term that breaks the budget is **the
extraction growing with the roughness**, and roughness is the whole point of the arc: §1.3 raises the
rms slope at a 1 m baseline from 1.7° to 18.9°. D29's three dials are **all aimed at the water mesh** —
mesh it only where a water level exists, mesh it at a coarser rung, `SUBDIV = 3`. **Not one of the three
touches the rock extraction.** So the owner would take D29 believing the risk is a water sheet he can
dial away, when the measured risk is the ground itself.

**The fix.** Put the cave-dense (3, 5) row in §1.4, state that the landform work alone puts it over
budget on `07`'s own arithmetic, and give D29 a fourth dial that acts on the rock extraction — a vertex
budget per chunk, a roughness cap keyed to the cell, or a coarser rung for the cave band.

---

### B3 — BLOCKER — The flagship picture needs three things no slice owns, and one of them is a 🟥 row nine slices away

**The arc's own acceptance is the reference picture.** §6.5 maps every row of it to a slice. Three rows
have no slice, and the synthesis says so in three separate places without drawing the conclusion.

1. **The sky.** §6.5: *"Blue haze with distance … 8b gives the number; **no slice owns the sky**."*
   Every picture the arc produces has a black sky.
2. **The shadow pass.** §1.5 item 6 and Q6: *"no slice owns a shadow pass."* VERIFIED in the code:
   both lights are created with `shadows_enabled: false` (`crates/client-render/src/terrain.rs:459,473`)
   and a fill light runs at `FILL_SHARE = 0.08` (`:56`). The reference picture's raking light — the
   thing that makes a ridge read as a ridge — cannot be reproduced at all.
3. **The stand's own up.** §6.4 makes the flagship *"the PILOT'S EYE at 1.8 m"*, standing on the home
   planet. VERIFIED: `D-TERRAIN-4` (🟥, `DEFERRED.md:7794-7812`) says the stand's up is stated by the
   operator through `VD_SPAWN_POSES`, and *"the windowed first-person camera still keeps the frame's
   `+Y` as its up, so a windowed walk on a planet is tilted until it lands."* Ruling **V11** (up is the
   server's) lands at slice 16. **The synthesis never names V11 and never names `D-TERRAIN-4`.**

**Why it is a blocker.** The owner is asked to approve 19–24 weeks whose flagship deliverable is a
50 km vista judged against a picture with haze, raking shadow and a standing character. The arc
delivers a black sky, flat light and an operator-typed up. The comparison the whole arc exists to win
cannot be run.

**The fix.** Either name the sky, the shadow pass and V11's up inside the arc — three more slices, and
say what they cost — or rewrite §6.5 so the owner sees, in one column, which rows of his own reference
picture this arc **cannot** deliver at any budget.

---

### B4 — BLOCKER — HR4 is not answered for one slice of the eight

**The law.** HR4: *"every feature ABOVE the seam passes the identical fixture on ≥ 2 shard kinds
(G-IDENTICAL) or it doesn't land."* The voxel foundation binds it to **every** feature slice, in its own
rule 2, after its law critic raised exactly this as finding F3 and the document FIXED it — each of
slices 9, 10, 11, 12, 14 and 16 now names a PAIR and an identical fixture body.

**What the synthesis does.** §6.2's eight slices name **no pair, no G-IDENTICAL fixture and no
Cartesian-profile leg**. §5.2 raises the subject and then leaves it: *"`G-MAPPING-TABLE` and
`G-MAPPING-ROUNDTRIP` are the geometry seam's gates, and naming them does not answer HR4 for this
layer."* That is a correct observation with nothing behind it. Q11 asks the question — *"May a hull or
a station hold a macro artifact?"* — and leaves it open while eight slices are priced against it.

**Why it binds here and is not a formality.** Ruling **V4** says a hull or a station MAY hold terrain,
and ★V38 (A) makes that the reason slice 10 can land at all. `crates/sim/src/capability.rs` gives the
planet `VoxelGeometry::Spherical` and the ship, the station and the asteroid `VoxelGeometry::Cartesian`.
A hull has no seed, so it has no charter, no macro lattice, no D8 chain and no climate. **In the game's
words:** a miner digs the same trench with the same tool on the home planet and in a station's soil
bay. Under this arc the first trench reads a macro artifact and a biome, and the second reads neither.
That is a fork decided by the shard kind, which is what HR3 and HR4 exist to refuse.

**One more thing unstated.** The macro lattice is a **new address family** (`07` calls it "the family-2
address"). Nothing says whether it sits above or below the `GridMapping` seam, and therefore nothing
says whether it owes `G-MAPPING-TABLE` and `G-MAPPING-ROUNDTRIP` at all.

**The fix.** Give every slice of §6.2 a PAIR and an identical fixture, answer Q11 before 8c prices the
solve, and place the macro lattice on one side of the geometry seam by name.

---

### B5 — DEFECT — The free window is claimed to close at slice 9; the foundation says it closes at slice 14

**What §6.1 says.** *"Format D's tolerance is ZERO (ruling S5-2). Every landform slice moves rung-0
bytes. The free window closes when the first world a player owns is saved, and the store lands at
slice 9. So the whole arc sits between slice 8 and slice 9."*

**What the foundation says, in its own words** (`00_proposed_voxel_foundation.md:492-499`, rule 1):
*"Therefore the stores of slices 9 to 13 are THROW-AWAY until slice 14 lands and the world identity is
pinned with the feature anchors in it."*

So the free window runs to **slice 14**, not to slice 9. The whole ordering rule of §6.1 is built on the
wrong boundary.

**What the wrong boundary costs.**

1. **19–24 weeks are inserted in front of the store, the diff lane, the collider and the character** —
   slices 9, 10, 11 and 16 — when they could be interleaved for free.
2. **8f cannot land where it is put.** It ships *"PLACED ROCK OBJECTS with their own colliders"*. The
   collider is slice 11. Under §6.1's rule, 8f runs before slice 9 and therefore before slice 11.
3. **Slice 18 cannot land where it is put.** §3.1 caches the live weather in *"the realm's
   checkpoint"*, which is slice 9's store.
4. **8e's ANCHOR DIGEST belongs with slice 14.** The foundation's rule 1 says slice 14 pins the world
   identity **with the feature anchors in it**. Adding an anchor digest at 8e means the identity moves
   again at 14, which is exactly what rule 1 is written to schedule.

**The fix.** Re-derive the ordering against rule 1. A defensible arc is: 8p, 8a and 8b before slice 9;
8c and 8d after slice 11, where the collider that must agree with the shape exists; 8f and 18 after
that; and the whole arc still closed before slice 14.

---

### B6 — DEFECT — D5, the decision "nothing else is safe until it is settled", contradicts itself

§7.2 merges four documents into one octave law. Two of its own rows disagree.

- **"The ends"**: the table runs from `C · cell_m(the rung below the macro node)` down to
  `C · cell_m(0)`, with `C = 8`, and `LONG_WAVE_CAP_M` and `SHORT_WAVE_M` both deleted. `03:386`
  confirms the composition: **`Z` REPLACES the coarse octaves.**
- **"The count"**: *"The count is NOT reduced to save cost — no saving is credited."*

**COMPUTED on the home planet.** `N = 5 263 360`, twelve rungs, `cell_m(11) = 2048 m`. The macro node is
8 224 m. `C · cell_m(11) = 16 384 m` at the top and `C · cell_m(0) = 8 m` at the floor. That is
`log2(16 384 / 8) = 11` octaves. The crate has **fourteen** today (VERIFIED,
`crates/terrain/src/home.rs` test: `octave_count == 14`). **So "the ends" reduces the count from 14 to
11 while "the count" says the count does not fall.**

`07` §7.1 books the five replaced octaves at **ZERO** and calls that conservative, so every cost figure
in the arc assumes the count does not fall while the octave law assumes it does.

**The fix.** State one rule. Either `Z` replaces the coarse octaves and the count falls, and then
re-price the column pass with the saving credited; or the octaves keep their coarse end and the fold
double-counts the relief `Z` already carries, which is a defect of its own.

---

### B7 — DEFECT — 8a's picture is taken at a relief that 8b replaces

**The order.** 8a lands the slope spectrum, the roughness placeholder and the wavelength rule, and
produces **the pilot's vista** — §6.3 calls it *"the cheapest possible test of the owner's own
sentence"* and *"the first honest test of 'not interesting enough'"*. 8b then lands the charter and
**D6, the new relief law**: `draw × min(strength bound, shape bound)` replaces `0.4 % of radius, capped
200–12 000 m, × [0.5, 1.5)`.

**Why the order is wrong.** The slope spectrum's amplitudes are anchored on the relief. VERIFIED in the
code: `crates/terrain/src/body.rs:181` re-normalises every amplitude so their sum is the relief. §7.2's
own trap paragraph says it plainly: *"the slope spectrum and the budget together are what make the
change safe, and they are ONE decision, not five."* D6 changes the relief total. So the 50 km vista the
owner judges at 8a is drawn at 14 304.9 m of relief, and 8b may raise or lower it.

**In the game's words.** The owner stands a pilot on the home planet at 8a, looks at the ridge line and
says *"yes, that is interesting."* At 8b the strength bound lowers the same planet's relief, and the
ridge he approved is not the ridge he gets.

**The fix.** Land D6's relief law and the sea before 8a — it is a small change beside the spectrum — or
stamp the 8a picture "the relief is provisional; 8b changes it" so the owner's YES is scoped.

---

### B8 — DEFECT — The free-coarsening law is proved for the octaves and claimed for the whole fold

**The claim.** §4.1: *"rung `L`'s height is exactly the first `n − L` steps of rung 0's fold, and
everything the coarse rung leaves out is a sum of amplitudes it can name."*

**The fold.** §2.1 runs `s0 = MACRO + Z`, then the octave loop, then `s+ = TERRACE(s_k)`, then
`h = CARVE(s+)`, then the 3-D removals. **TERRACE, CARVE and the removals come AFTER the octave loop.**
A prefix claim cannot hold while a later step is dropped or changed.

**What is unstated.** Do TERRACE and CARVE run at every rung, or do they stop?

- If they run at every rung, the coarse rung's answer is **not** a prefix and the bound is **not** the
  dropped amplitudes. §4.1's proof does not apply and no other bound is offered.
- If they stop at a rung, a channel appears in ONE rung step. `07` §6 states the step: *"at most the
  channel depth the artifact states at that node."* On a trunk valley that is hundreds of metres,
  against a half-cell budget of 32 m at rung 6.

§4.4 admits it in one line — *"The only pop left is the dropped channel carve, and `M-L14` measures
it"* — and `07` answers with a hope: *"If it shows, the cure is the crossfade the octaves already
use."* Under SL8 a step of that size is a jump, and a measurement is not a rule. **8f's 3-D removal has
no coarsening rule at all**, and an arch that vanishes between two rungs is the same defect again.

**In the game's words.** A pilot flies toward the home planet's great valley. At one rung boundary the
valley floor rises two hundred metres under his nose, and the river's bed with it. That is the seam SL8
names.

**The fix.** State the coarsening rule for every term of the fold, not only the octaves: at which rung
each term stops, and what its dropped magnitude is bounded by. Then §4.1 can say "prefix" honestly.

---

### B9 — DEFECT — After this arc the static shape is a function of (seed, address, STORED CHARTER), and the text never says so

**SL10 clause 1**, as CLAUDE.md states it: the client may derive what *"`(seed, address)` alone decides,
never a function of time or live state."*

**What D14 and D15 do.** The charter becomes *"quantised INTEGERS, authored ONCE and STORED, never
re-derived"*, the parent system computes and quantises the orbit-derived half and states it one hop
down, and A1 ships the whole charter to the client in `TAG_SURFACE`. The reasoning is sound and I
verified its ground: **there is no `crates/physics/clippy.toml`**, so the forest's floats sit outside
the fence, and two hosts can quantise either side of a grid line.

**What follows, and is not stated.** The shape is now `f(seed, address, charter)`, and the charter is
**stored state**. Two consequences the text does not draw:

1. **No body's shape can be derived before its realm runs and states its self-look.** `look.rs:52-56`
   VERIFIED: `TAG_SURFACE` *"rides `BodyStmt::SelfLook`, which only a RUNNING realm may send"*. So
   nothing can be warmed ahead for a dormant realm, and the residency band of slice 8's precondition
   (b) has to wait for a realm to speak before the client may build one chunk.
2. **D17's tuning re-shapes every stored body.** D17 says the draw laws for spin, obliquity, water
   inventory, optical depth, elastic thickness and the glacial offsets *"are WRITTEN AND TUNED before
   slice 9"*, and adds *"a first draw law is always tuned … They are a one-way door of their own."*
   A re-tuned draw re-authors every stored charter, which re-shapes every stored body.

**The fix.** Put it to the owner as its own register row: *"SL10 clause 1 is widened — the static shape
becomes a function of the seed, the address AND the body's stored charter."* It is his ruling to give,
not an implication for a synthesis to carry.

---

### B10 — DEFECT — D1(c) needs a CONTENT digest that A6 leaves no lawful way to learn

**What D1(c) requires.** §3.1: the client's disk cache is keyed *"by the world identity **and a CONTENT
digest**"*.

**What A6 withdraws.** *"~~The macro-artifact digest inside the look~~ — **WITHDRAWN.**
`crates/core/src/look.rs:70-73` forbids the measured half in a look by name … The artifact carries its
digest with the artifact."* I VERIFIED the citation; it says exactly that.

**Why the two do not meet.** A **derived** artifact carries only the digest the client itself computed,
so comparing it validates nothing. A **cached** artifact carries the digest that was stored beside it,
so comparing it proves the file is not corrupt and never proves it matches the realm's. There is no
third source. So a client whose charter moved between sessions, or whose binary changed, re-uses a
stale cache and draws different ground from the server — with **no refusal anywhere**, because S7-3's
refusal compares the GENERATOR tag and nothing else.

**In the game's words.** A pilot lands in the same valley he landed in last week. His client reads its
cached artifact. The realm re-drew its charter after a tuning. The river on his screen runs where last
week's river ran; the shard's collider stops his boots where this week's river cut. He falls through the
ground he can see.

**The fix.** Either key the cache on the FULL input tuple — the seed, the charter bytes and
`GENERATOR_VERSION` — and say so instead of saying "a content digest"; or make the digest an SL6 ask on
a lane, which is a request the register does not currently make.

---

### B11 — DEFECT — The client's derive has no ceiling, and the warm-up lead is unstated against a speed the owner refused to cap

**What D1 commits the client to.** Every client derives every body's artifact: **12–20 s of one core,
plausibly 40 s, and 125–130 MB transient, per body.**

**What is fenced.** §5.4 fences the GATEWAY: *"A structural control proves the gateway holds no
artifact."* Correct, and it names the right precedent — the measured 5.3 GB-at-boot defect. **Nothing is
fenced on the client.** There is no cap on concurrent derives, none on resident artifacts, and none on
the transient peak.

**The lead nobody names.** Slice 8's precondition (b) asks the residency band to *"STATE the distance at
which a body's whole-body data must be complete, because nothing today names it"*. The owner's ruling of
2026-08-27 says: *"grow the interest radius with the closing speed; never cap the speed — a cap makes
the game unfair, because the player would pay for a server's start-up cost."* So the required lead is
`closing speed × the derive time`, and the closing speed has no upper bound.

**COMPUTED, to show the shape.** At 40 s of derive, a hull closing at 240 m/s needs 9.6 km of lead; at
30 km/s it needs 1.2 million km; at a warp leg's speed it needs a lead no window can carry. **That is a
new time constant of exactly the class the suit ruling deleted for the walk.**

**What M-L7 measures, and what it misses.** §9.1's M-L7 has two legs, and both stand on the **home
planet**, whose artifact is a committed build artifact. That is the one body where the problem cannot
appear. The measurement that matters — a pilot arriving at the home system's second planet — is not in
the plan.

**The fix.** Add three numbers to §7 as owner rows: a client ceiling on concurrent derives, a ceiling on
resident artifacts, and the residency band's stated completion distance with the closing-speed rule
written out. Add a leg to M-L7 that arrives at a body whose artifact is **not** pinned.

---

### B12 — DEFECT — The artifact is never re-solved, so a player who reshapes the ground gets a river that ignores him

**What Q10 asks.** *"Does a terraformed ridge cast a new rain shadow?"* Answer given: no new GRID
shadow, a new LOCAL and MID shadow. The question stops at the rain.

**The same artifact carries more than rain.** It carries the D8 receiver chain, the discharge, the water
level and the facies. 8d keys the channel carve, the floodplain and the meshed water sheet to them.

**So, in the game's words.** A player raises a ridge across the valley below his hull's berth. The macro
artifact is not re-solved — SL10 forbids time inside the recipe, and D35 makes it static for this arc.
The channel still runs where the seed's water ran, straight through his new rock. The water sheet still
stands at the old level, so his dam holds nothing back and his new lake never fills. He digs a gorge to
the sea and no water follows it.

**No slice, no decision row and no DEFERRED entry owns this.** It sits at exactly the place players
spend their time — the ground they change — and it is more visible than the rain shadow Q10 does name.

**The fix.** Widen Q10 into one question — *"what does the artifact owe a changed surface?"* — and give
it a stated answer, most likely: **nothing, and the local diff wins inside its own width.** Then name
the compromise so the owner rules on it instead of a player finding it.

---

### B13 — WEAKNESS — Two register rows carry no recommendation, and one of them gates the decisive measurement

Every row of §7 carries a **Recommended** except two.

- **D28, the ocean fraction.** *"the owner picks a target share, and the water inventory's draw range is
  set to it."* No number is proposed. The row gives him Earth's 71 % and today's measured 1 %, and the
  fact that +1 270 m of sea offset reaches 71 %. That is good ground; it is still not an answer.
- **D31, the client's mesh residency ceiling.** *"the owner states one."* No anchor at all — no figure
  for what a client holds today, no comparable from any shipped game, no measurement of the machine the
  pictures are taken on. The owner has nothing to reason from.

**Why D31 matters more than D28.** §9.1 makes **M-L19** — *"whether the reference picture is affordable
at all"* — pass or fail **against D31's ceiling**. A measurement whose threshold nobody proposes cannot
go red. The kill table then fires *"fewer rings"*, which the same table admits *"is a detail-by-distance
seam"*, so the fallback is an SL8 defect chosen by a gate that could not fail.

**The fix.** Propose a number for both, with its basis, even provisionally. For D31 the basis exists:
measure what a client resident set costs today at one rung and state the ceiling as a multiple of it.

---

### B14 — WEAKNESS — D37 makes the erosional age a typed world constant, where the law and the owner's own words make it a body fact

**What D37 does.** The solve's pass counts become *"stated constants … stated as a physical EROSIONAL
AGE with `dt = age / PASSES`"*. The reasoning is right: naming a physical age makes the pass count a
RESOLUTION rather than a dial, and M12 proves it.

**What it misses.** An **age** is precisely the kind of number the no-magic-numbers law says must be
seed-derived or a physical fact of the body. The owner's own sentence names *"position, spin,
trajectory, size and gravity"*. A body's age belongs beside those five, and the charter already carries
eighteen to twenty integers.

**What one typed age costs the picture.** Every planet in the world gets the same amount of wear. A
young world's sharp unworn ridges and an old world's rounded stumps are the same landform family under
one constant — and telling them apart is free variety of exactly the kind §1.1 says the arc is buying.

**The fix.** Draw the erosional age per body from the seed, add it to the charter, and keep `PASSES` as
the resolution. M12 is unaffected: it asserts that `PASSES` and `2 × PASSES` at the **same** age agree.

---

### B15 — WEAKNESS — 8f's placed rock objects have no format, no host and no collider

D13 and slice 8f land *"PLACED ROCK OBJECTS with their own colliders — talus, boulders, outcrops"*, and
D13 is right that ruling V4's *"trees, grass and decoration"* does not cover them, because they need a
collider. Three questions are then left open inside a slice the register asks the owner to approve:

1. **Seed-placed or live state?** If a boulder is a function of `(seed, address)` it is SHAPE: both
   hosts must place it byte-identically, and it belongs beside slice 14's feature anchors **inside the
   world identity**. The synthesis argues exactly this for a tree at 8e (*"a tree a player stands on is
   part of the shape"*) and does not carry the argument to a rock.
2. **Which Format B row?** VERIFIED against the frozen record: `object_param` is 8 bits and is legal
   only on a kind whose form row carries OBJECT; a non-zero `object_param` on any other kind is
   REJECTED at decode. A boulder must therefore be an OBJECT-form catalogue kind, and nothing says it
   is one.
3. **Which slice supplies the collider?** Slice 11 owns colliders, and B5 shows the ordering rule puts
   8f in front of it.

**The fix.** Answer the three in 8f's paragraph before the register asks for it.

---

### B16 — WEAKNESS — D4's chooser must be told what it may read, or it is a twelfth drift class

**What D4 says.** *"one node size, with a chooser walking coarser until a stated memory and time ceiling
hold."*

**The ambiguity.** If the ceiling is a **committed constant**, the chooser is a pure function of the
body and it is lawful. If the chooser reads **the host's free memory** or its measured speed, then two
hosts pick two lattices for one body, and the shape drifts. That is the one failure §5's eleven
determinism rules exist to prevent, and **none of the eleven names it.**

**The fix.** Add rule 12: *the chooser reads only the body's own charter integers and committed
constants; it never reads the host.*

**A smaller point in the same row.** Rule 10 requires the macro lattice to divide `N` exactly, and D3
claims *"A METRIC target that divides `N` exactly gives every body the same skeleton."* COMPUTED on the
Earth-like candidate (`N = 10 235 904 = 2^12 · 3 · 7² · 17`): **8 224 does not divide it** —
`10 235 904 / 8 224 = 1 244.64`. The nearest legal divisors give node sizes of 7 168, 7 616, 8 704,
9 408 and 9 996 m. So "the same skeleton" holds only to about ±20 % before the chooser walks at all.

---

### B17 — NOTE — Four numbers I could not reproduce

I recomputed the arithmetic myself. Four figures did not come out.

1. **`07` §5.1's Earth-sized node count, 11 692 896.** That is `6 × 1 396²`. COMPUTED:
   `10 235 904 mod 1 396 = 432`, so 1 396 does **not** divide that body's `N`. The figure breaks the
   document's own determinism rule 10. It is used in D4's case, so the case survives; the number does
   not.
2. **§5.2's G-MACRO-AREA "a 30 % spread".** The two weights quoted are 0.702 at a face-edge midpoint and
   0.758 at a cube corner (`03:490-491`). Those differ by **8 %**. Against a face centre's 1.0 the spread
   is **42 %**. Thirty per cent is neither.
3. **§1.4's vista row, "2 805–4 200 chunks, 1.0–1.1 GB".** The two ends come from different per-chunk
   byte figures: `08` uses 405 KB (`2 805 × 405 KB = 1 136 MB`, COMPUTED, matches) and `07` uses about
   248 KB (`4 200 × 248 KB = 1.04 GB`). More chunks give **fewer** bytes, so this is not a range and it
   should not be printed as one.
4. **§1.4's chunk figures.** The synthesis prints "4.10–4.67 ms" for the surface chunk and "7.42–7.47 ms"
   for the cave-dense seam. `07` §7.1 gives **4.10–4.34** and **7.07–7.42**. Neither upper figure
   appears in the source.

None of these changes a decision on its own. Together they say the cost tables were copied between
revisions rather than recomputed, which is what B1 and B2 also found.

---

### B18 — NOTE — Q4 is the arc's whole enforcement mechanism and it has no owner

**Q4:** *"How does a RED gate land without turning `just gate` red for every unrelated change? Nobody
owns this. A believability gate that blocks every merge will be switched off, and a gate nobody runs is
not a gate."*

**How much rests on it.** §6.2 turns nineteen picture verdicts, a slope histogram, a pop detector, a
vista census and a drainage-density band into gates, and §6.1 says slice 8's detectors *"are written to
be RE-RUN by every later slice with one command"*. VERIFIED in `justfile:408`: `gate` runs
`terrain-pin`, `terrain-link-scan` and `terrain-fence-control`, and **not** `terrain-cost`;
`terrain-legs` needs Docker and is deliberately outside. So the project already has three tiers of gate
and no written rule for which tier a new one joins.

**What a ratchet needs, and none of the three is named.** A stored baseline; an owner for the baseline;
and a rule for who may move it and on what evidence. Q3 asks for the ratchet by name — *"the headline
gate lands as a RATCHET — it may not get worse"* — without any of the three.

**The fix.** Make Q4 a register row, not an open question. The arc's acceptance is the reason it exists.

---

## CHECKED, SOUND

These I opened, recomputed or verified, and they hold.

1. **The diagnosis is a measurement, not an argument.** VERIFIED in the code: the picture harness puts
   the star 25° up (`crates/bins/tests/terrain_pictures.rs:44`) and then points the nose at the star's
   own azimuth (`:458-462`, `toward_sun`), and both lights carry `shadows_enabled: false`
   (`crates/client-render/src/terrain.rs:459,473`) with `FILL_SHARE = 0.08` (`:56`). Every picture the
   owner judged was shot into the light, with no shadow, on 1.4° ground. §1's fourth cause is right.
2. **The octave constants and the re-normalisation trap are real.** VERIFIED: `SHORT_WAVE_M = 30` and
   `LONG_WAVE_CAP_M = 400_000` (`crates/terrain/src/body.rs:24-25`), and `body.rs:181` re-normalises
   every amplitude so the sum is the relief. §7.2's warning — shortening the coarsest wave without
   changing that rule makes the world eight times steeper — is a real trap and it is well placed.
3. **The ladder's bound is exactly as weak as §4.5 says.** VERIFIED: `octaves_at` drops by COUNT with a
   "never fewer than one" clamp, and `relief_bound_m` / `dropped_bound_m` are the bound
   (`body.rs:252-277`). The property test at `height.rs:80-105` asserts only
   `|h_L − h_0| ≤ dropped_bound_m(L)` over 400 samples, so it cannot see a channel. Growing it two legs
   — the wavelength rule and a channel-following fixture — is the right repair.
4. **D21's arrival pop exists today.** VERIFIED: `biome_at(body, dir, surface_m)` takes the rung's own
   height (`height.rs:39-65`), so a column near the snow line flips class with the detail level and the
   white patch moves as a pilot flies in. One derived climate rung, with an `assert_eq` instead of a
   share, is the right cure.
5. **D22 is correct against the FROZEN record.** VERIFIED against Format B: the 12-byte record carries
   no biome field, and `object_param` is 8 bits and legal only on an OBJECT-form kind. Nineteen biomes
   in five bits change no saved byte, because a biome is a column property both hosts derive.
6. **D10's mesa argument is right in the code.** `crates/terrain/src/strata.rs` selects a substance by
   depth under the surface, so every layer drapes over the hill and the order is soft over hard. A
   hard cap over a soft layer — a mesa, a hoodoo, a benched cliff — is impossible at any erosion rule
   until the column is fixed at a radius.
7. **§5.4's refusal to fold the solve into the boot self-check is the most important defensive call in
   the document.** The gateway self-checks the home planet at boot today, and it owns no planet. Folding
   a 12–40 s solve there would repeat the measured 5.3 GB-at-boot defect in a new place. The four-part
   cure — a committed home artifact on the SHIPPED path, a kernel canary, a digest checked on every
   read, a structural control — is the right shape and it is stated as a control that could fail.
8. **D14's ground is verified.** There is **no `crates/physics/clippy.toml`**, so the forest's floats
   sit outside the float fence. Authoring the charter once, quantising it and storing it is the right
   cure, and the operational half — a realm rescheduled to another architecture would re-address every
   stored edit — is a real failure mode and a good reason.
9. **D18 is right, and it keeps a pin.** VERIFIED: `POLE_AXIS = +Z` is documented as the orbit's own
   axis (`height.rs:30-35`) and cross-pinned in `crates/bins/tests/home_body_pin.rs`. Obliquity is a
   relation between two frames, the parent owns that relation under SL1 clause 1, and carrying the
   cosine as a charter scalar keeps `sin` and `cos` outside the fence. The pin survives.
10. **D25 and A5 read the reach ruling correctly.** One statement per realm that every observer
    evaluates costs the realm nothing per observer; a composed-per-observer field is the shape the reach
    ruling killed by name. A4's refusal to ride `TAG_SURFACE` is also right: VERIFIED at
    `crates/core/src/look.rs:52-56`, the tag is *"ONCE PER REALM ON CHANGE, carried retained, never on a
    keep-alive"*, which weather is not.
11. **The artifact and pyramid arithmetic reproduces exactly.** COMPUTED by me: `N = 5 263 360`,
    `N / 640 = 8 224`, `6 × 640² = 2 457 600` nodes, `6 B × 2 457 600 = 14.7456 MB`, a 2-byte pyramid
    over five levels `= 1.6368 MB` with a 4 800-byte top. All four equal the document's figures. (What
    the record **omits** is B1; the raster arithmetic itself is right.)
12. **D33 and D34 obey the seed-and-secrecy ruling.** A seed-derived rock map is refused because a
    published map is a 20× prospecting advantage; a placer bar is refused by the same rule; sand, gravel
    and clay stay lawful as bulk stock. And the price of refusing is stated honestly — the benched cliff
    reads the body's own horizontal strata instead, so (a) now costs the ore geology and not the look.
13. **D26 is right to block.** VERIFIED at `justfile:753-758`: the `linux/amd64` leg runs under the
    Mac's emulation and the file's own comment calls it a smoke test; `D-TERRAIN-1` G4 owes a real
    machine before SL10 clause 3 is called satisfied. A forty-pass accumulation over 2.46 M nodes is the
    right reason to make that purchase block the solve slice.
14. **D38 and D39 keep the crate lawful.** No thread pool inside `vd-terrain` — a new library is the
    owner's to approve — and the cache in the HOST, with the crate exposing
    `solve(seed, body) -> Artifact`. Both keep the `staticlib` linkable into another engine and keep HR5
    reachable, and D38's cost is named rather than hidden.
15. **D2's refusal of a plate simulation and of fine-grid iteration is sound.** Not addressable, a
    second shape under the ladder, four ordering drift classes and a cost in hours: four structural
    reasons, each of which SL10 or the ladder makes binding on its own.
16. **§1.5's six non-deliverables are handed over early and plainly**, including the meander arithmetic
    (a 931 m river's meander wavelength is 10.2 km, longer than one macro segment, so the D8 tree
    cannot express it) and the "no live landscape change" law. Telling the owner what he will not get,
    before he pays for the rest, is the right way round.
17. **The SL6 register is otherwise complete.** I looked for a hidden crossing and found the two named
    in B10 (the content digest) and B11 (the residency-band distance) and no other. The wind is not a
    crossing: the realm's medium is the parent's own, and the parent already computes drag from the
    child's stated mass and cross-section under the movement contract. The almanac is not a crossing:
    it reads the placements the window already carries, which is ruling S7-7 read one step further.
18. **The instrument comes before the content, and it comes for a measured reason.** §6.3 records that
    three of the last set's verdicts used `relief_bound_m` as a tolerance — 14 304.9 m, a 277 px
    tolerance against a 69.4 px signal — so a verdict **passed the very picture the work exists to
    refuse**. Building 8p first, and banning `relief_bound_m` from every verdict, is the correct order.
