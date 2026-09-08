# Slice 5 — The generator crate, with the fence

What is built, why, how it is tested, and what the owner decides. Sources: `03_generator_sl10.md`
(refuted twice, revised), `02_smooth_terrain.md` §0, §2.3, §2.4 and §5.3, `topic_00_format_sitting.md`
Parts C and D, rulings V1 (SL10), V4, V6 Part D, V8 (the load order), and THE CODE:
`crates/core/src/rng.rs` (the hash), `crates/core/src/digest.rs`, `crates/core/src/store_stamp.rs` (the
world tag today), `crates/physics/src/worldgen/` (the forest, which owns a body's radius today),
`crates/core/src/grid/` (the address and the ladder, slice 2), `crates/core/src/registry/` (the
substances, slice 3), `crates/core/src/look.rs` (`TAG_SURFACE`, slice 4). Simplified Technical English,
examples from the game.

---

## 1. What this slice decides, and what it leaves to slice 6

Slice 5 lands THE ONE GENERATOR (SL10): the Rust crate both hosts compile, which turns `(seed, address)`
into the world's static shape and nothing else. The moon's shard links it to collide; every client
links it to draw. A port is forbidden, so this crate is the only place the shape is ever computed.

```text
              (seed, address)
                    |
        +-----------v-----------+
        |      vd-terrain       |   slice 5: the body from the seed, the height field with
        |  body | height | gap  |            octave dropping, the gap lattice per rung, strata,
        |  strata | biome | band|            biome, water, carvers, the band, the digest,
        |  carvers | digest     |            the golden self-check, the declared world tag
        +-----------+-----------+
                    | a gap lattice (one i8 per cell) + substances
        +-----------v-----------+
        |  extractor + compose  |   slice 6: the surface through the lattice, the
        +-----------------------+            composition order, the triangulation rule
```

**What slice 5 outputs.** For one chunk at one rung: the substance of every cell (from the strata) and
the RADIAL GAP of every cell — `r − h`, the cell centre's radius minus the seed's height along that
direction, negative inside solid, clamped to one cell and quantised to one signed byte in steps of
1/128 cell (the registry's `GAP_STEPS_PER_CELL`, slice 3; Format B's gap byte). Water where
`h < r < sea level`. A carver's distance enters the gap as "air wins".

**What slice 6 does with it.** The extractor draws the surface through the gap lattice and hands
triangles to the physics engine on the shard and to the renderer on the client. The composition order
lays the realm's diff over the seed's lattice before either host extracts. Slice 6 is its own
discussion.

**Example.** A dirt cell on a hillside of the home moon holds `substance = dirt, gap = −0.30 cell`: "the
surface passes three tenths of a cell above my centre". The shard evaluates that cell to collide; the
client evaluates the same cell to draw. No byte crosses the wire for it.

---

## 2. What goes IN the crate, and what stays OUT

The membership test: *if two players evaluate this on two continents in two different years, must they
get the same bytes?* Yes: inside. No: outside.

| Inside (a function of `(seed, address)`) | Outside (live, or only taste) |
|---|---|
| The body definition from the seed: radius ON THE LADDER (the crate becomes the ONE owner of a body's radius; the forest reads it — ruling V6 D), sea level in whole metres, crust depth, the octave table, the strata table, the biome constants, the carver parameters | Player edits, placed blocks, sub-metre blocks, attachments (the diff, one hop) |
| The height field `h(dir, L)` with the top `L` octaves omitted, and the gap lattice per rung | Growth stage, damage, block life |
| The strata lookup (depth below `h` → a common substance); the biome field; water | Every deposit live state decides (a server-only crate and a depletion ledger, never in a client binary) |
| Carvers (tube caves as polylines) and the coarse-lattice cavern field | The edit pyramid (derived from live state) |
| The generator band per `(rung, column)`: the min and max radius the relief can reach, so an all-air chunk is refused without one noise sample | Materials, textures, colours, grass, the client's style (V2.6) — taste, not shape |
| Seed-placed feature ANCHORS and feature GEOMETRY from parameters (a tree's trunk and canopy from its kind, height, seed and a stage the DIFF supplies) — the functions land here, the trees themselves in slice 14 | The GPU vertex format, the greedy packing, the draw call |
| The chunk digest and the golden self-check; the DECLARED world tag | Anything that reads a tick, a clock, a pose or a velocity |

**The seed law, applied (ruling V4: common ore is NOT public by default).** The seed decides bulk stock
only: bedrock, loose ground, ice, water, salt, ash. Every ore, common or strategic, is live state placed
by the world, never a function of `(seed, address)`. No indicator material (a gossan, a stain) hints at
a live deposit. The registry already refuses to place an ore by hand; this slice emits none.

**Example.** A miner reads the wiki: "granite under two metres of dirt at 40° north". True, and
worthless: granite is everywhere. The wiki cannot say where the copper is, because the copper was put
there last month by the world's state, and it moves when it is mined out.

---

## 3. The fence: four layers, each with a control that must be seen failing

SL10 clause 4 says: integer hashing for every draw, a fixed evaluation order, no fast-math, no fused
multiply-add, no platform transcendental. `+ − × ÷ sqrt` are IEEE-exact everywhere and allowed.

```text
   layer 1  TYPE     Gf(f64): a private field, no From, no Deref; exactly
                     + − × ÷ Neg sqrt floor trunc abs from_i64 to_i64 to_bits.
                     NO min, NO max (they are documented non-deterministic on ±0),
                     NO mul_add (a fused multiply-add). x.0 and x.sin() do not compile.
   layer 2  LINT     crates/terrain/clippy.toml: every transcendental, mul_add, min, max,
                     f32, glam vectors, HashMap disallowed. Control: inject x.sin(), lint goes red.
   layer 3  LINK     nm -u over the staticlib: no sin|cos|tan|exp|pow|log|cbrt|fma symbol.
                     The only layer inlining cannot hide from; catches a dependency's libm.
   layer 4  GATE     the golden set: ~832 chunk digests (64 keys × 13 rungs) equal byte for
                     byte across debug/release, with and without target-cpu=native, aarch64
                     and x86-64, server build and client build. Red on one byte.
```

More rules the code needs: `f64` only (an `f32` cannot address 1 m cells past ~100 km); no `glam`
arithmetic on the path (`dot`, `normalize` have no SIMD parity promise); no runtime SIMD dispatch; no
`HashMap`; a rung is an integer on the key, never a float distance; every float compare that selects a
branch is against an integer-exact value, so a branch cannot flip on one ulp.

**The gate's legs, and their honest status.**

| Leg | Target | Runs where | Status after slice 5 |
|---|---|---|---|
| G1 | aarch64 macOS, debug and release, stable and the coverage nightly | this Mac | RAN (2026-09-08): equal |
| G2 | the same with `-C target-cpu=native` | this Mac | RAN: equal |
| G3 | aarch64 Linux, the k3d image | the image build | DID NOT RUN (Docker off) — `D-TERRAIN-1` |
| G4 | x86-64 Linux | emulation on this Mac (a smoke test: it can find a drift, never prove its absence); a real x86-64 machine before V1.3 is called satisfied (ruling V6 D) | DID NOT RUN (Docker off; no x86-64 machine) — `D-TERRAIN-1` |
| G5 | the client binary against the server binary | `vdctl gen-digest` and the shard's boot digest | OWED to slice 7, when the client links the crate |

**Which layer covers which crate (the refuter's finding 18, stated honestly).** Layer 1, the `Gf`
type, lives in `vd-terrain` only: the recipe's arithmetic is behind it, a body's fields are
crate-private (a body is DRAWN from a seed, never assembled from numbers computed elsewhere), and the
one float that enters from outside is the look radius, which the ladder snaps to an integer edge.
The leaf `vd-seed` computes the bend and the ladder in plain `f64` with `ceil` and `round` (IEEE-exact
operations `Gf` has no need to offer) under layers 2 and 3 — its own `clippy.toml` with the same
bans, and the link scan over its rlib — and layer 4 covers both crates at once, because every golden
chunk folds the bend and the ladder through the recipe.

**Example.** A contributor adds a nicer ridge curve with `powf`. Layer 1: it does not type-check.
Layer 2: the lint goes red. Layer 3: the archive names a `pow` symbol. Layer 4: the moon's chunk at
face 2, column (1181, 77), rung 0 differs by one byte between the Mac and the pod, and the merge is
refused.

---

## 4. The world tag: two halves on different carriers

```text
   declared = fnv( GENERATOR_CRATE_VERSION, universe seed )      a const; bumping it opens a world epoch
   measured = digest of 8 golden chunks evaluated AT BOOT       this binary, this chip; never a const

   store stamp ............ declared   (a chip must never cost a player's tunnels)
   mesh ALPN .............. declared   (a mixed-chip cluster must still dial)
   TAG_SURFACE ............ declared   (a client refuses ONE realm's surface, not the session)
   HelloWorld.declared .... declared   (the gateway refuses by name: WorldRefused { ours, theirs })
   HelloWorld.measured .... measured   (the gateway refuses a client whose arithmetic drifted)
```

R-10 (a mesh peer's measured profile) was NO: shards do not exchange the measured half. The build gate
(layers 1–4) is what keeps two shards' arithmetic equal; a client's measured half is checked at login
because a client binary is the one build the cluster did not make.

**Example.** A player updates the client on Tuesday; the cluster runs Monday's generator. The client
says hello, then states its world: the declared halves differ, and the refusal names both values.
Nothing is drawn wrong. A week later the moon's shard reschedules onto a pod with a different chip: the
shard opens its own store (the declared half is unchanged; the tunnels are safe) and the build gate is
what promised its arithmetic equal.

---

## 5. The coarse answer, and M-16 restated

The owner ruled (V8 addendum) that M-16 runs first: "the exact cheap coarse summary at every rung". Here
is what that must mean, because two reports state it two ways and the difference decides whether the
store is sparse.

**The problem.** At rung L an unedited cell's gap comes from `h(dir, L)`, the generator with L octaves
dropped: cheap, one evaluation with fewer octaves, and what the far view draws. The FOLD of the 8^L fine
cells under it comes from the full field, and the two differ by up to `4·k_rough·2^L` metres where nobody
dug. If the pyramid stored ABSOLUTE folds and pruned "on equality with the generator's coarse answer",
almost every coarse cell would differ, and the pyramid would be dense: an entry per coarse cell of every
touched chunk at every rung. That is R-7, "the largest unresolved requirement".

**The resolution: the pyramid stores DELTAS, never absolutes** (`02_smooth_terrain.md` §5.3). A terrain
entry holds the mean of `edited gap − seed gap` over its children, halved per rung, in exact integer
arithmetic:

```text
   delta(L+1) = (sum of the eight children's delta(L), as i16, + 8) >> 4      exact on every target

   drawn gap at rung L  =  seed gap from h(dir, L)  +  delta(L)
   prune rule           =  delta == 0 and no block occupancy (every edit reverted)
```

Under this form the generator is never asked to fold anything. It supplies ONE thing per coarse cell:
the octave-dropped gap and the strata substance at the coarse centre, one evaluation with fewer octaves.
The delta rides on top of whatever base the rung supplies, so an edited cell's rim is flat by
construction, which is also the SL8 anti-seam argument.

**So M-16, restated for slice 5:**

1. the coarse answer at rung L costs ONE evaluation with `L` fewer octaves, and tier L costs strictly
   less than tier L−1 (a gate, per rung, measured);
2. the coarse answer is byte-identical on both hosts (it is in the golden set at every rung);
3. the band property holds on the shipped octave table: `band_L ⊇ band_0` and
   `|h_L − h_0| < 4·k_rough·2^L` (U11);
4. the rim step of a delta laid over the coarse base is under the drawable floor at the rung it is drawn
   (slice 9 measures it with real entries; slice 5 states the bound).

This touches ruling C-1 ("prune on equality with the generator's coarse answer"), which becomes "prune
on a zero delta and no occupancy". For BLOCKS the entry keeps the base's occupancy, fill and dominant
substance, and its prune compares against the seed's coarse cell at one evaluation. The owner decides
(S5-3).

**Example.** A player digs a 20 m pit on a moon. At rung 0 it is 8 000 records. At rung 3 it is a few
entries that say "a hollow, four metres deep on the mean". From orbit, at rung 9, the mean delta is under
one quantum and the entry is pruned: the pit is gone from the far view, which is right, because a 20 m
pit is not visible from orbit. A quarry 2 km wide keeps an entry at rung 9 and shows as a dent.

---

## 6. Dependencies and the workspace rule

```text
   vd-seed (new, ~150 lines: SplitMix64, child_seed, fnv1a — MOVED from vd-core, re-exported there)
      ^            ^
      |            |
   vd-core     vd-terrain (new; depends on vd-seed and NOTHING else)
      ^            ^      ^
      |            |      |
   vd-wire ... vd-sim   vd-physics (the forest READS the radius; terrain never names physics — SL4)
                  ^
              vd-client (slice 7 links it)
```

- **The noise is vendored**: about 400 lines of gradient noise on our integer hash, inside `Gf`
  (ruling V6 D). The unused `noise = "=0.9.0"` pin in the workspace is deleted.
- **`glam` is not on the path.** `glam 0.30` depends on `libm`; a generator that depended on `vd-core`
  would carry that edge into the staticlib and the link scan would have to forgive it. The leaf keeps
  the archive to the generator plus 150 lines.
- **CLAUDE.md's dependency rule gains:** `core → seed; terrain → seed; sim → terrain; physics → terrain;
  client → terrain`. The crate-isolation test gains rows: `vd-terrain` names no motion crate, no clock,
  no I/O; the crossing path names no `vd-terrain`.
- **`std`, not `no_std`** (ruling V6 D): `no_std` deletes `f64::sqrt` on stable and does not stop an
  `extern "C"` libm call; the link scan is the fence either way.

---

## 7. The budget, and the measurements the slice owes

| # | Measurement | Why first | Gate |
|---|---|---|---|
| U7 + U5 | Per-chunk compose cost at EVERY rung, with the real hash (`vd_core::rng`, not the bench's proxy) | the only number today (0.885 ms per tier-0 chunk) was measured with a different hash composition; every budget rests on it | tier L strictly cheaper than tier L−1; the number is recorded per rung |
| M-16 (restated, §5) | the coarse answer's cost and its byte identity | the owner's order: before any terrain is drawn | as §5 |
| G1, G2, G4-smoke | the golden set, ~832 digests, across builds and under x86-64 emulation | no leg has ever run | zero differing bytes |
| U6 | the boot self-check (8 chunks) | it runs on every client at login | recorded; ESTIMATED ~7 ms today |
| U3 | whether clippy resolves `f64::sin` in `disallowed-methods` | decides the lint file's shape | the control goes red |
| U11 | the band and tier-agreement properties on the SHIPPED octave table | the far view and the band refusal depend on them | property tests, in the slice |

What the slice does NOT measure: the extractor's cost (slice 6, U12), the client's thread budget (slice
7, U14), the pop at a rung boundary (slice 8, U15).

**Example of the budget.** A hull descends on the home moon at 200 m/s. The client must fill about 505
tier-0 columns before the boots touch. At one millisecond a chunk that is half a second of one core;
the true number is U5's, and the client's share of it is slice 7's decision.

---

## 8. Laws

- **SL10.** One crate, two hosts; no drift as a measurement (§3); the client refusal BUILT
  (`HelloWorld` and `WorldRefused` are planted; slice 5 turns the comparison on).
- **SL5.** One world: the golden set is evaluated on the home seed (2298), never a test-only body.
- **SL4.** The generator is not a motion crate and never depends on one; the edge runs
  `physics → terrain` for the one radius.
- **SL1 clause 5.** The fence is structural (a compile-fail control, a lint, a link scan), never care.
- **SL3.** `TAG_SURFACE` is the realm stating HOW IT LOOKS; its first producer is this slice.
- **HR3.** The surface is a `ShardProfile` capability ("does this profile carry a body definition?"),
  never "is this a planet?".
- **HR5.** `vd-terrain` and `vd-seed` are Tier-A at 100 % region and branch; `Gf` is the branchless-shim
  shape HR5's discipline names.
- **The seed law (2026-08-27) and V4.** Nothing valuable is a function of `(seed, address)`; no ore, no
  indicator.

---

## 9. What you decide

| # | Question | Recommendation |
|---|---|---|
| S5-1 | The dependency shape: a `vd-seed` leaf under both (B) | Yes (ruled V6 D; restated because it moves two files out of `vd-core`) |
| S5-2 | Format D's append-only tolerance in metres (OWED since V6) | **Zero.** Any edit of the generator that moves one rung-0 byte bumps the version and opens a world epoch; the octave table is frozen with the first saved world. A tolerance would let the seed's surface slide under a saved edit's gap byte and open a rim seam of that height; the only tolerance that closes no seam is one quantum (8 mm), which no real octave fits. Detail is added on the client as style (V2.6), which never moves the shape |
| S5-3 | The pyramid stores DELTAS for terrain (§5), and C-1's prune rule becomes "zero delta and no occupancy" | Yes: it retires R-7 as a generator requirement and it is the SL8 rim argument |
| S5-4 | The golden set: ~64 keys at EVERY rung (~832 digests), literals committed in an integration test | Yes (a rung-0 pin proves nothing about rung 3) |
| S5-5 | The x86-64 leg: emulation as a smoke test now; a real machine before V1.3 is called satisfied | Yes (ruled); the slice records the emulated digests and marks the real leg OWED |
| S5-6 | The seed decides bulk stock only; every ore is live; no indicators | Yes (ruling V4) |
| S5-7 | The crate's name `vd-terrain` (the forest keeps `worldgen`) | Yes |

---

## 10. How it is built

Two parts, in order. **First the measurements** (§7: the real-hash bench at every rung, M-16 restated,
the golden legs G1/G2/G4-smoke), because the owner's order says M-16 runs before any terrain is drawn
and because every budget rests on U7. **Then the crate**: `vd-seed`, `Gf` and its three controls, the
body definition and the radius on the ladder, the height field with octave dropping, the gap lattice,
strata, biome, water, carvers, the band, the digest, the golden self-check, the declared tag on the
stamp and the ALPN and `TAG_SURFACE`, and the `HelloWorld` comparison at the gateway. One Opus 5
refuter attacks the fence's three controls, the golden set's coverage of the ladder, the tag's carriers
and every "cheap" claim, and I answer every finding before the slice is called done. Estimated size:
about 2 500 lines of code and 1 500 of tests, plus the bench.

---

## 11. Results (2026-09-08, after the refuter)

**What landed** (uncommitted until the owner's word):

```text
  crates/seed      vd-seed — THE LEAF: rng (the hash), digest, bend (the face bend, moved from core),
                   ladder (the snap, the cell radius, the face parameter, moved out of the shell grid);
                   core re-exports every one at its old path; its own float-fence clippy.toml (the
                   transcendental, fused, min/max, f32, HashMap, time, thread and network bans)
  crates/terrain   vd-terrain — THE ONE GENERATOR: gf (the fenced float, four compile-fail controls),
                   noise (gradient + value noise on ONE round of the hash), body (the definition drawn
                   from the seed: relief, octaves, sea, strata, biome, caves; the band DERIVED from the
                   relief; every field crate-private), height (the octave-dropped field, the biome with
                   the pole on +Z, the orbits' axis), strata (18 strata by key, 4 biomes, 5 bedrocks),
                   carve (the cavern field on ONE GLOBAL node lattice, the tube carvers asked over the
                   regions a tube can reach; both as detail that stops at coarse rungs), chunk (the
                   column pass shared along the radial, the cell pass, and skips that write EXACTLY
                   the cell pass's bytes), digest (the 128-bit chunk digest, the 8-key self-check that
                   REFUSES a key it cannot name), tag (the version, the declared tag, the
                   WorldIdentity), home (THE home planet's two literals, found by the forest and
                   cross-pinned in bins); the crate is a staticlib too; a `fence-control` feature
                   builds the lint control the gate watches go red
  gates            tests/terrain_pin.rs + golden_home.txt (2 592 digests: 72 columns per rung — ten
                   strides, the corner and the edge chunk of every face — × the surface chunk and
                   the chunks two below and two above it (one skip of each kind) × 12 rungs);
                   scripts/terrain_link_scan.sh with `--control` (vd-physics MUST name platform
                   math); `just gate` now runs terrain-pin, terrain-link-scan (control first) and
                   terrain-fence-control (three red lints expected); `just terrain-cost`;
                   tests/tests/voxel_pins.rs (GAP_STEPS_PER_CELL is one number in the registry and
                   the generator); crate_isolation: the leaf's whole external set is serde, the
                   generator's is empty
  wiring           the shard's self-look carries TAG_SURFACE if and only if the recipe DEFINES its
                   body (a seed AND a look radius the ladder accepts — a capability, never a kind
                   test); the gateway checks a client's world hello against its WorldIdentity, refuses
                   by name (WorldRefused names both values AND the half, the declared half first) and
                   ENDS the session; the world label folds GENERATOR_VERSION; bins: HOME_SYSTEM (cited),
                   ONE home_body rule the gateway and the bench both use, world_identity as a Result;
                   the home-body cross-pin ties the generator's literals, the forest, and the pole to
                   the orbits' z = 0 plane; DEFERRED.md `D-TERRAIN-1`; CLAUDE.md's real graph
```

**Deviations from §9, stated:**

- **The radius stays the forest's.** The generator takes the look radius as an INPUT and snaps it on
  the ladder (the home planet: 3 351 154 m → 3 350 759 m, MEASURED by `terrain_cost`, within one snap
  unit); owning the radius law (an integer table in place of the forest's power law) moves every
  body in THE world and is owed to the collider slice with its own measurement (`D-TERRAIN-1`).
- **The band is derived, not given.** The crust and the room above come from the relief the seed draws
  (plus the strata, the caves and 64 m), so the surface, every stratum and every cave fit the grid. The
  grid's `BandParams::provisional` stays for the callers that have no body definition yet
  (`D-TERRAIN-1`).
- **Caves are detail.** A tube is carved only where a cell is no wider than the tube, a cavern only
  where a cell is no wider than a quarter of its wavelength; coarse rungs draw the hill without its
  caves, as they draw it without its ripples. The cavern field is sampled on a GLOBAL 4-cell node
  lattice (a node's direction from its own face parameter) and interpolated with weights that are
  multiples of a quarter, so a cell reads the same value from whichever chunk holds it.
- **The corner hash is ONE round** of SplitMix64 over the folded corner key (the earlier three chained
  rounds cost seven times more and bought nothing); the noise pins were re-recorded before any world
  was saved.
- **Feature anchors and tree geometry** are not in this slice: they are slice 14's, and a stub would be
  a placeholder.
- **The measured half rides the client's hello only** (R-10 was NO): shards do not exchange it; the
  build gate is what keeps two shards' arithmetic equal.
- **The world label folds the recipe's VERSION** (the label already carries the seed), which is what
  a store or a peer must refuse on. CONSEQUENCE, stated: every store file and every peer built before
  this slice is refused by the new label. Nothing was saved under the old one on any cluster that
  matters, and the version stays 1, because every byte-moving fix below landed before the first save.
- **The generator's version is still 1** after the refuter's four shape fixes: no world was saved
  between the first table and the second, so no epoch opened; the golden table was re-recorded.

**Measured** (`just terrain-cost`, release, THE world's home planet: seed 7 701 581 858 760 374 086,
look radius 3 351 154 m → ladder 3 350 759 m, 12 rungs, 14 octaves, relief 14 304 m; every number
below is re-derivable: the column is face PosX, chunk (3, 5), the radial column is face PosY, rung 0,
chunk (11, 13)):

| Rung (cell) | Column pass (the coarse answer) | Surface chunk cell pass |
|---|---|---|
| 0 (1 m) | 710 µs, 14 octaves | 579 µs |
| 1 (2 m) | 665 µs, 13 octaves | 541 µs |
| 2 (4 m) | 621 µs, 12 octaves | 1 061 µs (a cave-bearing chunk: the tube list is not empty) |
| 3 (8 m) | 581 µs, 11 octaves | 407 µs |
| 6 (64 m) | 458 µs, 8 octaves | 526 µs |
| 9 (512 m) | 341 µs, 5 octaves | 526 µs |
| 11 (2 048 m) | 266 µs, 3 octaves | 411 µs |

The EXACT half of M-16 (every rung sums strictly fewer octaves than the rung below it) is a unit test
in the crate; the timing is a measurement, and the bench's own assert keeps only the wide margin (the
top rung cheaper than rung 0). One radial column at rung 0 holds 470 chunks
(`cells_in_band(0) = 29 105`, ÷ 62, rounded up); the column extremes skip 463 without a cell pass; the
whole column costs 417 ms. The boot self-check (8 chunks) costs 13.5 ms; its digest on the home planet
is `0x9331e1fdfd902272` (the first table's `0x8651…` was before the four shape fixes). Before the
rework the surface chunk cost 8.4 ms at rung 0 and coarse rungs cost MORE than fine ones.

**A fact for the owner, found by the measurements:** the home planet's sea level is 5 297 m under the
ladder radius against a relief of 14 305 m, so the sea covers about one column in a hundred (0 to 5 of
300 sampled columns per rung are under water; the deepest sampled column at rung 2 stands 489 m under
the sea). The sea level is a seed draw; whether the home planet should look like this is the owner's
call, and it is a version bump if it changes.

**The fence, each layer with its control, all re-run after the refuter:**

| Layer | Control | Result |
|---|---|---|
| Type | four `compile_fail` doc tests on `Gf` (the field, a sine, an `Into<f64>`, a fused multiply-add) | 4 pass |
| Lint | `just terrain-fence-control`: the `fence-control` feature builds a sine, a fused multiply-add and a max; clippy must go red with exactly three disallowed-method lints | red, three lints (exit 0 of the recipe) |
| Link | `scripts/terrain_link_scan.sh --control` over `vd-physics` (MUST name platform math), then the scan over the generator's and the leaf's rlibs | control hits; scan PASS |
| Gate | 2 592 digests, debug / release / `-C target-cpu=native` on aarch64 macOS (G1, G2) | equal, three legs; the tampered-table control red |

**The refuter's 24 findings, every one answered** — see `verdicts/slice_05_refutation.md`, the
answers table at its end. The four that moved bytes (F1 the skip bytes, F2 the cavern lattice edge,
F3 the tube region walls, F4 the pole) each carry a test that was red before the fix.

**Owed, by name:** `D-TERRAIN-1` in `DEFERRED.md` — G3 (the aarch64 Linux image) and G4 (x86-64) did
not run: Docker is off on this host and no x86-64 machine exists in the project; the emulated smoke
test runs when Docker is up, and the real x86-64 leg before SL10 clause 3 is called satisfied (ruling
V6 D). G5 (the client binary against the server binary) is slice 7's, when the client links the crate.
U12 (the extractor) is slice 6's.

Tests (release): seed 20, terrain 30 + 4 doc + the pin (2), core 377, wire 254, sim window lane 30,
connection-plane 103 + 4 + 7, bins 58 + the home-body cross-pin, crate isolation 6 + the voxel pins 1 —
all green; `cargo clippy --workspace --all-targets -D warnings` clean.

Full `just coverage` (2026-09-08, after the refuter): Tier-A 100 % of merged source lines and branch
sides (the first run found 12 real misses, all in `vd-terrain` — the three accessors, a short-circuit
in the octave gate, a loop that broke on its first turn — fixed with tests, then 0); io-prod
95.12 % / 95.31 % regions over the 94 floor. PRE-EXISTING RED, not this slice: `crates/bins/tests/
flight_table.rs` (two tests: the governed-row roster reads the 3.5 M-region forest against a pin of
51; the warp cruise term is off with the no-cruise legs) — recorded at slice 4 already.
