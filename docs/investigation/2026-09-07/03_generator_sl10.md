# 03 — The generator crate under SL10: one generator, two hosts, no drift

**Date:** 2026-09-07. **Revision 2** (after the law refuter and the feasibility refuter).
**Domain:** the generator crate (SL10, `owner_decisions_2026-09-07_voxels.md` V1).
**Status:** an investigation report for the owner. It designs the crate. It decides nothing by itself.
**Binding law read first:** CLAUDE.md (HR1–HR6, SL1–SL10), the 2026-09-07 voxel ruling (V1.1–V1.8, V2.1–V2.9),
the 2026-08-27 seed ruling (S1–S6, S6 superseded for the static shape only), the 2026-08-27 galaxy-shape
correction, the 2026-09-02 reach ruling (R1: the star field ships once), and SL8 (a seam is a defect).

Every number below is marked MEASURED (with how) or ESTIMATED. Every claim about the code cites `file:line`.
The investigation base (`docs/investigation/`) is NOT binding; where it disagrees with a ruling, the ruling wins.

**What revision 2 changed, in one sentence:** the SURFACE — not the cell array — is the crate's output,
because the surface is what the player stands on; and the world tag splits into a declared half that a
durable file may refuse and a measured half that only a live connection may refuse. The revision log is §13.

---

## 0. What the code holds today (the only current truth)

| Fact | Where | What it means for SL10 |
|---|---|---|
| No terrain generator exists. The only generator is the region FOREST (bodies, orbits, looks). | `crates/physics/src/worldgen/generate.rs` (1973 lines, MEASURED `wc -l`), `crates/physics/src/worldgen/body.rs:28-60` | The static-shape generator is new code. Nothing is retrofitted. |
| The forest generator calls libm: `ln`, `cos`, `sin`, `tan`, `powf`, `powi`. | `crates/physics/src/worldgen/generate.rs:571,598,633,639,807,1727,1880`; `crates/physics/src/taxonomy.rs:616,672,684,734-737,850,856` | The forest is NOT clause-4 clean. It does not have to be: its output ships once (the star field, reach R1) or is used by the parent alone. |
| **A body's DRAWN radius comes out of `powf` today, and it is the realm's own look.** | `crates/physics/src/taxonomy.rs:616` (`coeff * x.powf(exponent)`) → `composition_radius_m` (`crates/physics/src/taxonomy.rs:934,984`) → `look: Some(shell(taxon.radius_m))` (`crates/physics/src/worldgen/generate.rs:1232`) and `look: Some(Boundary::Shell { r: taxon.radius_m })` (`generate.rs:1819`); read for visibility at `crates/physics/src/worldgen/visibility.rs:85,184` | **A moon may have exactly ONE radius.** If the generator computes a second one, the terrain sits above or below the moon's own drawn edge. §1.6 names the one owner. (Corrected: the path is `crates/physics/src/taxonomy.rs`, not `worldgen/taxonomy.rs`; two files carry the name — `crates/core/src/taxonomy.rs` is the other.) |
| The client crate is engine-free. `vd-physics` is a DEV-ONLY dep. | `crates/client/Cargo.toml` (no bevy; `[dev-dependencies] vd-physics`) | The generator crate cannot live inside `vd-physics`. The shipped client never links the motion crate (SL4). |
| **`vd-client` spawns no thread and holds no `rayon`. Its clippy fence bans the clock and `thread::sleep`.** | MEASURED: `grep -rn 'std::thread\|thread::spawn\|rayon' crates/client/src/` returns nothing; `grep -n rayon Cargo.toml` returns nothing; `crates/client/clippy.toml:6-10` | "The client evaluates chunks on a worker pool" is a NEW capability in a Tier-A crate that must reach 100 % region and branch coverage. It is an open decision, not a detail (§5.6, D13). |
| The Bevy renderer is a pure consumer of `vd-client`. | `crates/client-render/Cargo.toml`; `crates/client-render/src/lib.rs:1-15` | The generator's output must reach the renderer through `vd-client`, as `MeshPrim` reaches it today (`crates/client/src/realm_scene.rs:788-795`). |
| The one integer hash is `SplitMix64` + `child_seed` + `realm_stream`. | `crates/core/src/rng.rs:22-28,70-74,82` | "No second hash" (design rule 8) means the generator uses THIS one. It lives in `vd-core`. |
| The one content digest is FNV-1a, `const`. | `crates/core/src/digest.rs:21-45` | The tag folds through it. |
| **Every SEED-GENERATED realm's frame carries its seed. Two arms do not.** | `crates/core/src/pose.rs:87-113`: six arms carry a seed (`PlanetCentered`, `SystemSpace`, `GalaxySpace`, `StationLocal`, `AreaLocal`, `StarCentered`); `ShipLocal { ship: EntityId }` carries an entity id and no seed; `UniverseSpace` is fieldless. | `(seed, address)` is already in the client's hands for a moon, a planet, a star, a station and an area. A HULL has no seed and therefore no body definition and no surface tag; its shape is its saved blocks, which arrive as a diff. (Corrected from "every realm's seed".) |
| A realm's look ships as a TLV bag (`TAG_LOOK` = 1, `TAG_LUMA` = 2, `TAG_EXTENT` = 3). | `crates/core/src/look.rs:31,35,48`; `crates/wire/src/session_flow.rs:625-640` (`BodyStmt::SelfLook`) | The bag is where a realm states "draw me from the seed at address X" (one new tag, skip-unknown). That new tag is an SL6 ask (§7). |
| **`ProtoVersion::CURRENT` is a `const`, folded at COMPILE time, and the code states that no caller may state it.** | `crates/wire/src/version.rs:355-364`; the doc at `:361-362` reads "Folded at COMPILE TIME … so no caller can state it"; `:366-373` reads "a caller that could state it could state it wrongly". `PROTO_MINOR_FLOOR = 24` (`version.rs:335`). The file's own rule at `version.rs:10-14`: "New data rides a new trailing variant, never a new field." | A digest MEASURED at boot cannot be appended to `ProtoVersion`. §3.3 corrects the carrier. |
| **`OccupantInterest` (with `coarsen_level`) is a TOMBSTONE, not a reservation.** | `crates/wire/src/intershard.rs:1174-1186`: "★TOMBSTONE payload (Step 5 slice D) … Kept only so the reserved discriminant keeps a decodable shape; nothing produces it"; the pose field is documented as "the SL2 breach that condemned it". | The coarse-detail lane is DELETED. Any coarse request the ladder needs is a NEW arm, and SL6 must approve it first. |
| The four mesh-speaking bins all fold the world tag. | `crates/bins/src/bin/gateway.rs:271,318`; `crates/bins/src/bin/orchestrator.rs:86,596`; `crates/bins/src/bin/client.rs:106`; `crates/bins/src/bin/shard.rs:279`; the fold is `crates/bins/src/lib.rs:970-972`; it rides the intershard ALPN (`crates/io-prod/src/trust.rs:41-43,209,229`). | Whatever the tag folds, EVERY one of these processes must be able to compute it. §3.3 is written to that constraint. |
| The durable stamp refuses a file on a differing world generation, and the only remedy discards the file. | `crates/core/src/store_stamp.rs:262-267`; the refusal text at `:182-192` names `VD_STORE_ALLOW_GENESIS` and says "Everything it held is discarded either way". | A refusal on the durable stamp costs a world. A refusal on a handshake costs a reconnect. The tag's two halves must ride accordingly (§3.3). |
| `noise = "=0.9.0"` is declared and UNUSED. | `Cargo.toml:78`; `Cargo.lock` has zero `name = "noise"` entries (MEASURED: `grep -c`) | Nothing is built on it. It can be deleted or used; owner's call (§10). |
| `glam = "0.30"` is a caret range; the lock holds `0.30.10`, and `glam` itself depends on `libm`. | `Cargo.toml`; `Cargo.lock:3047-3048` | The generator must not use glam on its arithmetic path. The `libm` edge under glam is one more reason the link scan (§2.2 layer 3) is the honest fence. |
| No rapier dependency exists anywhere. | MEASURED: `grep rapier Cargo.toml crates/*/Cargo.toml` → one comment at `Cargo.toml:96` | "Collision on the same shape" (V1.5) has no consumer yet. P5 builds it. The crate must expose the surface the P5 collider reads — a triangle list, not a cell array (§1.1). |
| `tier_depth` exists in the design and NOWHERE in the code. | MEASURED: `grep -rn "tier_depth" crates/` returns nothing | Every per-tier count in this report is a design quantity, not a measured one. |
| Toolchain: stable 1.94.1, no `.cargo/config.toml`, no `RUSTFLAGS`, no `target-cpu`. | `rust-toolchain.toml`; `grep target-cpu justfile scripts` → none | Every build uses the target's BASELINE features. The gate must still prove that `-C target-cpu=native` does not move a byte (§2.4). |
| Server image builds inside `rust:1.94.1-slim-bookworm`. The agent image the same. | `docker/server.Dockerfile:22-33`; `docker/agent.Dockerfile:8-15` | The k3d leg of the gate lives in these stages (§2.4). |
| This host is `arm64`, Apple M4 Pro. Docker was not running at investigation time. | MEASURED: `uname -m`, `sysctl machdep.cpu.brand_string`; `docker version` failed | The k3d image architecture is UNMEASURED today (expected linux/arm64 on this host). No x86-64 machine is listed anywhere. |
| The noise bench is a standalone workspace with ZERO dependencies. | `scripts/noisebench/Cargo.toml` (no `[dependencies]`) | Its arithmetic discipline is right (integer hash, quintic fade, no transcendental). **But its hash is NOT the repo's hash** — see the next row. |
| **The bench's `hash3` is a different COMPOSITION from `child_seed`.** | `scripts/noisebench/src/main.rs:22-35` declares `mix64` (the avalanche alone) and folds it three times with three multipliers. `crates/core/src/rng.rs:22-28` is `SplitMix64::next_u64`, which ADDS `0x9E3779B97F4A7C15` to the state before the avalanche; `rng.rs:70-74` is `child_seed`, three `SplitMix64::new(..).next_u64()` calls. | The bench's 0.885 ms is a proxy for the proposed crate, not a measurement of it. Every number derived from it is one further remove from MEASURED (§6 U5, U7). |

---

## 1. The crate boundary

### 1.1 What goes IN: everything that is a function of `(seed, address)` and of nothing else — **including the surface**

The crate is `vd-terrain` (name is the owner's; "worldgen" today names the FOREST generator in
`vd-physics`, so a distinct name avoids two things with one name — §10, D9).

**The correction that reopened this section.** Revision 1 put the mesher outside the crate and made the
crate's output a block of cell ids. Both refuters showed that this breaks V1.5. Under V2.1 the terrain is
SMOOTH. A smooth surface is not the cell array; it is the surface a rule draws THROUGH the cell array
(`docs/investigation/2026-09-07/02_smooth_terrain.md:32-40`: *"The same extracted surface is the
collider. … Nothing else is the ground."*). If the client holds that rule and the shard does not, then
two implementations make the ground, and a port into an Unreal client makes a third — which V1.2 forbids
by name. So the extractor is IN the crate.

| Inside the crate | Why it is static | Game example |
|---|---|---|
| **Body definition from the seed**: radius on the integer ladder, `tier_depth`, sea level (integer metres), crust depth `D_sat`, the octave table (amplitudes, `k_rough`, Lipschitz bounds), the strata table, the biome classifier constants, the carver parameters. | `f(seed)`, computed INSIDE the crate with clause-4 arithmetic. See §1.6: the crate becomes the ONE owner of a body's radius, and the forest READS it. | A moon's seed says: radius 161 671 m (a ladder step), sea level −120 m, seven octaves, ten strata. The moon's shard, the client and the forest all read the same row. |
| **The density field** `d(cell, L)`: a clamped signed distance to the surface in cell units, one signed byte per cell (`02_smooth_terrain.md:28-31`). | `f(seed, cell, L)`. | A dirt cell on the hillside holds "the surface passes through me, three tenths above my centre". |
| **The height field** `h(dir, L)` on the unit sphere, evaluated with the top `L` octaves omitted. The density reads it. | `f(seed, direction, L)`. | The hill the player walks to. |
| **★ THE SURFACE EXTRACTOR**: naive surface nets over cell-centred density samples — one vertex per 2×2×2 group with a sign change, at the mean of the edge crossings, in a fixed order. `+ − × ÷` only. | `f(density lattice)`. The rule that makes the ground must be one rule. | The moon's shard extracts the hillside and hands the triangles to the physics engine. The client extracts the same triangles and draws them. The boots stand on the pixels. |
| **★ THE COMPOSITION RULE**: the fixed order in which the generated density and the realm's diff combine into the composed lattice the extractor reads (§4.4). | The ORDER is static even though the diff is live. | A player mines a cell and then sets a steel foundation in the hole. Both hosts apply the mining first and the foundation second, so both get the same density in that cell. |
| **The strata lookup**: integer depth below `h` → common material. | A table read. | Stone under dirt under grass; bedrock below `D_sat`. |
| **The biome field** (temperature, humidity, elevation class → biome). | Low-frequency noise + a closed-form classifier. | A desert on the equator of the moon, tundra at its pole. |
| **Water fill**: every cell with `h < r < sea_level` is water. | An integer compare. | An ocean chunk is pure generator output: zero bytes stored, zero shipped. |
| **Carvers** (tube caves as parametric polylines) and the **cavern field** on the coarse lattice (every 4th cell, trilinear, `+ − ×` only). The carver's distance IS a density, so a tunnel mouth is round. | `f(seed, region)`. | A cave mouth in a cliff. |
| **Common-material deposits the owner declares PUBLIC** (§4, open decision D3). | `f(seed, chunk)`. | A coal seam visible in a road cut. |
| **Indicator materials**, IF the owner allows them (§4.5, open decision D11 — new in revision 2). | `f(seed, chunk)`. | A gossan on a hillside: a hint a prospector learns to read, and a row a wiki can publish. |
| **Seed-placed feature ANCHORS**: the cell, the kind and the parameter word of every seed-placed large object (V2.2), **plus the support rule that says what the anchor does when its ground is gone** (§4.6, new in revision 2). | `f(seed, chunk)`; the support rule is a pure function of the composed lattice. | "A pine, 23 m tall, seed 0x4f…, at cell (12, 40, 7)" — and, when the ridge under it is mined out, both hosts drop it by the same rule. |
| **Feature GEOMETRY from its parameters**: the trunk and canopy cells of a tree of kind K, height H, seed S, at growth stage G. | `f(kind, params, stage)`. The stage is an INPUT. The crate never knows the current stage; the diff carries it (§4). | The moon's shard calls `tree_cells(params, stage)` for collision; the client calls the same function to draw it. |
| **The generator band** `GeneratorBand { min_r, max_r }` per `(tier, column)` and the relief min/max pyramid. | Derived from the octave table. | The shard rejects an all-air chunk without evaluating one noise sample. |
| **The chunk digest** (a 128-bit integer-only mixer) over the composed density AND the extracted vertices. | Pure. | The gate's unit of comparison. |
| **The golden self-check**: a fixed list of chunk keys and their digests for `HOME_SEED` (`crates/physics/src/worldgen/census.rs:79`, MEASURED value 2298). | Pure. | Every process that must agree evaluates it and folds the result into the measured tag (§3). |

### 1.2 What stays OUT: anything live, and anything that is only taste

| Outside the crate | Where it lives | Why |
|---|---|---|
| Player edits (removed cells, placed terrain voxels, placed blocks, sub-metre blocks, attachments). | The owning realm's shard + its redb store; shipped as a DIFF one hop (V1.6). | Live state. The crate consumes a diff as an INPUT to the composition; it never owns one. |
| Growth STAGE of any feature; damage; block life. | The owning realm's shard. | Live state. |
| Deposits that live state decides (the strategic tier; V1.6 and the 2026-08-27 seed ruling S5). | A server-only crate + the realm's depletion ledger (§4.3). | Must NEVER be in the client's binary (concealed C-4, M12). |
| The edit pyramid (coarse summaries of edits). | The owning realm's shard; shipped as part of the diff for the coarse rungs. | Derived from live state. |
| **Materials, textures, colours, decoration (grass) and V2.6 client-side style** — and NOT the surface. | `vd-client` and the renderer. | Taste, not shape. Two clients may paint the hillside differently and still stand in the same place. |
| The GPU vertex format, the greedy-quad packing and the draw call. | `vd-client` + the renderer. | The crate emits positions in the realm's own frame; the packing is display. |
| The dormant-world fold (NPC mining, regrowth). | The owner shard at spin-up (economy → game, one-way). | Live state. |
| Anything that reads a tick, a clock, a pose, or a velocity. | Nowhere near this crate. | V1.1: never a function of time or state. |

**The test that decides membership:** "If two players evaluate this on two continents at two different
years, must they get the same bytes?" Yes → inside. No → outside. The extractor answers yes. The colour
of the grass answers no.

### 1.3 The dependency rule — three options for the owner

The crate needs ONE thing from the rest of the tree: the integer hash (`SplitMix64`, `child_seed`) at
`crates/core/src/rng.rs`. Design rule 8 ("no second hash") and HR3 forbid a copy.

| Option | Shape | Cost | Consequence for the Unreal static library |
|---|---|---|---|
| **A** `vd-terrain → vd-core` | The generator depends on core (serde, postcard, glam, rstar, thiserror come along). | Zero refactor. | The staticlib carries glam/rstar/serde code it never calls, and glam pulls `libm`, which the link scan then has to forgive. Larger archive (ESTIMATED: single-digit MB). |
| **B** (recommended) a leaf `vd-seed` crate holds `rng.rs` + `digest.rs` (about 150 lines, zero deps); `vd-core` and `vd-terrain` both depend on it; `vd-core` re-exports at the same path. | One small move. Every caller of `vd_core::rng` is unchanged. | The staticlib is the generator + 150 lines. The link scan sees no `libm` edge at all. |
| **C** `vd-core → vd-terrain` (the generator becomes the root leaf). | Zero refactor of callers; the generator owns the hash. | Every crate rebuilds when the generator changes. It inverts the meaning of "core". |

The dependency rule in CLAUDE.md ("bins → node → sim → wire → core") gains these lines under B:
`core → seed; terrain → seed; client → terrain; sim → terrain;` **`physics → terrain`** (new in revision 2,
for the one radius — §1.6). `vd-terrain` never depends on `vd-physics`, so SL4's fence holds in the
direction that matters: the crossing path still cannot reach a motion crate through the generator.

**Example.** The moon's shard (a `vd-sim` capability) asks the crate for chunk
`(face 2, 1181, 77, 4, tier 0)` and gets a density lattice and a triangle list. The client asks the same
crate, linked into `vd-client`, for the same key and gets the same bytes. The forest asks the same crate
for the moon's radius and puts THAT number in the moon's look shell.

### 1.4 How the two hosts call it

**Server (the realm's shard).** A new `ShardProfile` capability `voxel_surface` (HR3: a capability
config, never a shard-kind match — the profile already carries `voxel: Option<VoxelGeometry>`,
`crates/sim/src/capability.rs:107,37-42`) holds a `BodyDefinition` resolved at boot from the realm's own
seed. The physics step (P5) asks the crate for the composed tier-0 density around every occupant,
extracts the surface, and hands the triangles to the physics engine as a static mesh. Nothing else is the
ground.

**Client (`vd-client`, engine-free).** The client learns "realm R draws itself from the seed" from R's
own look bag: one new skip-unknown tag `TAG_SURFACE` beside `TAG_LOOK` (`crates/core/src/look.rs:31`),
whose payload is the realm's `FrameRef` (the seed is inside it for a seed-generated realm,
`pose.rs:87-113`) plus the crate's generator tag (§3). This is SL3 exactly: the realm authors HOW IT
LOOKS. The parent authors WHERE (the placement row), and ships nothing else. **This is an SL6 ask; it is
written out in §7.**

The client then composes and extracts chunks at the tier the screen and the closing speed need (§5),
converts the triangles to `MeshPrim`s in `vd-client`, and hands them to the renderer through the seam
that exists today (`realm_scene.rs:788-795`, whose doc line 789 already reserves this).

**A dormant realm is still never drawn, and the design should say so rather than get it by luck.** The
surface tag rides `BodyStmt::SelfLook`, which only a RUNNING realm may send (`crates/wire/src/session_flow.rs:626-630`:
"only a running realm can ship one"). So the seed alone never lets the client draw a sleeping moon. The
2026-09-01 visibility ruling is kept by the carrier, and that is stated here on purpose.

**Example.** A hull descends toward a moon. The moon's shard runs (reach woke it) and ships its placement
row and its look bag. The bag says `TAG_SURFACE { frame: PlanetCentered { planet_seed }, gen_tag }`. The
client compares `gen_tag` with its own; equal, so it starts composing coarse chunks of that seed while
the hull is still 100 km out. No chunk crosses the wire.

### 1.5 The Unreal client: the C interface (V1.2)

The rule: the SAME Rust crate, compiled to a static library per target, called through `extern "C"`. No
port. **Because the extractor is inside the crate (§1.1), the C surface must return a SURFACE, not only
cell ids.** Revision 1's signature could not, and the "476 KB of cell ids" it quoted is withdrawn.

| Option | Header | New dependency? | Note |
|---|---|---|---|
| **H1** (recommended now) hand-written `vd_terrain.h`, beside a thin `vd-terrain-ffi` crate (`crate-type = ["staticlib", "cdylib"]`). A unit test asserts a size/alignment check on every `#[repr(C)]` struct. | ~180 lines, by hand. | None. | The surface is about a dozen functions. |
| **H2** `cbindgen` as a `build-dependency`. | Generated on every build. | YES: a new crate in the product graph (owner decision by standing rule). | Drift-proof; adds a build-time tree. |
| **H3** `cbindgen` as a developer TOOL (`cargo install`, never a Cargo dep); the header is committed; a gate regenerates and diffs it. | Committed. | No Cargo dep; a tool install. | Same drift-proofing without touching the product graph. |

**The ffi crate is separate on purpose.** The generator crate is `#![forbid(unsafe_code)]`; only the ffi
crate holds the `unsafe` pointer edge, and it holds NO arithmetic. HR5 Tier-A covers the generator; the
ffi crate is Tier-B (like `io-prod`), proven by a link test.

**The function set** (all buffers caller-owned; no allocation crosses; no panic crosses —
`catch_unwind` at every entry returns an error code):

```c
uint32_t vd_terrain_abi_version(void);
int32_t  vd_terrain_tag(uint64_t seed, VdGenTag* out);                  // §3
int32_t  vd_terrain_body(uint64_t seed, VdBodyDef* out);                // radius step, tier_depth, sea level, D_sat ...
int32_t  vd_terrain_band(const VdBodyDef*, VdColumnKey, uint8_t tier, VdBand* out);
int32_t  vd_terrain_cells(const VdBodyDef*, VdChunkKey, uint8_t tier,   // identity + density, per cell
                          VdCellRecord* cells, size_t cells_len);
int32_t  vd_terrain_compose(const VdCellRecord* generated, size_t n,    // §4.4: the ONE fixed order
                            const VdDiffRecord* diff, size_t diff_n,
                            VdCellRecord* out, size_t out_len);
int32_t  vd_terrain_surface(const VdCellRecord* composed, size_t n,     // ★ the ground itself
                            VdVertex* verts, size_t verts_cap, size_t* verts_written,
                            uint32_t* indices, size_t idx_cap, size_t* idx_written);
int32_t  vd_terrain_features(const VdBodyDef*, VdChunkKey, VdFeature* out, size_t cap, size_t* written);
int32_t  vd_terrain_feature_cells(const VdFeature*, uint8_t stage, VdCell* out, size_t cap, size_t* written);
int32_t  vd_terrain_chunk_digest(const VdBodyDef*, VdChunkKey, uint8_t tier, uint8_t out[16]);
```

**The chunk payload size, corrected.** 62³ = 238 328 cells (MEASURED, arithmetic). A `uint16_t` per cell
carries the block identity ALONE, and the identity is not the shape. Report 02 needs an 8-bit signed
density on EVERY planet cell beside a 16-bit identity and a 6-bit orientation
(`02_smooth_terrain.md:181-186`). So the honest sizes are:

| Packing | Bytes per cell | Bytes per chunk | Mark |
|---|---|---|---|
| identity only (revision 1's number, **withdrawn**) | 2 | 476 656 (~476 KB) | wrong payload |
| identity + density, packed to 3 bytes | 3 | 714 984 (~715 KB) | ESTIMATED, arithmetic |
| identity + density + orientation in a 4-byte word | 4 | 953 312 (~953 KB) | ESTIMATED, arithmetic |

The extracted surface is a SECOND buffer, and it is the one the collider reads. Its size is UNMEASURED
(§6 U12). **The cell record's exact width is not this domain's to freeze**; reports 01, 02 and 04 own it,
and the C signature above must follow their answer, not lead it. That ordering is a one-way door (§9).

One static library per shipped target: `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`,
`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`. The Bevy client does not use the C surface; it
links the Rust crate directly. Both paths call the same functions, so the gate (§2.4) covers both.

**Example.** The Unreal client on a Windows x86-64 machine calls `vd_terrain_surface` for the hill's
chunk. The moon's shard on a Linux arm64 pod calls the Rust function for the same key. When the gate of
§2.4 has RUN and printed two equal digests, the boots will stand where the pixels are. It has not run
yet (§2.4, §6 U1).

### 1.6 The one owner of a body's radius (new in revision 2)

**The defect revision 1 left open.** The report told the generator to compute a body's radius inside the
crate and never to import it. It never said what happens to the radius the forest already computes and
already SHIPS as the body's look (`generate.rs:1232`, `:1819`). Two radii for one moon is a second
implementation of one world fact. SL5 forbids it, and the physical consequence is a seam: the terrain
pokes through the moon's own drawn edge, or floats above it.

**The rule.** A body that has a voxel surface has ONE radius, and the generator crate owns it:

1. `vd_terrain::body_definition(seed).radius_m` is the number.
2. The forest READS it (`vd-physics → vd-terrain`, §1.3) and puts it in the look shell and in the
   containment bound. `composition_radius_m` stops deciding the shipped radius for such bodies.
3. The astrophysics that `taxonomy.rs` expresses with `powf` must be re-expressed inside the crate with
   clause-4 arithmetic — a piecewise polynomial or an integer table on the mass ladder. **This is real
   work and a real loss of fidelity, and it is an owner decision (§10, D12).**
4. A body with NO voxel surface (a star, a gas giant with no ground) keeps the forest's radius. The two
   ladders never meet, because such a body has no `BodyDefinition`.

**The migration is a world epoch.** Every existing body's radius moves the day this lands. The crate
version bump (§9) must happen in the same commit, so every saved store is refused rather than opened
against a moon that changed size.

**Example.** Today the moon's look says "shell, r = 161 671.4 m" from a `powf`. After this change the
moon's look says "shell, r = 161 671 m" — a step on the crate's integer ladder — and the hillside the
player walks on ends exactly there. One number, one moon.

---

## 2. Determinism, concretely

### 2.1 V1 clause 4 as code rules

| Rule (V1.4) | Code rule | Where it holds |
|---|---|---|
| Integer hashing for every random draw. | Every draw is `SplitMix64` / `child_seed` on `(seed, lattice ints)`. No float ever feeds a hash. The planet's identity enters through the gradient hash, never through a coordinate offset. | The crate. `noisebench` shows the discipline but not the composition (§0, last row). |
| Fixed evaluation order. | Octaves summed low → high in one loop; tier `L` sums the same prefix in the same order. The extractor sums a group's edge crossings in a fixed corner order. No parallel REDUCTION inside the crate; chunk-level parallelism happens outside (order-independent). Fixed iteration counts, like `KEPLER_FIXED_ITERS` in `celestial.rs:24`. | The crate. |
| No fast-math flag. | Stable Rust has no fast-math flag; the `fadd_fast` intrinsics are nightly-only. The rule is a GATE (§2.4), not a promise. | Gate. |
| No FMA contraction. | Rust never contracts `a*b + c`. The crate exposes NO `mul_add`. | `Gf` surface (§2.2) + lint. |
| No libm transcendental. | No `sin cos tan asin acos atan atan2 exp exp2 ln log2 log10 powf powi cbrt hypot sinh cosh tanh to_radians to_degrees`. Curves are polynomials (the quintic fade) or integer tables. | `Gf` + lint + link scan (§2.2). |
| `+ − × ÷ sqrt` allowed. | Plus `floor`, `trunc`, casts, `abs`, comparisons. `sqrt` is IEEE-exact (correctly rounded) on every target. | `Gf` surface. |

**★ `min` and `max` are REMOVED from the allowed list (corrected in revision 2).** Rust's standard
library documents `f64::min` and `f64::max` with this note: if the inputs compare equal — the case of
`+0.0` against `−0.0` — either input may be returned non-deterministically. It is the one operation on
revision 1's list whose result is not fixed by IEEE-754 alone, and §2.4's representativeness argument
rests on the list being fully determined. `Gf` therefore has no `min` and no `max`. Write
`if a < b { a } else { b }`, which the comparison fully determines. A waterline chunk and a sea-level
clamp column join the golden set.

> **DISPUTED (scope, not the fix): the feasibility refuter's F5 example is wider than the hazard.**
> The refuter writes that `height.max(sea_level)` on a beach cell "decides which of two identical-looking
> values flows into the cell id". When two f64 values compare equal they are the same bits, EXCEPT for
> `+0.0` against `−0.0`. So the only case that can change an output byte is a signed zero, which a height
> field can produce when a difference cancels exactly at sea level. The fix is right and cheap, so the
> report adopts it in full. The claim that a generator "meets this case constantly" is ESTIMATED and
> UNMEASURED; the honest statement is that it meets it rarely and unpredictably, which is worse for a
> gate, not better.

Additional rules that V1.4 does not name and that the code needs:

- **f64 only.** No `f32` anywhere in the crate (an f32 cannot address distinct 1 m cells above ~100 km radius).
- **No `glam` arithmetic** on the path (`dot`, `length`, `normalize` have no SIMD/scalar parity promise). Write `v / sqrt(v·v)` in scalars.
- **No runtime SIMD dispatch**: no crate that picks SSE/AVX/NEON paths at run time.
- **No `HashMap`** (the existing sim ban, `crates/core/clippy.toml`): iteration order is arithmetic order.
- **A tier is an integer on the key.** The octave count comes from `L`, never from a float distance.
- **Every float compare that selects a BRANCH is against an integer-exact value** (a power of two, a table entry), so the branch cannot flip on an ulp.

### 2.2 Enforcement — four layers, each one measurable

1. **Type-level (the fence).** A newtype `Gf(f64)` with exactly `Add Sub Mul Div Neg sqrt floor trunc abs from_i64 to_i64 to_bits`. **The inner `f64` field is PRIVATE, the crate exposes no `From<Gf> for f64` and no `Deref`, and the type is `#[repr(transparent)]` only for the ffi edge.** Without those three statements a newtype is a habit, not a fence — the law refuter is right that revision 1 asserted "structural" without saying what made it so. The first commit adds an OBSERVED-FAILING control in the shape `tests/tests/crate_isolation.rs` already uses: a `trybuild`-style compile-fail case that reaches for `x.0` and for `x.sin()` and must fail to compile. The idiom is the same as `EffectFree` in `crates/sim/src/coupling.rs`.

2. **Lint-level (the tripwire).** A crate-scoped `crates/terrain/clippy.toml`, in the exact shape of the existing I/O ban (`crates/core/clippy.toml:1-6`, `crates/sim/clippy.toml`, `crates/client/clippy.toml:6-10`): `disallowed-methods` for every f64/f32 transcendental, for `mul_add`, and for `f64::min`/`f64::max`; `disallowed-types` for `f32`, `glam::Vec3A`, `std::collections::HashMap`. **UNMEASURED:** whether clippy resolves primitive inherent paths such as `f64::sin` in `disallowed-methods` (U3). The SEAL-1 control decides it: inject `x.sin()`, expect a red `just lint`. If clippy cannot name a primitive method, the fallback tripwire is a 30-line test that scans the crate's own sources for the banned identifiers.

3. **Link-level (the proof that survives inlining).** Build the crate as a staticlib and scan its undefined symbols: `nm -u target/…/libvd_terrain.a | grep -E ' _?(sin|cos|tan|exp|pow|log|cbrt|fma)'` must be empty. This catches a transcendental that arrived through a dependency, which no lint sees — and `glam 0.30.10` depending on `libm` (§0) shows the edge is real. `nm` ships with Xcode and with the Debian builder image; no new dependency. **This is the strongest fence, and it stays whatever the owner decides about `no_std`.**

   > **`no_std` is NOT the compile-error fence revision 1 claimed (corrected).** Two things are wrong
   > with that claim. First, `no_std` removes the std INHERENT METHODS; it does not remove the C symbols,
   > and a crate under `no_std` may still declare `extern "C" { fn sin(x: f64) -> f64; }` and call it.
   > Second, on stable Rust `f64::sqrt` and `f64::floor` live in `std`, not `core` (the `core` float-math
   > methods sit behind an unstable feature). V1.4 ALLOWS `sqrt` by name. So `no_std` would delete an
   > operation the law grants, and the cure would be either a nightly toolchain (the project is pinned
   > stable 1.94.1, `rust-toolchain.toml`) or the `libm` CRATE — a NEW dependency, which the standing
   > rule says must be offered to the owner and never adopted silently. Both facts stay UNMEASURED here
   > (U4); D6 now recommends **No** until they are measured.

4. **Gate-level (the measurement, §2.4).** Byte-for-byte equality of the composed density AND the
   extracted vertices across builds and targets. This is the only layer that is a MEASUREMENT in the
   sense of the "never assume" rule; layers 1–3 make it pass by construction, the gate proves it passed.

**Example.** A contributor adds a "nicer" ridge curve with `powf`. Layer 1: it does not type-check on
`Gf`, and it cannot reach the inner `f64` to escape. Layer 2 goes red under `just lint`. Layer 3 finds
the `pow` symbol in the archive. Layer 4 shows the moon's chunk 0x3a… differs by one byte between the
Mac build and the pod build, and the merge is refused.

### 2.3 The chunk digest and the golden set

- `chunk_digest(body, key, tier) -> u128`: an integer-only mixer over the composed cell records **and the extracted vertex bits** in a fixed order. FNV-1a from `vd_core::digest` folded twice is enough — one definition, `digest.rs:21-45`. **The vertices are in the digest because the vertices are the ground.** Revision 1 digested cell ids only, and a gate that digests cell ids is green while two hosts stand on two different slopes.
- **The golden set:** for `HOME_SEED` (2298, `census.rs:79`), ~64 chunk keys spanning all 6 cube faces, 12 edges, 8 corners, the band floor and ceiling, **a waterline column and a sea-level clamp column** (new, §2.1), a carver region, **and a COMPOSED chunk (generated + one mined cell + one placed block, §4.4)** — each at EVERY legal tier.
- **The size, corrected.** Revision 1 wrote "64 × tier_depth" in one place and costed the gate at "64 × ~8 tiers" in another. `tier_depth` is about 13 on the starter world (`block_provenance_collapse.md:953`, a design quantity — `tier_depth` does not exist in the code, §0). So the set is about **832 digests**, not 512 (MEASURED: arithmetic on the report's own two sentences). Committed literals in `crates/terrain/tests/terrain_pin.rs` (an integration test, so an exactness assertion can never turn the HR5 gate red — the coverage recipe excludes `/tests/`, `justfile:31-33`).
- **The boot self-check:** 8 of those chunks, evaluated at boot, folded into the MEASURED half of the tag (§3.3). ESTIMATED cost: 8 × 0.9 ms ≈ 7 ms on the M4 Pro — **and that estimate rests on a bench that used a different hash composition (§0, last row), so it is ESTIMATED at one further remove and must be re-measured (U7).** Extraction cost is not in it at all (U12).

### 2.4 The gate: byte-for-byte across builds and across x86-64 and aarch64

**No leg of this gate has run. Nothing below is a result.** Revision 1's §1.5 example said "the gate
proved"; that sentence is struck, and §12 now marks the x86-64 leg OWED.

| Leg | Target | Where it runs | How | Status |
|---|---|---|---|---|
| G1 | `aarch64-apple-darwin`, debug + release, stable 1.94.1 + the coverage nightly | This Mac | `cargo test -p vd-terrain --test terrain_pin` and `--release`; `just terrain-pin` runs both toolchains. | Buildable when the crate exists. NOT RUN. |
| G2 | `aarch64-apple-darwin` with `-C target-cpu=native` | This Mac | Same test with `RUSTFLAGS`. Proves the profile does not move with CPU features. | NOT RUN. |
| G3 | `aarch64-unknown-linux-gnu` (the k3d image, expected) | The server image builder stage (`docker/server.Dockerfile:22-33`) | Add `RUN cargo test --release -p vd-terrain --test terrain_pin`; `just image-build` fails on drift. The shard logs its boot self-check digest; `just k3d-dod` compares it with the host's literal. | Docker was not running at investigation time (MEASURED). Image architecture UNMEASURED. |
| G4 | `x86_64-unknown-linux-gnu` | No x86-64 machine exists in this project. | (a) `docker build --platform linux/amd64` on this Mac, under Rosetta or QEMU. (b) an x86-64 cloud runner — "no CI yet"; owner's call. (c) a contributor's x86 laptop running `just terrain-pin`, digests pasted into the gate log. | OWED. |
| G5 | The CLIENT build vs the SERVER build (V1.3 names both) | This Mac + the pod | The client bin gains `vdctl gen-digest` (HR6: agent-operable); the shard prints its boot digest; a gate compares the two strings. Feature unification and thin LTO differ per binary; only the bins prove the bins. | OWED with the first voxel slice. |
| G6 | `x86_64-pc-windows-msvc` (an Unreal client target) | None here. | Same as G4 (b)/(c). | OWED before the Unreal client links. |

**★ An emulated green is NOT "no drift on x86-64" (corrected).** Revision 1 wrote "Rosetta executes
x86-64 SSE arithmetic exactly" as a fact and let D7 rest on it. That sentence is withdrawn. Subnormal
handling and the flush-to-zero and denormal-as-zero flags are the classic divergence between a real x86
part and a translation layer, and a height field that divides two nearly-equal metres can produce a
subnormal. **G4(a) is necessary and not sufficient: it can FIND a difference and can never prove its
absence.** V1.3 says "every target the game ships on", and a shipped target is real hardware. V1.3 stays
OPEN until one real x86-64 machine prints an equal digest. That is now a one-way door (§9).

**Red on one differing byte** (V1.3): the test compares 128-bit digests; the live handshake refuses a
peer whose measured tag differs (§3.3). Both are exact equality; neither has a tolerance.

**What the gate costs, ESTIMATED:** G1+G2 about 832 digests per leg; at ~1 ms per chunk that is under a
second of evaluation, plus extraction, which is UNMEASURED (U12). G3 adds the crate's test to an image
build that already takes minutes. G4 under emulation is the unknown; measure it first (§11).

**What the gate does NOT prove, said plainly:** it proves the golden set, not every chunk in the galaxy.
Layers 1–3 (§2.2) are what make the golden set representative: if no operation outside the fully
determined list exists on the path, a chunk that agrees proves the arithmetic agrees, and every other
chunk is the same arithmetic on other integers. Two things could break that argument, and both are now
closed by rule: an operation whose result IEEE-754 does not fix (hence `min`/`max` are gone, §2.1), and
a branch on a float that is not integer-exact (hence §2.1's last rule).

---

## 3. The world-generation tag

### 3.1 What it names NOW (code)

- `StoreStamp.world_generation: u64` — "THE WORLD-LAW GENERATION — folded over the named constants that shape the forest" (`crates/core/src/store_stamp.rs:117-122`); computed by `world_generation(&[f64])` as an FNV fold over each constant's bit pattern (`store_stamp.rs:301-307`).
- The constants are seven forest-shape numbers (`crates/physics/src/worldgen/scale.rs:535-545`); a test proves every one of them moves the label (`worldgen/tests.rs:4967-4997`).
- `vd_bins::world_generation()` folds exactly that list (`crates/bins/src/lib.rs:970-972`).
- **It refuses two things:** (1) a durable FILE whose stamp disagrees (`store_stamp.rs:262-267`), whose only remedy DISCARDS the file (`store_stamp.rs:182-192`); and (2) a PEER shard/gateway/orchestrator whose intershard ALPN disagrees (`crates/io-prod/src/trust.rs:41-43,209,229`; `crates/wire/src/admin.rs:373-376`).
- **It does NOT refuse a player client.** The client handshake carries `ProtoVersion { major, minor, coordinate_generation }` only (`crates/wire/src/version.rs:337-351`); `negotiate` refuses on major, the minor floor and the coordinate generation (`version.rs:386-405`). `ClientControlMsg::Hello { version, login }` (`crates/wire/src/channels.rs:44-49`) carries nothing else. **So V1.3's sentence "the world-generation tag that already refuses a client whose generator disagrees" describes the shard-to-shard and file case, not the player client.** This is a stale claim to correct (§8).

### 3.2 What it must name under SL10

Two parts, and they must ride different carriers, because a refusal costs a different thing in each place.

```text
declared_gen = fnv( GENERATOR_CRATE_VERSION,     // a u32 const the crate exports; bumped by hand on any output-changing edit
                    universe_seed )              // the world (SL5: one world; the seed still names it)
world_generation' = fnv( world_generation(shape_constants), declared_gen )

measured_profile = GOLDEN_SELF_CHECK_DIGEST      // 8 chunks evaluated AT BOOT on this binary, this CPU
```

- **The declared part** is the human declaration and the epoch counter. Bumping it opens a world epoch (§9). It is a `const`. Every process can compute it with zero work.
- **The measured part** is the arithmetic profile, MEASURED and not declared, because a declared profile can lie (a build with `-C target-cpu=native`, a dependency with runtime dispatch). D8 keeps this recommendation.

### 3.3 Where each part rides (rewritten in revision 2)

Revision 1 folded the measured digest into ONE value and put that value on every carrier. Both refuters
showed the consequences, and both are severe.

| Carrier | Today | Revision 1 (withdrawn) | **Revision 2** |
|---|---|---|---|
| Durable store stamp | `world_generation` | `world_generation'` including the measured digest | **`world_generation'` — the DECLARED part only.** A store must never be refused because a pod landed on a different chip. `store_stamp.rs:182-192` says the only remedy discards the file, and a saved diff is a player's tunnels. A build difference must cost a reconnect, never a world. |
| Intershard ALPN | `world_generation` | `world_generation'` including the measured digest | **`world_generation'` — the DECLARED part only.** The measured digest on the ALPN would silently forbid a mixed-architecture cluster (an arm64 pod could never dial an x86-64 pod), and it would force the ORCHESTRATOR — which draws nothing — to evaluate eight terrain chunks at boot. |
| A new admin-lane statement, right after the mesh handshake | absent | absent | **The MEASURED profile.** Each mesh peer states its measured digest; a mismatch is a loud, typed refusal of that PEER, reversible by a redeploy. This is a new wire arm and therefore an SL6 ask (§7). |
| Client Hello | absent | append `world_generation` to `ProtoVersion` (D4 (a)) | **A new trailing `ClientControlMsg::HelloWorld { declared, measured }`, sent immediately after `Hello` (D4 (b)).** See the box below. |
| The realm's look bag | absent | `TAG_SURFACE` carries the tag | **`TAG_SURFACE` carries the DECLARED part**, so a client can refuse ONE realm's surface without dropping the session. |

> **★ `ProtoVersion` cannot take the field (corrected).** `ProtoVersion::CURRENT` is a `const` whose
> `coordinate_generation` is folded at COMPILE time, and the code says why in its own words: *"Folded at
> COMPILE TIME … so no caller can state it"* and *"a caller that could state it could state it wrongly,
> and the one value this negotiation exists to protect would become the one value a test could fake"*
> (`crates/wire/src/version.rs:355-373`). A digest MEASURED at boot is not a `const`. Appending it forces
> `CURRENT` and `const fn speaking()` to stop being `const` and makes the value caller-stateable — which
> destroys the exact property that file was written to hold. The file's own rule (`version.rs:10-14`)
> is *"New data rides a new trailing variant, never a new field."* So D4 flips to **(b)**, and the
> `PROTO_MINOR_FLOOR` move (24 today, `version.rs:335`) is not needed.
>
> A second cost that (b) also avoids: a client that must evaluate eight chunks before it may send
> `Hello` puts terrain work in front of the login handshake. Under (b) the client says `Hello` at once
> and states its measured profile in the next message, while the connection is already up.

> **★ Every mesh-speaking node links the generator (corrected).** Revision 1 wrote that the storeless
> gateway "skips" the self-check. The tag on the mesh handshake is ONE value for every node kind, and
> four bins fold it (`gateway.rs:271,318`; `orchestrator.rs:86,596`; `client.rs:106`; `shard.rs:279`).
> A node that skipped the check would fold a different number and refuse every peer, and nobody would
> see the moon at all. Under revision 2 the ALPN carries the DECLARED part, which is a `const` read: the
> orchestrator links the crate, pays no boot cost, and dead-strip removes the arithmetic it never calls.
> Only a node that must AGREE ON A SURFACE — a shard with a `voxel_surface` capability, and a client —
> runs the eight-chunk self-check and states it on the admin lane.

**Example.** A player updates the client on Tuesday; the cluster is still on Monday's generator. The
client sends `Hello`, then `HelloWorld`. The declared parts differ, and the refusal names both values
("this build's world generation is …; the cluster's is …"), exactly as the coordinate-unit refusal does
today (`version.rs:415-430`). Nothing is drawn wrong. A week later the cluster reschedules the moon's
shard onto a node with a different chip: the shard opens its own redb file (the declared part is
unchanged, the tunnels are safe) and the mesh refuses that shard's measured profile until the image is
rebuilt. The tunnels are never at risk from a chip.

---

## 4. The diff lane split

### 4.1 The seed decides (shipped as NOTHING)

The terrain surface at every tier, as a density field AND the extracted triangles; strata (stone, dirt,
sand, gravel, clay, bedrock); water at sea level; ice; biome; caves and caverns; seed-placed feature
anchors and their parameter words; feature geometry as a function of parameters and a GIVEN stage; the
common-material deposits the owner declares public (open decision D3); indicator materials IF the owner
allows them (open decision D11).

### 4.2 Live state decides (shipped as a DIFF, one hop, from the owning realm)

| Diff content | Shape on the wire | Game example |
|---|---|---|
| Cell edits: removed cells, placed terrain voxels (V2.1: the surface reshapes around them), each carrying the cell's new DENSITY. | The record family of report 01/04, per chunk, in a TLV-framed body (`crates/core/src/tlv.rs`, skip-unknown). | A tunnel dug last week arrives and is cut into the derived hill; the tunnel mouth is round, not cubic. |
| Placed blocks, ~20 shapes, sub-metre blocks inside a 1 m cell, no-volume attachments. | Same record family; sub-metre and attachment payloads are their own tags. | A HUD on a hull wall; a piston. |
| Growth stage per feature (one byte, sparse). | `(feature id, stage)`. | A sapling planted on Monday is a tree on Friday; the client calls `feature_cells(params, stage=4)`. |
| Damage / block life. | Sparse. | A cracked wall. |
| The edit pyramid for the coarse rungs. | Per coarse cell: fill count + dominant material. | The tunnel stays visible from a hilltop at tier 3. |
| The ASSAY result of a mined cell (what the rock yielded). | The item grant on the edit acknowledgement — ALREADY a message. | A player breaks grey rock and receives titanium ore. The rock was drawn as rock. |

**What is NOT in either lane:** the strategic deposit's POSITION. It is never in the grid (R11; concealed
C-4 "a drop table, not an overlay"). No pixel, no byte, no reveal radius, no timing signal.

### 4.3 Where the deposit-by-live-state is decided and stored

- **Decided:** in the OWNING realm's shard, by a server-only crate (`vd-assay`, name open) that is ABSENT from the client's dependency closure — a `cargo metadata` test asserts it (concealed C-10). Input: `(realm_key, coarse region, live ledger)`. `realm_key = PRF(master, RealmUid)` with HMAC-SHA256, whose crates are already declared (`Cargo.toml`: `hmac = "0.12"`, `sha2 = "0.10"`). **Never `SplitMix64`/`child_seed` for this** — one observed output recovers the master secret (concealed_resources.md, "THE TRAP").
- **The live half** (2026-08-27 S5: "anything that pays must read world state that MOVES"): the drop table reads a per-region DEPLETION LEDGER — what players extracted, what the dormant fold says NPC life consumed. The ledger is an ordinary durable row in the realm's own store (one writer: the realm head).
- **Stored:** the ledger only. Deposits are regenerated on demand and never persisted; a mined cell is an ordinary edit resolved BEFORE any table lookup, so a key rotation cannot move ore under an existing shaft.
- **Crosses a realm boundary:** nothing.

**Example.** A miner on the moon breaks cell (12, 40, 7). The moon's shard: (1) records the edit and
ships the diff to every window over that chunk; (2) asks `vd-assay` with the moon's key, the region and
the ledger row; (3) the answer is "3 units of iridium"; (4) the ledger's extracted count rises; (5) the
item grant rides the acknowledgement. No client ever computed step (2).

### 4.4 ★ The composition order (new in revision 2)

**The gap.** Revision 1 fenced the GENERATOR and never fenced the composition `generated ⊕ diff`. Under
V2.1 a placed terrain voxel reshapes the surface, and the density survives UNDER a placed square block
and returns unchanged when the block is broken (`02_smooth_terrain.md:45-49`). So the COMPOSED density
decides both the picture and the collider, and that composition runs on two hosts. An unstated order is
a drift the golden set does not see.

**The rule.** The composition is one function in the crate (`vd_terrain_compose`, §1.5), and its order is
fixed and stated:

1. The generated density and identity for the cell (`f(seed, cell, tier)`).
2. Terrain cell edits, in the diff's own record order (removals, then placements of terrain voxels).
3. Placed catalogue blocks (the ~20 square shapes) — they set the cell's FORM and hide the density; they never erase it.
4. Sub-metre blocks inside the cell.
5. No-volume attachments (they change nothing in the lattice).

**The gate.** A COMPOSED chunk joins the golden set: the generated shape, one mined cell, one placed
block, one sub-metre block, digested after composition and after extraction. Without that row the byte
gate is green while two hosts stand on two slopes.

**Example.** A player mines a cell on the moon and then sets a steel foundation into the hole. The client
applies the mining first and the foundation second. The moon's shard applies them in the same order and
gets the same density. When the player later breaks the foundation, the slope is exactly what it was.

### 4.5 ★ Indicator materials are an owner decision, not a free feature (new in revision 2)

Revision 1 put indicator materials inside the crate without opening a decision. The 2026-08-27 seed
ruling says anything a fixed seed alone decides must be SAFE TO PUBLISH, because one world plus many
players makes any static map public by wiki. An indicator whose factor is DECLARED and whose position is
`f(seed, chunk)` is a static ranking of where to prospect, and it ships in every client binary.

*Game example.* A wiki lists every gossan on the moon by chunk key. Every prospector flies the same list,
and the survey instrument the concealment ruling paid for stops being worth carrying.

The decision is D11 (§10).

### 4.6 ★ A feature anchor whose ground was mined (new in revision 2)

**The gap.** The anchor is `f(seed, chunk)` and the growth stage is a diff. Neither says what happens when
the cell the anchor sits on is air because somebody dug it out. Two hosts must agree, or the pine's
collider and the pine's picture separate.

**The rule this report recommends (owner may overrule).** The support test is a PURE function of the
COMPOSED lattice, so both hosts compute it and neither ships it:

- If the anchor's support cell is solid in the composed lattice, the feature stands.
- If it is air, the feature is CULLED — it is not drawn and not collided — and nothing is stored.
- A feature is never moved, never made to fall, and never becomes live state, because a falling tree is
  a physics event and this crate may not know time (V1.1).

The consequence is honest and small: mining under a pine deletes the pine for everyone, the same way for
everyone, with zero bytes. A tree that should FALL is a live feature and belongs to the shard's diff
lane, not to the generator (§10 has no decision for this; it is a design statement the owner may reject).

**Example.** A player mines the ridge under a pine on the moon. The client recomposes the chunk, the
support cell is air, the pine vanishes from the picture. The moon's shard recomposes the same chunk, gets
air, and drops the pine's collider in the same tick.

---

## 5. Detail-by-octave: the client picks its own level, the server keeps the full set

### 5.1 The rule

- `h(dir, L)` sums the first `n − L` octaves in the same order tier 0 sums its first `n − L` terms. Tier `L` is an INTEGER on the chunk key. The surviving partial sum is bit-identical to a prefix of the tier-0 sum. Carvers contribute at tier `L` iff `radius_m ≥ 2^L` (an integer compare).
- **The server composes and extracts tier 0 for collision, always.** The character controller, the sweep test and the collider never see a coarse chunk; the type signature says so (the collider builder takes a `Tier0Surface`, not a `Surface`).
- **The server may evaluate coarse tiers for its OWN bookkeeping** (the band, the relief pyramid, the AoI cull), never for a fact a player can act on.
- **Where it matters, the two agree:** every cell a player can touch is drawn at tier 0 on the client and collided at tier 0 on the server — the same bytes from the same crate, through the same extractor.

### 5.2 ★ The tier choice reads the closing speed, not only the screen (corrected in revision 2)

Revision 1 chose `L` by screen-space error alone: the coarsest tier whose cells still cover
`px_per_cell_target` pixels, `R_L = 2^L / (px_per_cell_target · θ_px)` — about 786 m · 2^L at 1920 px,
70°, 2 px/cell (ESTIMATED, design §3.7.1). That is not enough, and the movement law already says why.

**The defect.** The server always collides at tier 0. The client draws a coarse rung outside 786 m, where
§5.3 bounds the height error at `4 · k_rough · 2^L` metres. A hull at 240 m/s crosses 786 m in about
3.3 s (ESTIMATED, arithmetic on the report's own two numbers). For over three seconds the pilot flies at
a cliff the client draws metres away from where the shard will collide.

*Game example.* A hull runs a canyon on the moon at 240 m/s. The client draws the far wall at tier 2
(4 m cells). The wall the shard collides on stands up to 8 m nearer. The hull explodes against air.

**The rule.** The tier choice reads the closing speed as well as the screen error, exactly as the interest
radius does. The 2026-08-27 movement ruling already settled the shape: *"grow the interest radius with the
closing speed; never cap the speed."* So:

> `R_tier0 ≥ max( 786 m , v_closing · (t_compose + t_extract + t_mesh) · safety )`

and the same lead applies to every rung. The constant is not guessable; U5 measures `t_compose` and U12
measures `t_extract`. Until they are measured the lead is UNKNOWN, and that is a blocking measurement,
not a tuning value.

The CHOICE of `L` still reads a float distance, which stays lawful: the choice is display-only and
changes nothing the server reads. The FUNCTION for a chosen `L` is exact.

### 5.3 The disagreement between rungs, bounded

For a self-similar stack with persistence ½ and declared roughness `k_rough`,
`|h(dir, L) − h(dir, 0)| < 4 · k_rough · 2^L` metres (design §3.5.6; a theorem for this prefix-sum
construction, ESTIMATED until the `terrain_tier_agreement` property test runs, U11). Two consequences:

1. The coarse band is the fine band widened by that tail, so the shard's all-air pre-rejection is conservative (`band_L ⊇ band_0`, a property test).
2. At the range where tier `L` is used, the height error subtends `(4·k_rough·2^L) / (786·2^L) = 4·k_rough/786` radians — **independent of `L`**. At `k_rough = 0.5` that is 2.5 mrad = **about 4 pixels** at the reference screen (θ_px = 6.36e-4 rad; ESTIMATED from the two formulas). That is a visible pop.

### 5.4 The anti-pop consequence (SL8: tolerances are physical)

The 4-pixel figure means the ladder alone does NOT satisfy the seamless law. Two things close it, and both
are required, not optional:

- **A dither crossfade between rung `L` and rung `L−1` of the SAME terrain** over the last `crossfade_band_fraction` of rung `L−1`'s range (the shipped default is 0.15, because 0.20 does not pass the residency gate — MEASURED 2026-08-03, `decision_board.md:104`). The crossfade is a pipeline bit; both rungs are resident in the band.
- **Skirts** on every chunk border, deep enough to cover the bound plus the rim-warp amplitude, so the crack between rungs can only open downward and is hidden.

**★ The gate speed, corrected.** Revision 1 called 240 m/s "the mesher-bound flight ceiling". That label
is wrong. The base names FOUR ceilings in binding order
(`docs/investigation/high_speed_flight_latency.md:366-376`):

| # | Ceiling | Speed | What binds it |
|---|---|---|---|
| 1 | Terrain UPLOAD bandwidth at the V1 vertex format | **~240 m/s** | the GPU byte budget, not the generator (`block_system_design.md:7965-7972`) |
| 2 | Latency + reaction + a 3 g pull-up at eye height | ~328 m/s | the control loop |
| 3 | The AoI warm ring | 393 m/s | `R₀ / t_warm` |
| 4 | **Terrain generation + meshing, 2 cores, 100 km view** | **528 m/s pessimistic**, 2 653 m/s typical | the generator and the mesher (`block_system_design.md:7702-7712`) |

A V2 vertex format moves ceiling 1 out of the way (`high_speed_flight_latency.md:377`). So **the pop
detector runs at BOTH 240 m/s and 528 m/s**: 240 m/s is what ships today, and 528 m/s is the generator's
own ceiling, which is what this domain must survive.

**The tolerance to write into the gate:** no pixel on a tier boundary changes by more than the dither's
own noise between two consecutive frames, at walking speed, at 240 m/s and at 528 m/s.

### 5.5 ★ Above the coarsest rung: the proxy handover (new in revision 2)

`R_L = 786 · 2^L` metres and the ceiling is the body's `tier_depth`, about 13 on the starter world
(`block_provenance_collapse.md:953`, a design quantity). So the coarsest generated rung reaches roughly
786 · 2¹² ≈ **3 200 km** (ESTIMATED, arithmetic). Beyond that the realm proxy draws — one fixed
tessellation today (`crates/client/src/realm_scene.rs:800-803`).

The crossfade of §5.4 covers rung `L` against rung `L−1` of the SAME terrain. It says nothing about the
proxy-to-coarsest-rung handover, and that handover is the FIRST thing a player sees when a hull comes out
of warp. SL8 names it twice: "arrival pop" and "detail-by-box".

*Game example.* A hull warps in and the moon is 10 000 km away. The client draws the proxy sphere. At
3 200 km the coarsest generated rung takes over. Revision 1 set no tolerance for that frame.

**The rule this report recommends:** the proxy is the rung ABOVE the coarsest generated rung, not a
different kind of thing. It uses the same crossfade band and the same pop tolerance. That makes the
proxy's radius the body definition's radius (§1.6) by construction, and it removes one seam kind from the
list. The proxy's own tessellation must therefore come from the generator crate too — a coarse sphere at
`radius_m` — and NOT from a client-side default.

### 5.6 ★ Who runs the work, and on what budget (new in revision 2)

Revision 1 said the client "evaluates chunks on a worker pool" and said no more. That is not a detail:

- `vd-client` spawns no thread today and holds no `rayon` (MEASURED, §0). Its clippy fence bans the clock and `thread::sleep` because "pacing belongs to the bin's render loop" (`crates/client/clippy.toml:6-10`).
- `vd-client` is Tier-A and must reach 100 % region and branch coverage (HR5). A thread pool inside it is a coverage problem as well as a design problem.
- The base's own figure says the budget is real: a 6 800-chunk 100 km view is **~6 s on one core and 0.75 s on eight** (MEASURED 2026-08-03, M4 Pro, `decision_board.md:102`). Eight cores is an assumption about the player's machine that no ruling has made.

*Game example.* A hull descends toward the moon. The client must fill the tier-0 disc — about 505 chunk
columns (`block_system_design.md:4152-4155`) — before the boots touch. On one core that is seconds. Who
runs them, and what is drawn while they run?

This is open decision D13 (§10) and owed measurement U14 (§6).

### 5.7 SL9: the cost grows with the observer, not with the moon

The strongest SL9 evidence in this domain is a MEASURED-by-arithmetic result the base states plainly:
*"every rung above the first holds about 379 chunk columns, regardless of which rung it is. Tier 0 is a
disc and holds about 505. Both figures are independent of the planet's radius"*
(`block_system_design.md:4152-4155`); `columns(T rungs) = 505 + 379·(T − 1)`.

**379 is a RESIDENCY, not a flux.** Revision 1 wrote "~379 new columns per rung per ring step", which
reads it as an arrival rate. It is the count of columns RESIDENT in a rung's annulus. The arrival rate at
a given speed is a different number, and this report does not have it — U5 now asks for it.

*Game example.* A hull cruises over the moon at 200 m/s. The rung two levels up holds about 379 chunk
columns whether the moon is 161 km across or 6 371 km across. How many of those columns are NEW each
second is the number the client's budget needs, and nobody has measured it.

**Example (the whole ladder).** A hull descends on the moon at 200 m/s. At 6 km the client draws the hill
at tier 3 (8 m cells); at 1.6 km it swaps to tier 1; at 786 m — or further out, if the closing speed says
so (§5.2) — to tier 0. Each swap is a dither over a 15 % band; the hill never jumps. When the boots
touch, the client extracted the surface at tier 0 and the moon's shard collided on the same triangles
from the same crate. The player stands on the pixel.

---

## 6. What is UNMEASURED, and the bench that measures each

| # | Unmeasured | The bench / measurement | Cost, ESTIMATED |
|---|---|---|---|
| U1 | x86-64 vs aarch64 byte equality of the generator AND the extractor. NEVER measured anywhere in this project (the only determinism gate re-runs the SAME binary on one host; `celestial.rs:15-30`; `DEFERRED.md:2574`). | G4 (§2.4): emulation first (necessary, not sufficient), then a real x86 box. | Minutes for the crate alone; emulator startup unknown. |
| U2 | opt-level 0 vs 3, debug vs release, stable vs the coverage nightly, `target-cpu=native` — same bytes? | G1+G2. | < 1 min after build. |
| U3 | Whether clippy `disallowed-methods` resolves `f64::sin` (primitive inherent method paths). | The SEAL-1 control: inject, run `just lint`, expect red. | Seconds. |
| U4 | Whether `f64::sqrt`/`floor` are callable under `#![no_std]` on stable 1.94.1, and what a `libm`-class dependency would cost. | One `cargo check` on a no_std stub. | Seconds. |
| U5 | **Per-chunk COMPOSE cost at every tier, and the column FLUX at a stated speed** (not the residency — §5.7). Appendix A measured ONE tier-0 chunk at 0.885 ms (MEASURED, M4 Pro, one core, `scripts/noisebench`) **with a different hash composition** (§0), so it is a proxy. | Re-run `noisebench` with `vd_core::rng` as the hash; add an octave-count parameter; then measure columns ENTERING a rung's annulus per second at 240 m/s and at 528 m/s. Make "tier L costs strictly less than tier L−1" a gate. | An afternoon. |
| U6 | The boot self-check cost (8 chunks, compose + extract) on the slowest shipped client. | Time it in `vdctl gen-digest`. | Seconds. |
| U7 | The 0.885 ms figure itself, under the mandated hash. | The re-run of U5. Until then, every number derived from it (the ~7 ms boot check, the gate leg) is ESTIMATED at one further remove. | Minutes. |
| U8 | The coarse-lattice cavern interpolation is bit-exact across targets (only `+ − ×` with power-of-two weights — an argument today). | A cavern chunk key in the golden set. | Free once G4 runs. |
| U9 | The C-ABI copy cost per chunk. **The payload is NOT 476 KB** (§1.5): it is ~715 KB packed or ~953 KB word-aligned for the cells, PLUS the extracted surface buffer. | A microbench in the ffi crate, sized on report 01/04's frozen record. | Minutes, after the record is frozen. |
| U10 | The k3d image architecture on this host. | `docker version --format '{{.Server.Arch}}'` with Docker running. | Seconds. |
| U11 | `band_L ⊇ band_0` and `|h_L − h_0| < 4·k_rough·2^L` hold for the SHIPPED octave table. | `terrain_tier_agreement` and the band containment property tests. | Included in the first slice. |
| U12 | **The EXTRACTOR's cost per chunk, and its output size.** Never measured. `02_smooth_terrain.md` ESTIMATES 0.3–1.5 ms per surface chunk from the surface-cell count. | Extend `noisebench` with the surface-nets pass. It gates §5.2's lead constant. | An afternoon. |
| U13 | **Whether the composed chunk (generated + edit + block) is byte-identical on two hosts.** The composition runs on both and was never fenced (§4.4). | The composed row in the golden set, on every leg of §2.4. | Free once G1 runs. |
| U14 | **The client's generation budget: threads, per-frame slice, and the one-core case.** `vd-client` has no thread today (MEASURED). | A ring benchmark inside the client harness, on one core and on eight, at 240 m/s and 528 m/s. | An afternoon. |
| U15 | The pop in pixels at a tier boundary with the crossfade on, and at the proxy handover (§5.5). | The HR6 readback pop detector at walk, 240 m/s and 528 m/s. | Needs the mesher (P4b). |

---

## 7. Law conflicts (SL6 requests and refusals)

**★ SL6 — THREE NEW WIRE ARMS ARE REQUESTED. Revision 1 said "NONE requested" while asking for two; that
line is withdrawn.** CLAUDE.md line 210 states SL6 in full: *"ASK BEFORE NEW DATA CROSSES A REALM
BOUNDARY, and before adding a wire arm. Default NO. … State what data, from which realm to which, why the
receiver cannot compute it from what it legitimately holds, and what doing without costs."* Here they are,
in that form.

**SL6 ask 1 — `TAG_SURFACE` in a realm's own look bag.**
- **What data:** a skip-unknown TLV tag (id 4, beside `TAG_LOOK` = 1, `TAG_LUMA` = 2, `TAG_EXTENT` = 3 at `crates/core/src/look.rs:31,35,48`) carrying the realm's own `FrameRef` and the DECLARED generator tag.
- **From which realm to which:** from the realm ITSELF, about ITSELF, to the window of a client already subscribed to it. It rides `BodyStmt::SelfLook` (`crates/wire/src/session_flow.rs:625-630`), which only a running realm may send and which `window_body_admissible` already fences to the realm's own head.
- **Why the receiver cannot compute it:** the client cannot know whether the moon's shard is running a generator that matches its own build, and it cannot know that the moon HAS a surface at all (a hull has none — `ShipLocal` carries no seed, §0).
- **What doing without costs:** the client either draws every realm from its seed on faith — and draws a hill for a hull that has none — or it draws none, which discards V1.1 entirely.
- **Law note:** the ONE-RADIUS LAW (`look.rs:39-48`) binds the parent's MARKER bag, not a realm's self-look, so this tag does not touch it. But a marker may still carry only a radius.

**SL6 ask 2 — `ClientControlMsg::HelloWorld { declared, measured }`, a new trailing variant.**
- **What data:** two `u64`s — the declared world generation and the boot-measured arithmetic profile.
- **From which realm to which:** it is not a realm-to-realm crossing at all; it is a build stating its own identity to the gateway during the login handshake, immediately after `Hello` (`crates/wire/src/channels.rs:44-49`).
- **Why the receiver cannot compute it:** the gateway cannot know what generator a client binary contains, nor what its CPU and compiler pair produce.
- **What doing without costs:** a client draws a hill the shard does not have and finds out when the boots fall through. V1.3 asks for exactly this refusal and states, wrongly, that it already exists (§8).
- **Why a variant and not a field:** `ProtoVersion::CURRENT` is a `const` and the code forbids a caller from stating its folded value (`version.rs:355-373`); `version.rs:10-14` states the rule "New data rides a new trailing variant, never a new field".

**SL6 ask 3 — a mesh-peer statement of the MEASURED profile, on the admin lane.**
- **What data:** one `u64`, the boot self-check digest, stated by a shard or a gateway after the mesh handshake.
- **From which realm to which:** node to node, not realm to realm; the surface it protects is shared by every realm that has one.
- **Why the receiver cannot compute it:** a peer's arithmetic profile is a property of that peer's binary and chip.
- **What doing without costs:** two shards of one world compute two different hillsides and neither knows. Putting it on the ALPN instead (revision 1's shape) would forbid a mixed-architecture cluster silently and force the orchestrator to run terrain code at boot (§3.3).

**★ The `coarsen_level` lane is NOT available for any of this.** `OccupantInterest` is a TOMBSTONE
(`crates/wire/src/intershard.rs:1174-1186`), kept only so a reserved discriminant stays decodable, and
the pose field beside it is documented as "the SL2 breach that condemned it". Any coarse-detail request
the ladder needs is a NEW arm and a fourth SL6 ask, which this report does not make.

Other laws:

- **SL2/SL1 — untouched.** No pose, no velocity, no placement is derived on the client (V1.7). The generator has no field a pose could ride in.
- **SL3 — one statement, named:** a realm's look bag gains `TAG_SURFACE`. This is the realm authoring HOW IT LOOKS; the parent still authors WHERE. It shrinks the parent's per-child message toward a placement, as SL3 demands. And because the tag rides `SelfLook`, a DORMANT realm is still never drawn (§1.4) — stated on purpose, not left to the carrier's luck.
- **SL4 — the generator is not a motion crate.** `vd-terrain` never names an orbit, a thrust or a velocity, and never depends on `vd-physics`. The new edge runs the other way (`physics → terrain`, §1.6), so the crossing path still cannot reach a motion crate through the generator. **The gate that holds this law is `tests/tests/crate_isolation.rs`** — named as the holder by `crates/core/src/lib.rs:17`, `crates/core/src/worldgen.rs:9`, `crates/physics/src/lib.rs:12` and `crates/wire/src/version.rs:834`. (Corrected: revision 1 cited `crates/physics/Cargo.toml`, which is a package DESCRIPTION that merely mentions the gate. A row in a description fences nothing.) That test gains one row: the crossing path may not name `vd-terrain` either.
- **SL5 — one world.** The golden set is evaluated on `HOME_SEED` (2298), THE world's seed. No test-only body, no reduced octave table. §1.6 removes the second radius, which was a second world fact.
- **SL9 — unbounded children.** Feature anchors per chunk are a bounded draw from the seed, never a walk of a realm's children. And the residency is independent of the body's radius (§5.7): the cost grows with the observer, not with the moon.
- **HR3 — no shard-kind match.** The surface is a `ShardProfile` capability beside the existing `voxel: Option<VoxelGeometry>` (`crates/sim/src/capability.rs:107`); the code asks "does this profile carry a body definition?", never "is this a planet?".
- **HR5 — 100 % region+branch.** The generator is Tier-A; the noise sampler's and the extractor's data-dependent branches each get a directed test. The ffi crate is Tier-B. A client-side worker pool (§5.6) is an HR5 cost that D13 must price.
- **SL8 — the seamless law.** §5.4 (the crossfade and the skirts, gated at 240 and 528 m/s) and §5.5 (the proxy handover as one more rung). The "tier refusal" seam kind is unrepresentable because `tier ≤ tier_depth` is checked on the KEY constructor.
- **The 2026-08-27 seed ruling S5.1** ("a block's substance may not be a pure function of (position, seed)") **vs V1.1** ("seed-decided common materials"). V1 is newer and wins for COMMON materials; S5 keeps binding VALUABLE ones. The line is the owner's (D3), and the indicator question is D11.

### 7.1 ★ HR4 — the second subject, named (corrected in revision 2)

HR4 says *"every feature passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't
land."* Revision 1 offered "a station: a flat generated floor, or an empty body". Both answers fail: an
empty body does not run the feature, and an invented station floor is new world content no ruling asks
for, which SL5 calls a reduced world grown to satisfy a gate.

**The correct reading is that the FEATURE is not "generate terrain". The feature is "compose the shape,
extract the surface, and collide on it".** That runs in full on both voxel geometries
(`crates/sim/src/capability.rs:37-42`):

| Profile | What the fixture does | What is identical |
|---|---|---|
| **Spherical** (a moon) | Place three terrain voxels on a hillside, mine one cell, extract the surface, build the collider, walk an occupant across the seam. | The composed density, the extracted triangles in the realm's own frame, the contact point. |
| **Cartesian** (a hull) | Place the SAME three terrain voxels on a steel deck, mine one of them, extract, collide, walk the same occupant across the same seam. | The same three, byte for byte, in the hull's own frame. |

The generator's contribution on the hull is empty (a hull has no seed and no body definition, §0), and
that is not a hole: the composition, the extractor and the collider all run. A player who heaps dirt in a
ship's hold gets the same rounded knoll as a player who heaps it on a moon, from the same code.

**A caveat this domain cannot close alone:** the fixture assumes a terrain-form cell is legal inside a
Cartesian realm. Reports 01, 02 and 04 own the cell record; `02_smooth_terrain.md:188-190` says the
density byte on a Cartesian realm is "always fully air", which would make the fixture empty. **If those
reports rule that a hull may hold no terrain voxel, then the surface capability is Spherical-only, and
HR4 must be discharged by a second REALM of the same profile with a different body definition — a moon
and an asteroid.** The owner must be told which, before the slice starts.

---

## 8. Stale claims in the investigation base and older documents

| Claim | Where | What supersedes it |
|---|---|---|
| "cross-platform float determinism explicitly not pursued" | `docs/design/PLAN.md:135` | SL10 V1.3: no drift is a byte-for-byte MEASURED gate on every target. |
| "The world-generation tag … already refuses a client whose generator disagrees" | `owner_decisions_2026-09-07_voxels.md` V1.3 (a statement of fact inside a ruling) | The tag refuses a PEER (ALPN, `crates/io-prod/src/trust.rs:41-43`) and a FILE (`crates/core/src/store_stamp.rs:262-267`); the player-client handshake carries `coordinate_generation` only (`crates/wire/src/version.rs:337-351`). The client refusal must be BUILT (§3.3), as a new trailing variant. |
| "`mul_add` is bit-exact and allowed"; "the FMA half of SPIKE-6a can be struck" | `block_system_design.md` §3.5.4 rule 1 | V1.4: no fused multiply-add. `mul_add` is banned on the surface. |
| "`min`/`max` are safe scalar operations" (implied by every allowed-operation list in the base) | `block_system_design.md` §3.5.4 | Rust documents `f64::min`/`f64::max` as returning either input non-deterministically when the inputs compare equal (`+0.0` vs `−0.0`). They are removed from `Gf` (§2.1). |
| "the design put the fence in `crates/core/src/grid/gf.rs`, i.e. inside `vd-core`" | `block_system_design.md` §3.5.4 rule 4 | Under SL10 the generator is its OWN crate compiled into two hosts; the fence lives in that crate. |
| "Body definition … closed-form inverse-CDF samplers following the `taxonomy.rs` idiom" | `block_system_design.md` §3.5.1 step 1 | `crates/physics/src/taxonomy.rs:616` uses `powf`. The body definition must be clause-4 clean and computed inside the crate, and it becomes the ONE owner of a body's radius (§1.6). |
| "S6: NO client-side derivation" | `owner_decisions_2026-08-27_seed_and_secrecy.md` S6 | Superseded for the static shape by V1.8. The star field stays shipped once (reach R1). |
| The concealed drop table is `f(realm_key, chunk_key)` — static inputs only (M12) | `concealed_resources.md`, "Placement, and why the generator is integer-only" | S5.2 (2026-08-27): anything that pays must read world state that MOVES. The table takes the region's depletion ledger as an input (§4.3). |
| "P4 deliverable: pinned `noise` crate" | `docs/design/roadmap.json` P4 (quoted by `decision_board.md:2217`) | `noise 0.9.0` is declared and unused (`Cargo.lock`: zero entries, MEASURED). The owner decides the noise source (D5); the roadmap text is not current. |
| **"`crates/wire/src/intershard.rs:303` reserves the WHAT lane … the reservation stands"** | `block_system_design_addendum_2.md` §A, **and revision 1 of THIS report** | **The lane is DELETED.** `crates/wire/src/intershard.rs:1174-1186` marks `OccupantInterest` a "★TOMBSTONE payload (Step 5 slice D) … nothing produces it", and the pose field beside `coarsen_level` is documented as "the SL2 breach that condemned it". A grep finds the name only there and in three test literals (`crates/wire/tests/intershard_closed.rs:405`, `crates/sim/src/stub/tests/aoi_demand.rs:607`, `intershard.rs:2079`). Reading it as room to build in would resurrect an SL2 breach. |
| "The realm proxy stays a single fixed tessellation … becomes the coarsest rung at P4" | `crates/client/src/realm_scene.rs:800-803` (a code comment) | Still true as a plan; §5.5 makes it a design statement with a tolerance, not just a plan. |
| Addendum 2 §B "Fact one: terrain is generated, not stored. The client has the generator and the seed." | `block_system_design_addendum_2.md` §B | Was an ASSUMPTION on 2026-08-03; the 2026-08-27 ruling refused it; SL10 grants it for the static shape only. It is lawful now by the newest ruling, not by the addendum. |
| "240 m/s is the mesher-bound flight ceiling" | **revision 1 of THIS report**, and a loose reading of `decision_board.md:106` | 240 m/s is the terrain UPLOAD bandwidth ceiling at the V1 vertex format (`high_speed_flight_latency.md:370`; `block_system_design.md:7965-7972`). The generation-and-mesh ceiling is 528 m/s pessimistic (`block_system_design.md:7702-7712`). §5.4 gates both. |
| "~379 new columns per rung per ring step" | **revision 1 of THIS report** | 379 is a RESIDENCY, not a flux: *"every rung above the first HOLDS about 379 chunk columns"* (`block_system_design.md:4152-4155`). The flux is unmeasured (U5). |

---

## 9. One-way doors

| Door | Deadline | Cost if wrong |
|---|---|---|
| **The generator's output is frozen the moment the first diff is saved.** Every edit is an offset into the seed's shape. A change to the noise source, the octave table, the coarse-lattice cave step, the cube-sphere curve, the radius ladder, the EXTRACTOR's rule, the COMPOSITION order, the surface-skin field or the body-definition arithmetic moves the hill under a saved tunnel. The crate version in the declared tag is the epoch counter; a bump refuses every saved store (`store_stamp.rs:262-267`), and the fail-safe on refusal is DISCARD (`:182-192`). | Before the first world is saved. Sharper: at the moment the golden literals are committed. | A galaxy-wide regeneration under existing player builds, or a migration nobody can write. |
| **★ THE EXTRACTOR'S PLACEMENT** (new; the biggest door in this domain). If the extractor stays outside the crate and an Unreal client links the staticlib, a second code base owns the ground. Moving it in afterwards rebuilds that code base and re-validates every saved diff against a surface that moved. | Before the C surface is frozen. | Two code bases and every player's terrain. |
| **★ THE CELL RECORD'S WIDTH** (new). Freezing `uint16_t` per cell and then adding the density byte changes every ABI signature and every golden literal. This domain must FOLLOW reports 01, 02 and 04, never lead them. | Before the ffi crate exists. | Every C signature and every golden digest. |
| **★ THE ONE OWNER OF A BODY'S RADIUS** (new, §1.6). The forest ships `taxon.radius_m` as the moon's look today (`generate.rs:1232`, `:1819`). Moving that ownership into the generator moves every body's radius. | The same commit as the first crate-version bump; before the first world is saved. | A galaxy-wide regeneration, plus a period in which the terrain and the drawn outline disagree. |
| **The body definition arithmetic is inside the crate, clause-4 clean.** If the radius is taken from `taxonomy.rs` (`powf`), an ulp on one target flips a ladder step and the entire moon differs. | Before the first chunk is generated. | Same as above, plus a drift the golden set may not catch until an x86 client arrives. |
| **The concealed tier never enters the grid** (concealed C-4). A visible strategic deposit leaks through the edit pyramid at 1 596 × its radius (`concealed_resources.md:520`). | Before the first world is saved. | A galaxy-wide pyramid rebuild and a wiki of every deposit. |
| **The public/valuable material line** (D3), and **the indicator strength** (D11). Moving a material from public to concealed after players mine moves ore out from under their tunnels. | Before the first world is saved. | A regeneration under existing builds. |
| **The client-facing tag carrier.** The door is WHICH MECHANISM, not when: option (a) is not available at all while `ProtoVersion::CURRENT` is a `const` (§3.3). | Before any client build with the generator ships. | A flag day that buys nothing, and a const the code says nobody may state. |
| **★ AN EMULATED GREEN IS NOT AN x86-64 GREEN** (new). If the project calls G4(a) "no drift on x86-64" and ships, a real x86 part may differ in subnormal handling and nobody learns until a player's boots fall through. | A real x86-64 host is owed before the Unreal client links. | A whole architecture of players standing on a different hill. |
| **The dependency shape** (§1.3 A/B/C). Moving `rng.rs` after the Unreal client links the staticlib changes the symbol set that client depends on. | Before the Unreal client links. | A coordinated rebuild of a second code base. |

---

## 10. Open decisions for the owner (with a recommended answer)

| # | Question | Options | Recommended | Why |
|---|---|---|---|---|
| D1 | The generator crate's dependency shape. | A: depend on `vd-core`. B: a leaf `vd-seed` (rng + digest) under both. C: `vd-core` depends on the generator. | **B** | One hash, one digest, the smallest staticlib, and no `libm` edge through glam. |
| D2 | The C header for the Unreal client. | H1 hand-written + a signature test. H2 `cbindgen` build-dep (new dep). H3 `cbindgen` as a tool, header committed, diff gate. | **H1 now, H3 if the surface grows** | About a dozen functions; no new crate in the product graph. |
| D3 | Which materials are seed-decided (public): bulk stock only, or also coal/iron/copper? | (i) bulk stock only; every ore is live. (ii) bulk + common ore (R11). | **(ii)** | V1.1/V1.6 imply a public class; the wiki test passes for plentiful low-value ore; the survey mechanic keeps its value on the strategic tier. |
| D4 | The client-facing tag carrier. | (a) append `world_generation` to `ProtoVersion`. (b) a new trailing `HelloWorld { declared, measured }` variant. | **(b)** — CHANGED in revision 2 | (a) is not available: `ProtoVersion::CURRENT` is a `const` folded at compile time and the code states that no caller may state it (`version.rs:355-373`); `version.rs:10-14` prescribes a trailing variant. (b) also keeps terrain work behind the login handshake. |
| D5 | The noise source. | (A) vendor ~400 lines (gradient noise on the integer hash) into the crate, in `Gf`. (B) `noise 0.9.0` with a const table. (C) another crate. | **(A)** | No dependency, no `rand` drift class; delete the unused `noise` pin. **Note:** the bench's numbers do NOT transfer as-is — its hash composition differs from `child_seed` (U7). |
| D6 | `#![no_std]` for the generator crate. | (a) `std` + the lint + the link scan. (b) `no_std` + an owner decision on a `libm`-class dependency. | **(a) — CHANGED in revision 2** | `no_std` does not stop an `extern "C"` libm call, and on stable it deletes `f64::sqrt`, which V1.4 grants by name. Revisit only if U4 measures otherwise. The link scan is the fence either way. |
| D7 | The x86-64 leg of the gate (G4). | (a) emulation on this Mac. (b) a cloud x86 runner. (c) a contributor's x86 box. | **(a) as a SMOKE TEST now, and (c) or (b) before V1.3 may be called satisfied** — CHANGED in revision 2 | An emulator can find a difference and can never prove its absence. V1.3 says "every target the game ships on", and a shipped target is real hardware. |
| D8 | The tag's arithmetic-profile part: a boot self-check (8 chunks) or a declared string. | measured / declared | **measured, on the live carriers only** | A declared profile can lie; a digest cannot. But a measured digest must never gate a durable FILE (§3.3), because the remedy discards a world. |
| D9 | Name of the crate (`vd-worldgen` collides with `vd_physics::worldgen`, the forest). | `vd-terrain`, `vd-shape`, `vd-surface`, … | **`vd-terrain`** | One word, one meaning. |
| D10 | Does the golden set get committed literals per TIER? | per tier / tier 0 only | **per tier** (about 832 digests at `tier_depth` 13) | `h(dir, L)` is a different function per `L`; a tier-0 pin proves nothing about tier 3. |
| **D11** | **How strong may a seed-derived INDICATOR material be?** (new in revision 2) | (i) none at all — no seed-derived hint of any live deposit. (ii) a hint with a small DECLARED factor. (iii) a hint whose factor is itself live state. | **(i) for the strategic tier, (ii) only for materials D3 already made public** | Anything a fixed seed decides is publishable by wiki. A declared multiplier on a static position is a prospecting map in every client binary, and it devalues the survey instrument the concealment ruling paid for. |
| **D12** | **Who owns a body's radius, and what fidelity does the clause-4 rewrite cost?** (new in revision 2) | (i) the generator owns it; `taxonomy.rs` keeps mass/class and the radius law is re-expressed as a polynomial or an integer table. (ii) the forest keeps it and the generator SNAPS to it — which re-admits `powf` into the shape. | **(i)** | (ii) makes a ladder step depend on an ulp of a `powf` on two targets, which is the exact drift SL10 exists to stop. (i) costs astrophysical fidelity, and the owner should say how much is acceptable. |
| **D13** | **Who evaluates chunks on the client, and on what budget?** (new in revision 2) | (i) the render bin owns a pool and feeds finished chunks into Tier-A `vd-client` through a seam, as it already feeds wall-time. (ii) `vd-client` gains a pool (new dep, HR5 cost). (iii) single-threaded with a per-frame slice. | **(i)** | It keeps the Tier-A crate pure, matches the existing clock seam (`crates/client/clippy.toml:6-10`), and adds no dependency to the engine-free lib. U14 must measure the one-core case before this is settled. |

---

## 11. Measurements owed (in order)

1. **U7 + U5 first**: re-run `noisebench` with `vd_core::rng` as the hash. Until that runs, the 0.885 ms figure and every number built on it are ESTIMATED at one remove, including the boot self-check and the gate's cost.
2. U12: the extractor's cost and output size. It gates §5.2's lead constant and the whole client budget.
3. U4 + U3 (minutes): `no_std` availability; clippy primitive-method paths. They settle D6 and the lint file's shape.
4. U1 + U2 + U13 (the gate legs G1–G4): the first byte-for-byte run, including the COMPOSED chunk and the extracted vertices, and including x86-64 under emulation. Nothing about "no drift" is true until two architectures print two equal digests.
5. U5's flux half and U14: columns entering a rung per second at 240 m/s and 528 m/s; the client's one-core case.
6. U6 + U10: the boot self-check cost; the k3d image architecture.
7. U11: the two property tests on the SHIPPED octave table.
8. U15: the pop detector at tier boundaries and at the proxy handover (lands with the mesher).
9. U9: the C-ABI per-chunk copy cost (lands with the ffi crate, after the record is frozen).

---

## 12. The recommended design in one paragraph

One Rust crate, `vd-terrain`, depends on a 150-line `vd-seed` leaf and on nothing else. It holds the body
definition (and it becomes the ONE owner of a body's radius, which the forest then reads), the density
field with an integer tier, **the surface extractor and the composition order**, strata, biome, water,
carvers, public deposits, feature anchors with a support rule, feature geometry, the band, the digest and
the golden self-check. Materials, textures and V2.6 style stay outside; the surface does not, because the
surface is what the player stands on. Its arithmetic is `Gf(f64)` with a private field and `+ − × ÷ sqrt
floor` — no `min`, no `max`, no FMA, no libm — fenced by a compile-fail control, a lint, and a link scan
that no inlining can hide from. A byte-for-byte golden gate over the COMPOSED density and the EXTRACTED
vertices is the measurement; **no leg of it has run**, and the x86-64 leg is OWED on real hardware before
V1.3 may be called satisfied. The moon's shard and the client link the same crate; an Unreal client links
its staticlib through about a dozen `extern "C"` functions, one of which returns the surface. The tag
splits: a DECLARED half rides the store stamp, the intershard ALPN and the realm's look tag, and a
MEASURED boot digest rides only the live handshakes, so a chip can never cost a player's tunnels.
Everything the seed does not decide is a one-hop diff from the owning realm, composed in one stated order
on both hosts; the strategic deposit is a server-only drop table that reads a live depletion ledger and
never enters the grid. The client picks a tier by screen error AND closing speed and evaluates fewer
octaves; the server extracts and collides at tier 0; the two are the same bytes wherever a boot can land;
the crossfade, the skirts and the proxy-as-a-rung close the pops the ladder alone leaves.

---

## 13. Revision log

Two refuters reviewed revision 1: `verdicts/generator_law.md` (12 findings) and
`verdicts/generator_feasibility.md` (11 findings + 4 missing cases). Every finding is listed with what
this revision did. I verified each cited `file:line` myself before acting.

### From the law refuter

| # | Finding | What revision 2 did |
|---|---|---|
| F1 | BREAKS_LAW — the mesher outside the crate makes two surfaces | **ACCEPTED.** The surface extractor moved INSIDE the crate (§1.1), into the golden digest (§2.3), and onto the C surface as `vd_terrain_surface` (§1.5). §1.2 now separates the surface (in) from materials, textures and style (out). A new one-way door records the placement (§9). |
| F2 | BREAKS_LAW — a moon gets two radii | **ACCEPTED.** New §1.6 names the generator as the one owner of a body's radius, adds the `physics → terrain` edge (§1.3), states the migration as a world epoch, and opens D12 for the fidelity the clause-4 rewrite costs. A new §0 row and a new one-way door carry it. Verified: `generate.rs:1232`, `:1819`, `taxonomy.rs:616,934,984`, `visibility.rs:85,184`. |
| F3 | WRONG — the gateway and the orchestrator cannot skip the self-check | **ACCEPTED, and the design changed rather than the sentence.** §3.3 splits the tag: the ALPN and the store carry the DECLARED part, which every node computes as a `const` read; only a node that must agree on a surface runs the eight-chunk check. Verified: `gateway.rs:271,318`, `orchestrator.rs:86,596`, `client.rs:106`, `shard.rs:279`. |
| F4 | WRONG — the WHAT lane is a tombstone, not a reservation | **ACCEPTED.** The §8 row is rewritten, a §0 row states it, and §7 says plainly that any coarse-detail lane is a NEW arm and a fourth SL6 ask. Verified: `intershard.rs:1174-1186`. |
| F5 | BREAKS_LAW — §7 says "NONE requested" while asking for two | **ACCEPTED.** §7 now opens with three SL6 asks in SL6's own four parts (the third is the measured-profile statement that §3.3 introduced). |
| F6 | MISSING — the tier choice ignores closing speed | **ACCEPTED.** New §5.2 states the rule, gives the 3.3 s figure, cites the 2026-08-27 movement ruling, and marks the lead constant UNKNOWN until U5 and U12 measure it. |
| F7 | MISSING — the HR4 fixture has no lawful subject | **ACCEPTED.** New §7.1 names the feature as "compose, extract, collide" and runs it on Spherical and Cartesian with the same three terrain voxels. It also states the caveat that reports 01/02/04 may rule a terrain cell illegal in a hull, and names the fallback. |
| F8 | UNMEASURED_AS_FACT — "the gate proved" | **ACCEPTED.** The §1.5 example now says "when the gate has RUN". §2.4 opens with "No leg of this gate has run." §12 marks the x86-64 leg OWED. |
| F9 | WRONG — not every realm's frame carries a seed | **ACCEPTED.** The §0 row now reads "every SEED-GENERATED realm", names `ShipLocal` and `UniverseSpace`, and states the consequence for a hull. Verified: `pose.rs:87-113`. |
| F10 | MISSING — indicator materials have no owner decision | **ACCEPTED.** New §4.5 and new decision D11. |
| F11 | UNMEASURED_AS_FACT — `Gf` called structural without saying why | **ACCEPTED.** §2.2 layer 1 now states the private field, the absence of `From`/`Deref`, and the observed-failing compile-fail control. |
| F12 | WRONG (minor) — the SL4 fence cited to a description | **ACCEPTED.** §7 now cites `tests/tests/crate_isolation.rs` and the four files that name it as the holder. |

### From the feasibility refuter

| # | Finding | What revision 2 did |
|---|---|---|
| F1 | BREAKS_LAW — the crate ships cells, not the surface | **ACCEPTED**, same fix as the law refuter's F1. |
| F2 | WRONG — `uint16_t` cells cannot carry the density; 476 KB is the wrong number | **ACCEPTED.** §1.5 withdraws 476 KB, gives 715 KB packed and 953 KB word-aligned as ESTIMATED arithmetic, and states that reports 01/02/04 own the record and this domain must follow. U9 is re-sized and a one-way door added. |
| F3 | BREAKS_LAW — D4(a) cannot be taken; `ProtoVersion::CURRENT` is a `const` | **ACCEPTED.** D4 flips to (b), a trailing `HelloWorld` variant; §3.3 carries the box with the code's own words. Verified: `version.rs:355-373`, `:10-14`, `:335`. |
| F4 | BREAKS_LAW — a CPU-measured digest on the durable stamp turns a build difference into data loss | **ACCEPTED, and it is the largest design change in this revision.** §3.2 and §3.3 split declared from measured. The store and the ALPN take the declared part only. The mixed-architecture-cluster consequence is now stated. Verified: `store_stamp.rs:262-267`, `:182-192`. |
| F5 | BREAKS_LAW — `min`/`max` are documented non-deterministic | **ACCEPTED for the fix, DISPUTED for the scope.** `min`/`max` are removed from `Gf` and from the allowed list, a waterline column joins the golden set, and a §8 row records it. **DISPUTED: the refuter writes that a generator "meets this case constantly" and that two equal-comparing values can be "two identical-looking values" — two f64 values that compare equal are the same bits except for `+0.0` against `−0.0`, so the only case that can move an output byte is a signed zero.** The box in §2.1 states this. The fix is adopted in full because it is cheap and because §2.4's argument needs a fully determined list. |
| F6 | WRONG — `no_std` is not a compile-error fence and it deletes `sqrt` | **ACCEPTED.** §2.2 layer 3 carries the correction; D6 flips to "(a) `std` + lint + link scan" and names the `libm`-crate consequence of the alternative. |
| F7 | UNMEASURED_AS_FACT — the 0.885 ms does not measure the mandated hash | **ACCEPTED, with one narrowing.** A new §0 row states the difference, §2.3 and §6 U5/U7 mark every derived number as one remove further from MEASURED, D5 carries the warning, and U7 is now the FIRST owed measurement. **DISPUTED (narrowing, not a defence): the bench's `mix64` IS the repo's SplitMix64 avalanche, byte for byte — the difference is the COMPOSITION (`hash3` folds it three times with three multipliers; `child_seed` chains three `SplitMix64::new(..).next_u64()` calls, each of which adds the golden increment first). So the charge is "a different composition of the same avalanche", not "a different hash function". Verified: `noisebench/src/main.rs:22-35` against `crates/core/src/rng.rs:22-28,70-74`. The consequence the refuter draws — that the number does not transfer — stands unchanged.** |
| F8 | WRONG — 379 is a residency, not a flux | **ACCEPTED.** New §5.7 states the residency correctly, cites it as this domain's strongest SL9 evidence, and U5 is rewritten to ask for the flux. A §8 row records the error. Verified: `block_system_design.md:4152-4155`. |
| F9 | WRONG — 240 m/s is a bandwidth ceiling, not a mesher ceiling | **ACCEPTED.** §5.4 now carries the four-ceiling table and gates the pop detector at BOTH 240 and 528 m/s. A §8 row records the error. Verified: `high_speed_flight_latency.md:366-377`, `block_system_design.md:7702-7712`, `:7965-7972`. |
| F10 | UNMEASURED_AS_FACT — the Rosetta sentence | **ACCEPTED.** The sentence is withdrawn in §2.4, G4(a) is restated as necessary-and-not-sufficient, D7 changes, and a new one-way door says an emulated green is not an x86-64 green. |
| F11 | BREAKS_LAW — SL6 "NONE requested" | **ACCEPTED**, same fix as the law refuter's F5, and the third ask (the measured-profile statement) is added because §3.3 introduces it. |
| M1 | MISSING — the composition order is never fenced | **ACCEPTED.** New §4.4 states the five-step order, puts it in the crate as `vd_terrain_compose`, adds a composed row to the golden set, and adds U13. |
| M2 | MISSING — a feature anchor whose ground was mined | **ACCEPTED.** New §4.6 gives a pure support rule (stand / cull, never fall), because a falling tree needs time and V1.1 forbids the crate to know time. The owner may overrule it. |
| M3 | MISSING — the proxy-to-coarsest-rung handover | **ACCEPTED.** New §5.5 gives the ~3 200 km figure, names the two SL8 seam kinds, and recommends that the proxy BE the rung above the coarsest, drawn from the generator at the body definition's radius. |
| M4 | MISSING — which thread, and what budget | **ACCEPTED.** New §5.6, new §0 row (MEASURED: no thread, no rayon, the clippy fence), new decision D13, new measurement U14, and the `decision_board.md:102` figure of 6 s on one core against 0.75 s on eight. |
| — | The gate's own arithmetic: "64 × ~8 tiers" against "64 × tier_depth" | **ACCEPTED.** §2.3 now says about 832 digests at `tier_depth` 13, and §0 records that `tier_depth` exists in the design and nowhere in the code (MEASURED: `grep -rn "tier_depth" crates/` returns nothing). |
| — | The `taxonomy.rs` path was written under a `worldgen/` heading | **ACCEPTED.** Every citation now reads `crates/physics/src/taxonomy.rs`, and §0 notes that `crates/core/src/taxonomy.rs` is a different file. |
| — | The law refuter's S5: the design keeps "a dormant realm is never drawn" by luck of carrier | **ACCEPTED.** §1.4 and §7 now state it: the surface tag rides `BodyStmt::SelfLook`, which only a running realm may send (`session_flow.rs:626-630`). |
