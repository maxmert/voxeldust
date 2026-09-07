# Verdict — feasibility refutation of `02_smooth_terrain.md`

**Lens:** Smooth terrain built of voxels. Technical refutation.
**Date:** 2026-09-07. **Refuted:** YES.
**Method:** I read the report. I read CLAUDE.md, `owner_decisions_2026-09-07_voxels.md`, the earlier
rulings and `DEFERRED.md`. I grepped every code claim and I opened every base citation I rely on. I ran
no build and no test.

---

## 0. The verdict in one page

The report's CENTRAL choice survives. A terrain cell that holds a substance and a signed sample, meshed
by naive surface nets, is the right answer to V2.1. The comparison against a blocky field with a
smoothing pass is correct, and the report adopts no library behind the owner's back.

The report is REFUTED on what it builds on that choice. Sixteen load-bearing claims are wrong, break a
law, or are absent.

The four that matter most:

1. **The report says it asks nothing under SL6. It asks for three new wire arms.** `BlockEdit` is a word
   in a comment, not an arm. The terrain diff does not even ride the inter-shard lane. It rides the
   shard-to-gateway lane and the gateway-to-client lane, and it moves the protocol's minor number.
   *Example: a player digs a scoop on a moon. The moon's shard must say so to the gateway, and the
   gateway must say so to every client that watches the moon. Neither sentence has a word for it today.*
2. **The collider's 64 m radius comes from a speed bound the owner deleted.** A hull that closes at two
   kilometres a second moves a hundred metres in one tick. It passes clean through a ball of terrain
   collider sixty-four metres wide.
3. **The report claims "what you see is what you stand on" holds by construction. Nothing enforces it.**
   The client picks a detail rung by an angle rule with no floor. The shard collides the finest rung
   only. Where the two disagree, the player's boots and the player's eyes are two cells apart.
4. **The FrameSpace seam is terrain's first slice by the owner's own word, and the slice list has no
   entry for it.** Slice 1 asks the generator for a cell's radial direction on a moon. Nothing in the
   code turns a cell address into a direction, and the owner ruled on 2026-09-05 that building it IS
   terrain's first slice.

---

## 1. Findings, most severe first

### F1 — BREAKS_LAW. "No SL6 ask" is false. The diff lane needs three new wire surfaces.

**The claim.** §9: *"SL6 — new data across a realm boundary: NONE asked."* §1: *"The terrain diff lane
(SL10 clause 6) rides the reserved `BlockEdit` arm, one hop, from the owning realm's shard. No new arm is
asked for here."*

**What the code says.** `BlockEdit` is not an arm. It is a word in a doc comment:
`crates/wire/src/intershard.rs:34` — *"RESERVED (variant lands with its consumer): `BlockEdit` (P6)"*. A
grep over `crates/` returns that comment, the same comment in `crates/wire/src/lib.rs:24`, a test's prose
at `crates/wire/tests/intershard_closed.rs:675`, and a capability error name
(`crates/sim/src/capability.rs:98`). There is no variant.

Worse, `InterShardFlow` is the shard-to-shard contract. The path from a realm's shard to a client is a
different pair of lanes: `ShardToGateway::RealmSceneDelta` (`crates/wire/src/session_flow.rs:244`) and
`ServerControlMsg::RealmSceneDelta` (`crates/wire/src/channels.rs:179`), under a protocol minor number
(`crates/wire/src/version.rs:43`).

**The law.** CLAUDE.md standing law 13 (SL6): *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, **and
before adding a wire arm**. Default NO."* The report adds arms and states no ask.

**Fix.** State the ask in §9. Name the three surfaces — an `InterShardFlow` arm if a shard must tell a
sibling shard, a `ShardToGateway` arm, a `ServerControlMsg` arm — the bytes each carries, and the minor
bump. *Example: the moon's shard says "cell (chunk 4-7-2, index 918) is now `(air, Terrain, +127)`" to
the gateway, and the gateway repeats it to each client that holds the moon.*

---

### F2 — BREAKS_LAW. The 64 m collider residency rests on a bound the owner has deleted.

**The claim.** §5.4: *"The collider is resident only inside the physics radius of a dynamic body (the
base's 64 m, `block_system_design.md:6485-6492`, derived from the swept motion per tick plus reach)."*

**The base's own words**, verified at `block_system_design.md:6485-6488`: *"64 m, one chunk edge, derived
as 'greater than any body's per-tick swept motion at 20 Hz plus the character's reach'."*

**The ruling that removes it.** `owner_decisions_2026-08-27_movement_answers.md:92` — *"M-D — THE CEILING
LEAVES THE FLIGHT PATH, AND CONTAINMENT BECOMES SWEPT"* — and lines 115-127: containment reads the LINE
from the last tick to this tick, because nothing may be slowed to be caught. There is no per-tick motion
bound any more. So "greater than any body's per-tick swept motion" names a quantity that no longer has a
value.

**The failure.** A hull closes on a moon at two kilometres a second. At twenty ticks a second it covers a
hundred metres in one tick. A ball of resident collider sixty-four metres wide never contains the hull's
whole path. The hull passes through the ground.

**Fix.** Derive the collider's residency from the body's OWN swept segment for this tick — the line from
the last pose to this pose, made fat by the body's radius and the character's reach — and state it as a
per-body set, not one number. Measure the worst case in M5 with a landing hull, not a walking player.

---

### F3 — MISSING. SL10 clause 5 has no residency invariant, and the report calls it construction.

**The claim.** §0 item 3: *"What the player sees is what the player stands on, by construction."*

**Why it is not construction.** §5.1 and §5.2 let the client draw a chunk column at rung L, picked by an
angle rule, with no floor named. §5.4 makes the collider rung 0 only. The two agree only where the
client's rung-0 band contains the shard's collider set, and the report never states that condition.

**The numbers, from the report's own sources.** The base puts the client's rung-0 floor at *"one
chunk-width of the camera"* (`block_system_design.md:6408-6410`) — sixty-two metres. The collider ball is
sixty-four metres (F2). The floor is NARROWER than the collider. The vertical disagreement between two
rungs is *"two cells"* at `k_rough = 0.5` (`block_system_design.md:3895-3898`, verified).

*Example: a player walks to the edge of a hollow sixty-three metres from the eye. The client draws the
hollow at rung 1, two metres shallow. The moon's shard collides rung 0. The player's boots hang two
metres over the drawn floor, or sink two metres into it.*

**Fix.** State the invariant: *the client's rung-0 band strictly contains the shard's collider set, at
every camera pose and every speed*. Add it as a gate to slice 3 and slice 4. Measure the see-versus-stand
vertical gap directly — M4 measures the POP between rungs, which is a different quantity.

---

### F4 — MISSING. The FrameSpace seam is terrain's first slice, and the slice list omits it.

**The claim.** §1 mentions it once: *"`FrameSpace`, `reanchor()`, `AnchorGen` do not exist. … The terrain
slice builds it."* §11 then lists six slices and names none of them.

**The owner's ruling.** `docs/design/DEFERRED.md:309-311`: *"★ THE FRAME-SPACE SEAM
(`FrameSpace`/`SphericalSpace`/`CartesianSpace`, `reanchor()`) stays owed and is TERRAIN'S FIRST SLICE by
the owner's word (2026-09-05, 'yes to all'): it IS voxel geometry, so it lands with the first voxel, not
before it."* The still-owed HR4 fixture that forces a `reanchor()` rides with it
(`DEFERRED.md:331-334`; the known limit is written into `crates/sim/src/capability.rs:13-18`).

**The failure.** Slice 1 asks the generator for `h(dir)` at a cell on a moon. Nothing today turns a cell
address into `dir`. The report's first slice has no ground to stand on, and V3 asked for the
implementation sequence.

**Fix.** Make the seam slice 0. Give it the D-38 reanchor-forcing fixture as its gate, and one
measurement: the cell-address round trip stays exact across a re-anchor.

---

### F5 — WRONG. The apron depth is a factor `2^(L+1)` too deep.

**The claim.** §5.2: the crack is closed by *"a downward apron of depth `(4·k_rough·2^(L+1)) ·
skirt_safety` cells"*.

**The base's formula**, verified at `block_system_design.md:3895-3898`: `|h(dir,L) − h(dir,0)| < 4 ·
k_rough · s_L`, where `s_L = 2^L` **metres** and the result is **metres**. The same passage states the
consequence in cells: *"At a characteristic 30° slope (`k_rough = 0.5`) that is **two cells**, whether the
cells are 1 m or 128 m."*

**The failure.** The report keeps the metres form and writes the unit as cells. At rung 4 its apron is
`4 · 0.5 · 32 = 64` cells deep instead of 2. A skirt sixty-four cells deep on every lateral chunk face is
a large, pointless triangle count, and a reviewer who trusts the formula sizes the residency budget on it.

**Fix.** Write the apron as `2 · 4 · k_rough · skirt_safety` cells — a constant in cells at every rung —
and say that this constancy is the base's own result.

---

### F6 — WRONG. The stored byte is called a signed distance. It is a radial gap.

**The claim.** §2.3: *"The density is the signed distance from the cell's centre to the terrain surface,
measured along the radial (the `h − r` of the generator)."* §7 door 2 pins that meaning in the
generator's digest, and states the cost of being wrong: *"the digest changes = a world-format epoch, and
the base's inherited fail-safe on an epoch mismatch is DISCARD."*

**Why it is wrong.** `h − r` is the gap measured ALONG THE RADIAL. The distance to the surface is that
gap times the cosine of the surface's slope. On a 45° hillside the two differ by about 1.41. The clamp to
±1 cell therefore truncates the field at a different true depth on every slope, so the same byte means a
different thickness of rock on a plain and on a mountainside. Second, the report combines the carver by
`max(h_term, carver_term)`. A maximum of two distance fields is not a distance field near the seam, so
the byte is least trustworthy exactly where a tunnel meets a hillside.

**Fix.** Name the byte what it IS — a clamped, quantised RADIAL gap — or divide by the local gradient
magnitude to make it a true distance, and pin THAT in the digest. A wrong name behind a discard-the-world
door is the expensive kind of wrong.

---

### F7 — WRONG. The quantisation contradicts the report's own lattice claim.

**The claims.** §1: *"Every quantised terrain length (density step, skirt depth, controller offsets) is an
exact count of these cells"* — the fine cells of `Tier::Fine`, `step_exponent = −10`, verified at
`crates/core/src/pose.rs:504` and `crates/core/src/pose.rs:520-527`. §2.3: *"Quantised to `i8` (1/127
cell, about 8 mm at tier 0)."*

**The arithmetic.** One metre divided by 127 is 7.874 mm. One fine cell is `2⁻¹⁰` m = 0.9765625 mm.
`1024 / 127 = 8.063…`. No density step lands on a fine-lattice cell. The two sentences cannot both be
true. Separately, an `i8` runs −128 to 127, so a 127-step mapping leaves one code unusable or the range
asymmetric about zero.

**Fix.** Quantise to `1/128` cell. That is `2⁻⁷` m, exactly eight fine cells, and it maps the `i8` range
symmetrically. Re-state the door with the power-of-two step, which is the same reason the tier ladder
stores an exponent rather than a decimal (`crates/core/src/pose.rs:507-513`).

---

### F8 — MISSING. The coarse density summary and the coarse seed base are different quantities.

**The claim.** §5.3: the pyramid stores *"the mean density of the eight children, exact in integer
arithmetic (`(sum + 4) >> 3` on `i8`)"*, and *"the coarse smooth lane reads the density mean directly as
the coarse sample"*.

**The failure.** At rung L an UNEDITED cell's sample comes from `h(dir, L)`, the generator with L octaves
dropped. An EDITED cell's sample would come from the pyramid, as a mean of eight rung-0 absolutes. The two
absolutes differ by up to `4 · k_rough · 2^L` metres (`block_system_design.md:3895-3898`) even where
nobody dug. Laying one over the other makes a STEP at the edited cell's rim, the height of the
octave-drop error — up to four metres at rung 2. That is a detail-by-box seam, which SL8 forbids.

*Example: a tunnel a player dug last week. At rung 2 the report wants a dimple. What it gets is a
four-metre ledge around the dimple, because the eight children were summarised from the fine world and
the ground around them was summarised by dropping octaves.*

A small, separate defect in the same sentence: eight `i8` values sum to as much as ±1016. `(sum + 4) >> 3`
needs an `i16` accumulator. As written it overflows.

**Fix.** Store the mean of the DELTAS — the edited sample minus the seed-derived sample at rung 0 — so the
summary adds onto whatever base a rung supplies. Gate the STEP at the rim, not only the tunnel's
visibility, which is all slice 5's gate checks today.

---

### F9 — MISSING. The diff lane has no baseline, so a joining client's shape is not the shard's shape.

**The gap.** SL10 clause 6 makes every edit a diff. §5.2's lane row and M8 measure LIVE digging only —
*"one player mines 10 cells per second for a minute"*. Nothing in the report says how a client that
arrives after a week of digging learns about the week. Nothing says what happens to a diff issued while a
player's subscription flips during a crossing.

**Why it is load-bearing.** Without a baseline the client draws the seed's hill and the shard collides the
dug hill. That is SL10 clause 5 broken on every join — the same defect as F3, from a different cause.

**Fix.** Name the baseline: on subscribing to a chunk, the owning realm's shard sends that chunk's whole
diff set, keyed by the chunk address, before any incremental diff. Budget its bytes for a heavily dug
region. Add the case *"an edit lands while the player crosses out of the moon"* to §3.3.

---

### F10 — WRONG. The measured 0.885 ms does not transfer, and the report does not say so.

**The claim.** §2.2 carries `0.885 ms per 62³ chunk` forward as MEASURED
(`block_system_design.md:17414-17423`, verified: strategy C, 0.885 ms; height field 0.252 ms; fill 0.037
ms).

**What the measurement depends on.** `block_system_design.md:17428-17430` (A.3): strategy C samples the
cave field *"on a coarser lattice"* — four cells by default — and interpolates.

**Why it does not transfer.** Under a BLOCKY field that interpolant only decided which cubes are air. It
never appeared on screen. Under a SMOOTH extractor the interpolant IS the cave wall a player sees and
stands on, so every cave's shape is now capped at four-metre detail by an optimisation that was chosen
for cost. If the look forces a finer cave lattice, the base's own table gives strategy B at 4.16 ms
(`block_system_design.md:17417`), which is over the 1.20 ms gate the report itself names
(`block_system_design.md:17471-17473`).

*Example: a player walks into a lava tube on a moon. Its mouth is a four-metre-smooth funnel, whatever
the generator's cave noise says, because the density between samples is a straight line.*

**Fix.** Make M1 measure BOTH lattice steps, one and four. State the visual consequence of the four-cell
step before the digest is pinned, because A.5 conclusion 2 already puts that step inside the one-way door.

---

### F11 — UNMEASURED_AS_FACT. SL10 clause 4 is claimed "by construction", and the mechanism does not exclude what the clause names.

**The claim.** §0 item 2: *"It uses only add, subtract, multiply and divide, so SL10 clause 4 holds by
construction."*

**What the clause says.** `owner_decisions_2026-09-07_voxels.md` V1.4: *"No fast-math flag. No fused
multiply-add contraction. No call into the platform's transcendental functions."*

**Why the mechanism is not enough.** The report's fence is a float newtype (`Gf`) that cannot reach `sin`.
A newtype hides library calls. It does not hide `f64::mul_add`, and the report never says the newtype
withholds it. A newtype also cannot govern a build flag, an LLVM contraction setting, or a `RUSTFLAGS`
value somebody adds later. The standing rule of method is explicit: byte identity is a MEASUREMENT, never
an argument.

**Fix.** State the fence in three parts. The newtype exposes `+ − × ÷ sqrt` and nothing else, by name. The
generator crate pins its own profile and an allowed flag list. The guarantee is M3's digest diff, not
construction. Then write "UNMEASURED until M3 runs" in §0, where the report today writes "by
construction".

---

### F12 — MISSING. The spherical address-to-direction step is unpriced under SL10 clause 4.

**The gap.** `crates/sim/src/capability.rs:39-40` defines the spherical profile as a *"tangent-anchored
spherical projection with re-anchoring"*. That projection turns a cell address into a direction on the
moon, once per cell, INSIDE the generator's path. A tangent projection is where `atan` and `tan` normally
live — the very calls SL10 clause 4 forbids.

The report prices only the CONTROLLER's one `atan` (§4.2, *"runs once per realm at activation"*). It never
states the projection's operation list.

**Fix.** Write the projection's exact operations, or say plainly that the operation list is unresolved and
that M3 gates it. This is the same slice as F4.

---

### F13 — MISSING. The frame budget is never closed for a landing at flight speed.

**The gap.** §5.2 says chunk work is *"never on the main thread … two job classes with caps in the render
tuning struct"*. M1, M2 and M6 price ONE chunk. M7 prices what is resident. No measurement prices the
ARRIVAL RATE: how many chunks per second a hull demands while it descends at its rated cruise, and
therefore how many threads the client needs.

Without it the report cannot answer V2.9 for a landing. *Example: a hull drops toward a moon and crosses
four rungs. Each rung swaps the chunks under it. At 0.885 ms of generation (MEASURED) plus an ESTIMATED
0.3–1.5 ms of extraction, the report cannot say whether four threads or forty are needed, so it cannot
say the descent is free of a tick hitch.*

**Fix.** Add a bench: a scripted descent at the hull's stated cruise. Count chunk requests per second and
the deepest queue, against a named thread budget. Gate slice 3 on it.

---

### F14 — WRONG. Three code claims are wrong as written.

- **A wrong citation.** §1: *"the renderer does `Mesh::from(prim.vertices)`"*, cited to
  `crates/client-render/src/lib.rs:820-821`. Those lines build a stub reference ground plate
  (`Cuboid::new(GROUND_HALF * 2.0, 0.2, GROUND_HALF * 2.0)`). The mesh build is `mesh_from_prim` and
  `mesh_from_vertices` at `crates/client-render/src/lib.rs:1828-1841`. The claim is true. The pointer is
  not.
- **A needless widening.** §1: *"The smooth lane emits `MeshPrim`s (widened with normals)."* `Vertex`
  already carries a normal: `crates/client/src/realm_scene.rs:770-773` — `pos: [f32; 3], normal: [f32;
  3]`. Nothing widens.
- **A hash that is not the same hash.** §1: *"The noise bench inlines exactly it
  (`scripts/noisebench/src/main.rs:22-35`). The terrain generator reuses it; no second hash."* The bench's
  `mix64` is only SplitMix64's FINALIZER. It omits the state step
  `state.wrapping_add(0x9E37_79B9_7F4A_7C15)` that `crates/core/src/rng.rs:23-28` performs, and the
  bench's `hash3` adds its own golden-ratio multiplies per axis. A stateful stream generator and a
  stateless position hash are two different functions. The choice sits inside the pinned-digest door, so
  it must be named exactly.

---

### F15 — WRONG. The cell record's bit arithmetic contradicts itself and the base's allocation.

**The claim.** §3.1's field table gives orient *"6 bits"*. §4.1 of the same report rules the opposite:
*"take §2.1.1's shape … The sixth bit is reserved and must be zero."*

**What the base already did with that bit**, verified at `block_system_design.md:716*`: *"Dropping the
sixth orientation bit frees one bit in the 5-byte persisted record, and §2.7 spends it."* The bit is
spent. The report reserves it a second time.

The report also gives no width sum. It writes *"reserved | the rest of R1's slack"*. §7 makes the density
byte a one-way door that the formats domain freezes, and V3 asks for the saved block record as one of the
three frozen formats. A frozen record with no arithmetic is not a format.

**Fix.** Say 5 bits in both places. Name what §2.7 holds. Give the sum against the 5-byte record: 16
identity + 5 orient + 8 density + what §2.7 took + slack = 40.

---

### F16 — BREAKS_LAW. The smooth lane's HR4 gate is empty.

**The claim.** §11 slice 2's gate: *"the same fixture on a Spherical and a Cartesian profile (the
Cartesian one emits nothing)"*.

**The law.** CLAUDE.md HR4: *"every feature passes the identical fixture on ≥2 shard kinds (G-IDENTICAL)
or it doesn't land."* A run that emits nothing does not pass the fixture. It is absent from it.

**Fix.** Either state which fixture gives the smooth lane a real second shard kind, or say plainly that
the smooth lane's HR4 obligation is discharged by the cube and shaped lanes of slice 6, and move the gate
there. Do not present an empty run as a pass.

---

## 2. What survives

- **The representation choice.** Candidate (A) — a substance plus one signed sample per cell, meshed by
  naive surface nets — beats candidate (B). The report's reasoning is right, and the base's own words
  support it: a smoothing pass over a binary field turns every cliff into a 45° ramp
  (`block_system_design.md:8998-9001`, verified). *Example: a player digs a scoop into a dune and gets a
  scoop, not a cube-shaped socket.*
- **Surface nets before dual contouring.** The operation list is short. There is no case table, and there
  is no solver where a platform library call can creep in. STANDS, if F11's fence is written properly.
- **No library adopted.** rapier3d and parry3d really are absent from the workspace (`Cargo.toml:41-78`,
  verified: postcard, glam, bevy_ecs, bevy, redb, noise, brahe; no rapier, no parry). The report says so
  at §1 and keeps the choice as owner decision D8 — *"nothing is adopted here"*. Correct conduct.
- **The dead `noise` dependency.** `Cargo.toml:78` declares `noise = "=0.9.0"`. No crate manifest lists
  it, and no source names it. Verified. The report is right to call it dead weight.
- **The stale-claim table (§10).** Every row I checked holds. `coarsen_level` really does live only in a
  tombstone that nothing produces (`crates/wire/src/intershard.rs:1174-1186`). The base really does
  contradict itself on orientation (`block_system_design.md:710-716` against
  `block_system_design.md:5379-5382`).

---

## 3. Completeness — the cases the domain still owes

| Case | State in the report | What it needs |
|---|---|---|
| A mined cell under a placed block | Covered, §3.3 Case 3 | — |
| A smooth voxel heaped against a hull wall | Covered, §3.3 Case 2 | — |
| A square block set into a slope | Covered, §3.3 Case 1 | — |
| **A sub-metre block on a slope** | **Absent.** §4.3 makes `SubGrid` a form; §3.3's cases cover only `Cube` | Does a `SubGrid` cell keep its density byte? Does the smooth surface run through its sub-cells? What is the collider where a half-metre trim piece meets a 30° hillside? V2.4 is an owner requirement, so the case is reachable. |
| **A tree on a mined edge** | Half absent. §3.5 says the anchor cell's density gives the trunk's base height, then hands the rest to the tree domain | The terrain owes the interface: what the trunk's base height becomes when a player mines the anchor cell, and what the surface reflow does to a tree that already stands. |
| **An edit during a crossing** | **Absent** | See F9. A diff issued while a subscription flips must not be lost, or the client's shape stops matching the shard's collider. |
| A HUD on a rotating turret | Correctly out of scope | — |

---

## 4. What the report must do to stand

1. Write the SL6 ask for the three wire surfaces (F1).
2. Re-derive the collider's residency from the tick's swept segment, not a fixed radius (F2).
3. State and gate the rung-0 containment invariant, and measure the see-versus-stand gap (F3).
4. Make the FrameSpace seam slice 0, with the reanchor-forcing HR4 fixture as its gate (F4).
5. Fix the apron formula's unit (F5), the byte's name (F6), and its quantisation step (F7).
6. Store deltas in the pyramid, not absolutes, and gate the rim step (F8).
7. Name the diff baseline and budget it (F9).
8. Measure both cave-lattice steps before the digest is pinned (F10).
9. Replace "by construction" with the named fence plus M3 (F11), and price the spherical projection (F12).
10. Add the descent-rate bench (F13).
11. Correct the three code claims (F14), and close the record's bit arithmetic (F15).
12. Give the smooth lane a real HR4 fixture, or move the obligation and say so (F16).
