# Verdict — the law refuter on "03 — The one generator under SL10"

**Date:** 2026-09-07. **Lens:** the binding law (HR1–HR6, SL1–SL10, SL8, the owner rulings).
**Report under test:** `docs/investigation/2026-09-07/03_generator_sl10.md`.
**Verdict: REFUTED.** Two load-bearing claims break a law. Two more claims about the code are false.

I re-read every ruling. I then read the code for every `file:line` the report cites. The report's
census of the code is good: almost every citation is true. The design on top of that census has
five holes. Two of them put the player's boots on a shape the player does not see, which is the one
thing SL10 exists to stop.

---

## F1 — BREAKS_LAW. The mesher lives outside the one crate, so the two hosts build two surfaces

**The claim.** §1.2 puts meshing outside the generator crate: *"Meshing, materials, textures,
decoration … `vd-client` (mesh) and the renderer. Display. Not shape. The server never needs it."*
§5.1 says the server's collider builder takes a `Tier0Chunk`. §1.5's C interface returns
`uint16_t* cells` — a block of 62³ cell ids and nothing else.

**Why it breaks the law.** V2.1 asks for a SMOOTH surface built of voxels. A smooth surface is not
the cell array. It is the isosurface a rule draws THROUGH the cell array. If the client's mesher
holds that rule and the server's collider does not, then two implementations produce the surface,
and V1.5 fails: *"What the player sees is what the player stands on."* The player sees a smooth
hillside and the boots stand on a stair of one-metre boxes. SL8 names that class of defect twice
(jump, detail-by-box). The FINAL-backend law forbids the second implementation outright.

**Game example.** A miner walks up the moon's hill. The client meshed the hill smooth from the cell
ids. The moon's shard built its collider from the same cell ids as boxes. The boots climb a step the
eyes never saw, and the camera sinks a half metre into a slope that looks flat.

**The fix.** The rule that turns cells into a surface belongs INSIDE the one crate, beside the height
field. Either the crate returns the surface (an isosurface the two hosts share), or the crate returns
a density value per cell and both hosts read the surface from it by one shared function. Only the
material choice, the texture and the style (V2.6) stay on the client. The C surface then needs a
density or a surface output; `uint16_t cells` alone cannot carry it.

---

## F2 — BREAKS_LAW. A moon gets two radii, and only one of them is drawn

**The claim.** §1.1: the body definition, *including the radius on the integer ladder*, is computed
INSIDE the crate and *"may NOT be imported from `taxonomy.rs`, which uses `powf`"*. §1.3's example:
*"The moon's radius came out of the crate's own `body_definition(seed)`, not out of `taxonomy.rs`."*

**What the code says.** Today a planet's and a moon's radius is `taxon.radius_m`, and it becomes the
realm's LOOK: `look: Some(shell(taxon.radius_m))` (`crates/physics/src/worldgen/generate.rs:1232`)
and `look: Some(Boundary::Shell { r: taxon.radius_m })` (`generate.rs:1819`). That number comes from
`composition_radius_m` (`crates/physics/src/taxonomy.rs:934`, `:984`), which goes through
`segmented_power_law` and its `coeff * x.powf(exponent)` (`taxonomy.rs:616`). The look is lowered
onto the forest row (`crates/physics/src/worldgen/body.rs:56`, `:252`) and the parent reads it to
decide visibility (`crates/physics/src/worldgen/visibility.rs:85`, `:184`).

**Why it breaks the law.** The report adds a second computation of one world fact and never says
which one wins. SL5 says one world and no second generator. HR3 says one tooling. The consequence is
physical: the parent tests reach against radius A, the client draws the realm's outline at radius A,
and the terrain sits at radius B. A hull descending on the moon sees the ground poke through the
moon's own drawn edge, or float above it. That is a seam, and SL8 calls a seam a defect.

**The fix.** Name the one owner of a body's radius. The clean answer is that the generator crate owns
it and the forest READS it, so `taxon.radius_m` stops being a second source. State the migration: the
radius of every existing body moves when this lands, so the world epoch (§9) must open at the same
moment.

---

## F3 — WRONG. The gateway and the orchestrator cannot skip the boot self-check

**The claim.** §2.3: *"every process (shard, gateway is storeless and skips it, client, Unreal
client…) evaluates 8 of those chunks at boot and folds the digest into the tag"*. §3.2 folds that
digest into `world_generation'`, and says the intershard ALPN carries it with *"no code change beyond
the fold (`bins/src/lib.rs:970`)"*.

**What the code says.** The tag on the mesh handshake is one value for every node kind. The gateway
calls it (`crates/bins/src/bin/gateway.rs:271`, `:318`), the orchestrator calls it
(`crates/bins/src/bin/orchestrator.rs:86`, `:596`), the client bin calls it
(`crates/bins/src/bin/client.rs:106`), and it rides the ALPN string
(`crates/io-prod/src/trust.rs:41-46`, `:209`, `:229`). A gateway that skips the self-check folds a
different number and refuses every shard on the mesh.

**Game example.** The moon's shard boots, evaluates its eight golden chunks and offers
`vd-intershard/1+unit-…+world-…`. The gateway skipped the check, offers a different `world-…`, and
the TLS handshake fails. Nobody sees the moon at all.

**The fix.** Say it plainly: EVERY node that speaks on the mesh links the generator crate and runs the
self-check, the storeless gateway and the orchestrator included. Then state the cost the report never
states — the orchestrator, which draws nothing, now carries the terrain generator and pays ~7 ms at
boot (the report's own ESTIMATE). If that cost is unwanted, the measured profile must ride a
different carrier than the ALPN, and the report must design that carrier.

---

## F4 — WRONG. The "WHAT lane" reservation does not stand; it is a tombstone of an SL2 breach

**The claim.** §8's last-but-two row: *"The field exists at `crates/wire/src/intershard.rs:1185` now;
the line numbers moved. The reservation stands."*

**What the code says.** Line 1185 is `pub coarsen_level: u8`, and it sits inside
`struct OccupantInterest`, whose own doc reads: *"★TOMBSTONE payload (Step 5 slice D) … Kept only so
the reserved discriminant keeps a decodable shape; nothing produces it."* The field above it is
documented as *"The occupant pose the deleted lane shipped — the SL2 breach that condemned it"*
(`crates/wire/src/intershard.rs:1172-1186`). A grep finds the name only there and in three test
literals (`crates/wire/tests/intershard_closed.rs:405`,
`crates/sim/src/stub/tests/aoi_demand.rs:607`, `intershard.rs:2079`).

**Why it matters.** The report reads a dead lane as a live reservation and offers it to a later slice
as room to build in. Building on it would resurrect the very lane that shipped an occupant pose across
a realm boundary, which SL2 forbids.

**The fix.** Replace the row with the truth: the WHAT lane is DELETED, its discriminant is a tombstone,
and any coarse-detail request the terrain ladder needs is a NEW arm that SL6 must approve first.

---

## F5 — BREAKS_LAW. §7 declares no SL6 request while §3.2 and §1.4 each make one

**The claim.** §7, first bullet: *"SL6 — new data across a boundary: NONE requested."*

**What the report itself proposes.** §3.2 says the client-facing tag *"Must be added"*, and D4
recommends appending `world_generation` to `ProtoVersion` plus a move of `PROTO_MINOR_FLOOR`. §1.4
adds a new tag `TAG_SURFACE` to the realm's look bag.

**Why it breaks the law.** SL6 reads *"ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, and before
adding a wire arm. Default NO."* A new field inside `Hello` is a flag day on the frozen wire contract:
`crates/wire/src/version.rs:1-15` states the rule that *"New data rides a new trailing variant, never
a new field"*, and `PROTO_MINOR_FLOOR = 24` (`version.rs:335`) records what the last such append cost.
A blanket "NONE requested" in the law-conflict section hides both asks from the owner's eye, even
though the report raises them elsewhere.

**The fix.** Move both into §7 in SL6's own words: WHAT data (the folded world generation; the
realm's surface tag), FROM which realm TO which (the realm about itself, to its own subscribed
client; the build to the gateway at the handshake), WHY the receiver cannot compute it (the client
cannot know the cluster's crate version), and WHAT doing without costs (a client draws a hill the
shard does not have, and finds out when the boots fall through).

---

## F6 — MISSING. The client picks its detail by screen size alone, so a fast hull hits what it cannot see

**The claim.** §5.1: *"The client picks `L` by screen-space error"*, and *"A coarse rung is drawn only
where no one can touch it."* §5.1 also fixes the tier-0 disc at 786 m (ESTIMATED).

**Why it is missing.** The server always collides at tier 0 (§5.1, Addendum 2 C.7). The client draws a
coarse rung outside 786 m, where §5.2 bounds the height error at `4 · k_rough · 2^L` metres. A hull at
240 m/s crosses 786 m in about 3.3 s (ESTIMATED from the report's own two numbers). So for over three
seconds the pilot flies at a cliff the client draws up to metres away from where the shard will collide.
The 2026-08-27 movement ruling already settled the general shape of this: *"grow the interest radius
with the closing speed; never cap the speed."* The report never applies that rule to the tier choice.

**Game example.** A hull runs a canyon on the moon at 240 m/s. The client draws the far wall at tier 2
(4 m cells). The wall the shard collides on stands up to 8 m nearer. The hull explodes against air.

**The fix.** Make the tier choice read the closing speed as well as the screen error, exactly as the
interest radius does: the resident tier-0 range must cover at least the distance the occupant covers in
the time the crate needs to evaluate it. State the constant, and measure it in U5.

---

## F7 — MISSING. The HR4 second-profile fixture has no lawful subject

**The claim.** §7: *"HR4 — features once, run anywhere. The G-IDENTICAL fixture runs the chunk
evaluation on a Spherical profile and on a Cartesian profile (a station: a flat generated floor, or an
empty body) in the same slice."*

**Why it is missing.** HR4 says a feature passes the IDENTICAL fixture on two shard kinds or it does
not land. "An empty body" is not a pass — the feature does not run. "A flat generated floor" for a
station is new world content that no ruling asks for, and V2.3 says a station is BUILT from blocks.
Under SL5 an invented floor is a reduced world grown to satisfy a gate.

**The fix.** Name a real second subject. A moon and an asteroid are both Spherical. The honest
candidates are an AREA realm inside a planet (`FrameRef::AreaLocal`, `crates/core/src/pose.rs:105`),
which already carries its own seed, or a declaration that the surface capability is Spherical-only
and HR4 is discharged by a second REALM of the same profile with a different body definition. Either
way, say which, and say it before the slice starts.

---

## F8 — UNMEASURED_AS_FACT. An emulated leg is presented as the measurement, and one example states a gate result that does not exist

**The claim.** §1.5's example: *"The gate proved both produce the same 476 KB of cell ids … so the
boots stand where the pixels are."* §12: *"a byte-for-byte golden gate on aarch64-darwin,
aarch64-linux (the pod) and x86-64-linux (emulated today) measures it."* §2.4 G4 also asserts
*"Rosetta executes x86-64 SSE arithmetic exactly."*

**Why it fails.** No gate has run. The report's own table marks G4 OWED and marks the k3d image
architecture UNMEASURED, and U1 says the cross-architecture equality was *"NEVER measured anywhere in
this project"*. That is honest. The example and the summary paragraph then speak as if it had run.
V1.3 demands the gate on *"every target the game ships on"*. A shipped target is real hardware. The
Rosetta claim is an assertion about an emulator, not a measurement.

**The fix.** Strike "The gate proved" from the example and write "the gate will prove". In §12 mark the
x86-64 leg as OWED and NOT YET MEASURED. State explicitly that the emulated leg is a smoke test that
can only find a difference, never prove its absence, and that V1.3 stays open until one x86-64 machine
prints an equal digest.

---

## F9 — WRONG. Not every realm's frame carries a seed

**The claim.** §0: *"The client already holds every realm's SEED as its frame."*

**What the code says.** `FrameRef` has eight arms (`crates/core/src/pose.rs:87-113`). Six carry a
seed. `ShipLocal { ship: EntityId }` carries an entity id and no seed, and `UniverseSpace` is
fieldless.

**Why it matters.** The SL6 argument in §7 rests on this sentence. It holds for a moon, a planet, a
star, an area and a station. It does not hold for a hull, which is a realm like any other under the
realm-unification law.

**The fix.** Write "every SEED-GENERATED realm's frame carries its seed." Then add the consequence the
report already implies in its HR3 row: a hull realm has no body definition, so it never states a
surface tag, and its shape comes from its saved blocks as a diff.

---

## F10 — MISSING. An indicator material is a wiki-able hint with no owner decision

**The claim.** §1.1 puts inside the crate: *"Indicator materials (a discoloured stratum that raises
the live-deposit chance by a declared factor)."*

**Why it is missing.** The 2026-08-27 seed ruling says anything a fixed seed alone decides must be
SAFE TO PUBLISH, because one world plus many players makes any static map public by wiki. An
indicator whose multiplier is DECLARED and whose position is `f(seed, chunk)` is a static ranking of
where to prospect. It ships in every client's binary. Whether that is acceptable is exactly the class
of question the owner answered for materials in D3, and the report opens no decision for it.

**Game example.** A wiki lists every gossan on the moon by chunk key. Every prospector flies the same
list, and the survey instrument the concealment ruling paid for stops being worth carrying.

**The fix.** Open D11: how strong may a seed-derived indicator be? Options: none at all; a hint with a
small declared factor; a hint whose factor is itself live state. Recommend one, and let the owner rule.

---

## F11 — UNMEASURED_AS_FACT. The `Gf` fence is called structural without saying what makes it so

**The claim.** §2.2 layer 1: *"A `sin` is unreachable because `Gf` has none. This is … a structural
fence, not care."*

**Why it is short.** SL1 clause 5 and SL4 both demand a fence the compiler holds, with an
observed-failing control — the shape `tests/tests/crate_isolation.rs` already uses. A newtype only
fences if its inner `f64` is unreachable outside the crate. The report writes `Gf(f64)` and never says
the field is private, and never names the control that proves an escape fails.

**The fix.** State that the field is private, that the crate exposes no `From<Gf> for f64`, and that
the first commit adds an observed-failing control, the way the I/O ban was proven
(`crates/core/clippy.toml:1-5`, the SEAL-1 note).

---

## F12 — WRONG (minor). The SL4 fence is cited to a description, not to the gate

**The claim.** §7: *"The `crate_isolation` gate that fences `vd-physics` (`crates/physics/Cargo.toml`
description)."*

**What the code says.** `crates/physics/Cargo.toml:6` is a package description that MENTIONS the gate.
The gate itself is `tests/tests/crate_isolation.rs`, named as the holder of the law by
`crates/core/src/lib.rs:17`, `crates/core/src/worldgen.rs:9`, `crates/physics/src/lib.rs:12` and
`crates/wire/src/version.rs:834`.

**The fix.** Cite the test file. A row added to a description fences nothing.

---

## What stands

- **S1 — the correction of V1.3 stands, and it is the report's best work.** §3.1 and §8 show that the
  world-generation tag refuses a PEER through the ALPN (`crates/io-prod/src/trust.rs:41-46`,
  `crates/wire/src/admin.rs:373-376`) and a FILE through the stamp
  (`crates/core/src/store_stamp.rs:262-267`), and that the player client's handshake carries only
  `major`, `minor` and `coordinate_generation` (`crates/wire/src/version.rs:337-351`;
  `crates/wire/src/channels.rs:44-49`). The ruling's sentence is stale. The report says so plainly.
- **S2 — the code census is accurate.** I checked every citation in §0. `noise = "=0.9.0"` is declared
  at `Cargo.toml:78` and the lock holds zero `name = "noise"` entries (MEASURED: `grep -c`). No rapier
  edge exists (MEASURED: `grep rapier Cargo.toml crates/*/Cargo.toml` → one comment,
  `Cargo.toml:96`). `glam 0.30.10` sits in the lock and itself depends on `libm`, which strengthens
  the report's own link-scan layer. `HOME_SEED = 2298` is at
  `crates/physics/src/worldgen/census.rs:79`. The coverage recipe does ignore `/tests/`
  (`justfile:31-33`). `scripts/noisebench/Cargo.toml` declares no dependencies.
- **S3 — no pose, no velocity and no placement is derived on the client.** I looked for a fold of an
  absolute, for a velocity crossing upward, and for a parent that draws a child. The report proposes
  none. SL1, SL2 and the movement contract are untouched.
- **S4 — no port and no second language.** §1.5 keeps one Rust crate and one C edge, which is V1.2
  exactly.
- **S5 — a dormant realm is still not drawn.** The surface tag rides `BodyStmt::SelfLook`
  (`crates/wire/src/session_flow.rs:625-628`), which only a running realm may send, so the seed alone
  never lets the client draw a sleeping moon. The report should SAY this; it is a law the design keeps
  by luck of carrier choice rather than by statement.

---

## What the report must do before the owner reads it

1. Move the surface rule (the isosurface, or a density field both hosts read) INSIDE the crate (F1).
2. Name the one owner of a body's radius, and the epoch that moving it opens (F2).
3. Say that every mesh-speaking node links the generator and runs the self-check (F3).
4. Correct the `coarsen_level` row: the lane is dead, not reserved (F4).
5. List the two SL6 asks in §7, in SL6's own four parts (F5).
6. Make the detail tier read closing speed (F6).
7. Name a lawful second subject for the HR4 fixture (F7).
8. Strike every sentence that says a gate has proved something (F8).
9. Fix the "every realm's seed" sentence (F9), open the indicator decision (F10), state what makes
   `Gf` a fence (F11), and cite `tests/tests/crate_isolation.rs` (F12).
