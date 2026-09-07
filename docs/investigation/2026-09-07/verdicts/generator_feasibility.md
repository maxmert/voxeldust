# Verdict — feasibility refutation of `03_generator_sl10.md`

**Date:** 2026-09-07. **Lens:** feasibility refuter, domain "the one generator under SL10".
**Target:** `docs/investigation/2026-09-07/03_generator_sl10.md`.
**Result: REFUTED.** The report holds four load-bearing claims that are wrong, one law it breaks, and
three cases its domain needs and does not carry.

I read the report. I checked every code citation with `grep` and `sed`. I checked every number against
its source. I state each finding with a `file:line`. I mark each number MEASURED or ESTIMATED.

---

## 1. What the report gets right (so the owner does not throw away the good parts)

These claims hold. I measured each one.

| Claim | I checked | Verdict |
|---|---|---|
| No terrain generator exists today. The only generator is the region forest. | `crates/physics/src/worldgen/generate.rs` is 1973 lines (MEASURED, `wc -l`); it calls `ln`, `cos`, `sin`, `tan`, `powf`, `powi` at lines 571, 598, 633, 639, 807, 1727, 1880. | STANDS |
| A planet's radius comes from `powf`, so the generator may not import it. | `crates/physics/src/taxonomy.rs:616` is `coeff * x.powf(exponent)`; the file holds 30 transcendental calls (MEASURED, `grep -c`). | STANDS. ⚠ The report writes the path as `taxonomy.rs` in a row headed `crates/physics/src/worldgen/`. The file is `crates/physics/src/taxonomy.rs`. Two files carry that name (`crates/core/src/taxonomy.rs` is the other). Fix the citation. |
| The client already holds each realm's seed. | `crates/core/src/pose.rs:86-113` (`FrameRef::PlanetCentered { planet_seed }`) and `crates/core/src/pose.rs:36-60` (`RealmId::Planet(seed)`, `Station(u64)`, `Area(u64)`, `Star(u64)`, `Galaxy(u64)`). A moon the player only LOOKS at arrives as `RealmId::Planet(seed)` on a body row. | STANDS. No seed needs to cross. |
| The world-generation tag refuses a peer and a file, never a player's client. | `crates/core/src/store_stamp.rs:262-267` refuses the file; `crates/io-prod/src/trust.rs:41-43,209,229` puts the tag in the intershard ALPN; `crates/wire/src/version.rs:337-350` shows `ProtoVersion` carries `major`, `minor`, `coordinate_generation` and nothing else. | STANDS. The report correctly corrects the ruling's own sentence. |
| `noise = "=0.9.0"` is declared and unused. | `Cargo.toml:78` declares it; `grep -c 'name = "noise"' Cargo.lock` returns **0** (MEASURED). | STANDS |
| No rapier dependency exists. | `grep rapier Cargo.toml crates/*/Cargo.toml` returns one comment at `Cargo.toml:96` (MEASURED). | STANDS |
| The client crate is engine-free and `vd-physics` is dev-only. | `crates/client/Cargo.toml` `[dev-dependencies] vd-physics`. | STANDS |
| The coverage recipe excludes `/tests/`, so a golden literal cannot redden HR5. | `justfile:31-33`, `--ignore-filename-regex '(/bin/|/tests/)'`. | STANDS |
| Minor 18 was a flag day, so the project has moved the floor before. | `crates/wire/src/version.rs:167,176`; the floor is 24 today (`version.rs:335`). | STANDS |

The dependency options A/B/C (§1.3), the four-layer fence idea (§2.2) and the honest UNMEASURED
register (§6) are sound work. The refutation below does not touch them.

---

## 2. The refutations, strongest first

### F1 — The crate ships the cells and not the SURFACE, so the boots and the pixels part company

**The claim.** §1.2 line 67 puts meshing outside the crate: *"Meshing, materials, textures, decoration
… `vd-client` (mesh) and the renderer. Display. Not shape. The server never needs it."* The C surface
(§1.5, lines 133-142) has ten functions and none of them returns a surface.

**Why it is wrong.** The sibling report in the same batch,
`docs/investigation/2026-09-07/02_smooth_terrain.md:32-40`, states the opposite and states it as the
recommendation: *"A smooth extractor turns the density lattice into the surface … **The same extracted
surface is the collider.** The realm's shard extracts the tier-0 surface … and hands it to the physics
engine as a static triangle mesh. Nothing else is the ground."* Line 112 of that report says the cube
collider under a smoothed picture **"Fails SL10 clause 5"**.

So under V2.1 (smooth realistic terrain) the extractor IS the shape. It is not display.

Three laws break at once:

1. **V1.5 breaks.** The moon's shard collides on the extracted triangles. The client draws the extracted
   triangles. The report never puts the extractor under the `Gf` fence, never puts it in the golden
   digest and never puts it in the C surface. So the one thing a player stands on is the one thing the
   no-drift gate does not measure.
2. **V1.2 breaks.** An Unreal client gets cells through `vd_worldgen_chunk` and must build the surface
   itself. That is a PORT of the extractor into a second code base. V1.2 forbids a port by name.
3. **SL8 breaks.** A crack of half a cell between the drawn slope and the collided slope is the "jump"
   seam kind, and it appears exactly where a player lands.

**Game example.** A hull lands on a moon. The client runs surface nets and draws a rounded hillside.
The Unreal client runs its own ported surface nets and places one vertex a millimetre elsewhere. The
moon's shard runs the Rust extractor. The boots of the Bevy player rest on the drawn slope. The boots of
the Unreal player sink into it. The generator gate is green the whole time, because it compared cell ids.

**Fix.** Move the extractor INTO the crate, under `Gf`, in the golden digest, and add one function to
the C surface: `vd_worldgen_surface(body, key, tier, VdVertex* out, ...)`. Then re-read §1.2: only
materials, textures and V2.6 style stay outside.

**Verdict: BREAKS_LAW.**

---

### F2 — The chunk payload cannot hold what a planet cell holds, so 476 KB is the wrong number

**The claim.** §1.5 line 138: `vd_worldgen_chunk(..., uint16_t* cells, size_t cells_len)` — "62³ cell
ids". §1.5 line 150 and §6 U9 both use **476 KB** (62³ × 2 bytes).

**Why it is wrong.** `02_smooth_terrain.md:185` puts a **terrain density, 8 bits, `i8`, present on EVERY
planet cell** in the saved record, and says the density is what the smooth lane and the collider read.
`02:181-186` also carries a 16-bit identity and a 6-bit orientation. A `uint16_t` per cell carries the
identity alone. It cannot carry the density, and the density is the shape.

62³ = 238 328 cells (MEASURED, arithmetic). At 2 bytes that is 476 656 bytes. At the record 02 needs it
is at least 3 bytes packed, so about 715 KB, or 953 KB at a 4-byte aligned word. The report's C-ABI copy
bench (U9) is therefore sized against the wrong payload, and the "same 476 KB of cell ids" sentence in
the §1.5 example is a claim about a buffer that cannot exist.

**Game example.** The crate hands the client the ids for the hill's chunk: dirt, dirt, stone, air. The
client cannot tell whether the surface passes three tenths above the centre of the top dirt cell or
seven tenths below it. It draws a staircase where the report promised a hill.

**Fix.** Decide the chunk record with report 01 and 02 first, then re-write the C signature and both
numbers.

**Verdict: WRONG.**

---

### F3 — D4(a) and D8 cannot both be taken: `ProtoVersion::CURRENT` is a `const` on purpose

**The claim.** D8 recommends a **measured** arithmetic profile: eight chunks evaluated AT BOOT, folded
into `gen_tag` (§2.3, §3.2). D4 recommends **(a) append `world_generation` to `ProtoVersion`**.

**Why it is wrong.** `crates/wire/src/version.rs:355-364` declares
`pub const CURRENT: ProtoVersion` and folds the coordinate unit at COMPILE time. The doc comment at
`version.rs:361-362` and `:366-373` states the reason in the code's own words: *"Folded at COMPILE TIME
… so no caller can state it"* and *"a caller that could state it could state it wrongly, and the one
value this negotiation exists to protect would become the one value a test could fake."*

A digest measured at boot is not a `const`. Appending it to `ProtoVersion` forces `CURRENT` and the
`const fn speaking()` to stop being const, and forces the value to become caller-stateable. That
destroys the exact property the file was written to hold. The report cites `version.rs:339-347` as its
argument FOR option (a) and never reads the twenty lines below it.

Two smaller points ride along. `version.rs:10-14` states the project's own rule: *"New data rides a new
trailing variant, never a new field."* The report cites that block for the flag-day fact and not for the
rule. And a client that must evaluate eight chunks (ESTIMATED 7 ms, §2.3) before it can even send Hello
puts terrain work in front of the login handshake.

**Game example.** A player starts the client. Before the client says hello to the gateway it generates
eight chunks of the home moon to learn what its own CPU does. Then it must state the answer in a struct
the code says nobody may state.

**Fix.** Take D4 option (b): a trailing `ClientControlMsg::HelloWorld { tag }` right after `Hello`,
which is exactly the mechanism `version.rs:10-14` prescribes. Keep `ProtoVersion` const.

**Verdict: BREAKS_LAW.**

---

### F4 — A CPU-measured digest on the DURABLE stamp turns a build difference into data loss

**The claim.** §3.2's carrier table puts `world_generation'` — which folds the boot-MEASURED digest —
on the durable store stamp.

**Why it is wrong.** `crates/core/src/store_stamp.rs:262-267` refuses a file whose `world_generation`
differs. `store_stamp.rs:187` names the only remedy: *"VD_STORE_ALLOW_GENESIS to start it over.
Everything it held is discarded either way."* The report itself repeats this in §9: *"the inherited
fail-safe on refusal is DISCARD."*

So the design makes this true: **if one pod's CPU or one build's flags move a single byte of the boot
self-check, the moon's shard cannot open its own store, and the operator's only lever discards every
tunnel every player ever dug.** The measured digest is the right thing for a live handshake, where a
refusal costs a reconnect. It is the wrong thing for a saved file, where a refusal costs a world.

The same fold on the intershard ALPN (`trust.rs:41-43`) silently forbids a mixed-architecture cluster —
an arm64 pod and an x86-64 pod could never dial each other. That may be the owner's wish. The report
does not say it.

**Game example.** The cluster reschedules the moon's shard onto a node with a different chip. The shard
boots, measures a different digest, and refuses the moon's own redb file. The tunnels are still on the
disk. Nothing can open them.

**Fix.** The durable stamp folds the DECLARED crate version only (the epoch counter of §9). The MEASURED
boot digest rides the handshake and the ALPN, where a refusal is cheap and reversible.

**Verdict: BREAKS_LAW.**

---

### F5 — `min` and `max` are on the allowed list, and Rust documents them as non-deterministic

**The claim.** §2.1 line 165: allowed are `+ − × ÷ sqrt`, *"Plus `floor`, `trunc`, casts, `abs`, `min`,
`max`, comparisons."* §2.4 then argues the golden set is representative *because* nothing but those
operations exists on the path.

**Why it is wrong.** Rust's standard library documents `f64::min` and `f64::max` with this note: if the
inputs compare equal — which is exactly the case of `+0.0` against `−0.0` — **either input may be
returned non-deterministically**. This is a documented property of the API, not a rumour. It is the one
operation on the report's allowed list whose result is not fixed by IEEE-754 alone, and it therefore
punches a hole in the §2.4 representativeness argument, which is the argument that lets a 64-chunk
golden set stand for a galaxy.

A terrain generator meets this case constantly. Sea level is an integer metre (§1.1). A clamp such as
`height.max(sea_level)` on a beach cell where the height is exactly sea level is precisely the equal-
inputs case, and the sign of the zero decides which of two identical-looking values flows into the cell
id and into the density byte.

**UNMEASURED note.** I did not run a build (my task forbids it). I refute the ARGUMENT, not the binary.
The report presents the argument as sufficient; it is not.

**Game example.** A player walks the shoreline of the moon's sea. On the Mac client the beach cell is
water. On the Windows client the same cell is sand. Both binaries pass the golden gate, because no
golden chunk sat on the waterline.

**Fix.** Delete `min` and `max` from `Gf`. Write `if a < b { a } else { b }`, which is fully determined
by the comparison. Add a waterline chunk and a sea-level clamp column to the golden set.

**Verdict: BREAKS_LAW.**

---

### F6 — "Under `no_std` a libm call is a COMPILE ERROR" is not true, and `no_std` deletes `sqrt`

**The claim.** §2.2 layer 3: *"make the generator crate `#![no_std]`. On stable Rust, `f64::sin` and
friends are `std`-only, so a libm call is a COMPILE ERROR, not a lint."* D6 recommends **Yes**.

**Why it is wrong, twice.**

1. `no_std` removes the std INHERENT METHODS. It does not remove the C symbols. A crate under `no_std`
   may still write `extern "C" { fn sin(x: f64) -> f64; }` and call it. The fence is weaker than the
   report states. The link scan of layer 3 is the honest fence, not `no_std`.
2. On stable Rust `f64::sqrt` and `f64::floor` live in `std`, not `core` (the `core` float-math methods
   are behind an unstable feature). V1.4 ALLOWS `sqrt` by name. So `no_std` would delete an operation
   the law grants, and the crate would then need either a nightly feature (the toolchain is pinned
   stable 1.94.1, `rust-toolchain.toml`) or the `libm` CRATE — a NEW dependency, which the standing rule
   says must be offered to the owner as an option and never adopted silently. The report offers the
   `no_std` recommendation without naming that consequence.

The report does mark this UNMEASURED as U4, which is correct method. The defect is that D6 recommends
"Yes" while the sentence justifying it is false and the cost of "Yes" is unnamed.

**Game example.** A contributor makes the crate `no_std` to be safe. The height field for the moon's
hill needs one square root for a direction vector. The build stops. The only stable cure is a new crate
in the product graph the owner never approved.

**Fix.** Drop D6 to "No" until U4 measures, and re-state D6's option list as: (a) `std` plus the lint
plus the link scan; (b) `no_std` plus an owner decision on a `libm`-class dependency. Keep the link
scan whichever wins, because only the link scan sees a symbol.

**Verdict: WRONG.**

---

### F7 — The MEASURED 0.885 ms does not measure the hash the report mandates

**The claim.** §0 row 32 calls `scripts/noisebench` *"the seed of the crate, not a stand-in"* and §2.1
line 160 cites `noisebench/src/main.rs:22-35` as already obeying "every draw is `SplitMix64` /
`child_seed`". D5 recommends vendoring the bench and says *"the bench's measured numbers apply as-is"*.
§2.3 derives the 7 ms boot self-check from the 0.885 ms figure.

**Why it is wrong.** The bench's hash is NOT the repo's hash. `scripts/noisebench/src/main.rs:22-35`
declares its own `mix64` (the avalanche only) and a `hash3` that calls `mix64` **three times** with three
multiplier constants. The repo's primitive is `SplitMix64::next_u64`
(`crates/core/src/rng.rs:22-28`), which begins with `state.wrapping_add(0x9E3779B97F4A7C15)`, and
`child_seed` (`rng.rs:70-74`), which is three `SplitMix64::new(..).next_u64()` calls. These are different
functions with different per-corner costs. The bench's own header even says the avalanche is *"inlined"*.

So two things follow. The bench, as written, is itself a second hash — the thing design rule 8 and HR3
forbid. And the 0.885 ms figure is a proxy for the proposed crate, not a measurement of it. Every number
built on it (the 7 ms boot check, the ~0.5 s gate leg, the U5 baseline) is ESTIMATED at one further
remove than the report says.

**Game example.** The crate draws the gradient for one lattice corner of the moon's hill. The bench drew
it with three folds. The crate draws it with `SplitMix64::new(...).next_u64()`. The two produce different
hills at different speeds, and only one of them was ever timed.

**Fix.** Re-run `noisebench` with `vd_core::rng` as the hash before any number from it is quoted. State
the 0.885 ms as "measured on a DIFFERENT hash" everywhere it appears until then.

**Verdict: UNMEASURED_AS_FACT.**

---

### F8 — 379 is a residency, not a refresh rate, so the owed bench U5 is mis-sized

**The claim.** §6 U5: *"at 240 m/s, ~379 new columns per rung per ring step (design §3.5.3)"*.

**Why it is wrong.** `docs/investigation/block_system_design.md:4152-4155` states the number plainly:
*"every rung above the first HOLDS about 379 chunk columns, regardless of which rung it is. Tier 0 is a
disc and holds about 505. Both figures are independent of the planet's radius."* 379 is the count of
columns RESIDENT in a rung's annulus. It is not the count that arrives per step of travel.

The mistake matters, because U5 is the measurement the whole client-derivation case rests on. A bench
sized on a residency instead of a flux measures the wrong thing.

The report also misses the one good SL9 fact in that same passage: the residency is **independent of the
planet's radius**, so the client's generation cost grows with the observer, not with the moon. §7's SL9
line only talks about feature anchors and leaves the strongest SL9 evidence on the floor.

**Game example.** A hull cruises over the moon at 200 m/s. The rung two levels up holds about 379 chunk
columns whether the moon is 161 km across or 6371 km across. How many of those columns are NEW each
second is a different number, and the report never names it.

**Fix.** Re-write U5 as a flux: columns entering a rung's annulus per second at the stated speed. Cite
`block_system_design.md:4152-4155` for the residency separately, under SL9.

**Verdict: WRONG.**

---

### F9 — 240 m/s is a bandwidth ceiling, not a mesher ceiling

**The claim.** §5.3 sets the pop-detector gate speed at *"240 m/s (the mesher-bound flight ceiling the
investigation base names, ESTIMATED)"*.

**Why it is wrong.** `block_system_design.md:7969` and `high_speed_flight_latency.md:370` both name
240 m/s as the **terrain UPLOAD bandwidth ceiling at the V1 vertex format** — the GPU byte budget, not
the mesher. The generation-and-mesh ceiling in the same base is **528 m/s pessimistic**
(`block_system_design.md:7702,7709`), and `high_speed_flight_latency.md:377` says a V2 vertex format
moves the bandwidth ceiling out of the way entirely.

The label is load-bearing, because a gate speed set from the wrong ceiling either under-tests the
generator or over-tests the wire.

**Game example.** The pop detector flies the hull over the moon at 240 m/s and calls the ladder proven.
The real generator limit sits at 528 m/s, and a warp arrival is faster than either.

**Fix.** Name both ceilings and their sources, and set the pop gate on the generator's own ceiling.

**Verdict: WRONG.**

---

### F10 — "Rosetta executes x86-64 SSE arithmetic exactly" is stated as a fact and carries a decision

**The claim.** §2.4 leg G4: *"Rosetta executes x86-64 SSE arithmetic exactly; a QEMU fallback also
implements IEEE semantics in software."* D7 recommends **(a) emulation on this Mac, now**.

**Why it is not sound.** The report itself measured that Docker was not running, so which emulator would
even run is UNMEASURED (the report says so in §0 row 31 and U10, to its credit). The sentence about
Rosetta is nonetheless written as an established fact and it is the reason D7 recommends (a). Subnormal
handling, flush-to-zero and denormal-as-zero flags are the classic divergence between a real x86 part
and a translation layer, and a height field that divides two nearly-equal metres can produce a subnormal.

The report DOES say "this is a measurement of an EMULATOR, and the report must say so". It then does not
carry that qualification into D7's recommendation.

**Fix.** Restate G4 as necessary-and-not-sufficient. Add to §9 a door: an emulated green may not be
called "no drift on x86-64"; a real x86-64 host is owed before the Unreal client links.

**Verdict: UNMEASURED_AS_FACT.**

---

### F11 — §7 answers "SL6: NONE requested" while the report requests two wire changes

**The claim.** §7 first bullet: *"SL6 — new data across a boundary: NONE requested."*

**Why it breaks the law.** CLAUDE.md line 210 states SL6 in full: *"ASK BEFORE NEW DATA CROSSES A REALM
BOUNDARY, **and before adding a wire arm**. Default NO. … State what data, from which realm to which,
why the receiver cannot compute it from what it legitimately holds, and what doing without costs."*

The report proposes two wire changes:

1. A new TLV tag `TAG_SURFACE` in a realm's look bag (§1.4, §3.2), beside `TAG_LOOK`
   (`crates/core/src/look.rs:31`), carried by `BodyStmt::SelfLook`
   (`crates/wire/src/session_flow.rs:626-630`).
2. A new field on `ProtoVersion`, or a new `ClientControlMsg` variant (§3.2, D4) — a change to
   `crates/wire/src/channels.rs:45-49`.

Both are wire arms. Neither is written in SL6's five-part form. Declaring "NONE requested" while
requesting two is the procedural defect, not the requests themselves — the requests look justified.

**Game example.** The moon states in its own look bag: *"draw me from my seed, at generator tag X."*
That is a new sentence the moon says to the window that never existed. It may well be right. SL6 says
the owner says so, in writing, before it is built.

**Fix.** Re-write §7's first bullet as two SL6 asks with data, sender realm, receiver, the reason the
receiver cannot compute it, and the cost of doing without.

**Verdict: BREAKS_LAW.**

---

## 3. What the domain needs and the report does not carry

### M1 — The composition order is never fenced (a mined cell under a placed block)

The report fences the GENERATOR. It never fences the composition `generated ⊕ diff`. Under V2.1 a placed
terrain voxel reshapes the surface, and `02_smooth_terrain.md:45-49` says the density survives UNDER a
placed square block and returns unchanged when the block is broken. So the composed density decides both
the picture and the collider — and that composition runs on the moon's shard and on the client, in two
code paths the golden digest does not cover.

Nothing in §4 states the order of application (generated shape, then cell edits, then placed blocks,
then sub-metre blocks), and an order is exactly what two hosts must share.

*Game example.* A player mines a cell, then sets a steel foundation into the hole. The client applies
the block first and the edit second; the moon's shard applies the edit first and the block second. The
two get different densities in one cell. The player's boots stand where the shard says and the picture
shows where the client says.

**Fix.** Name the composition order in the crate, put a composed chunk (generated + a mined cell + a
placed block) in the golden set, and gate on it.

### M2 — A seed-placed feature anchor whose ground was mined

§1.1 makes the feature anchor `f(seed, chunk)` and §4.2 makes the growth stage a diff. Neither says what
happens when the cell the anchor sits on is air because somebody dug it out.

*Game example.* A player mines the ridge under a pine. The moon's shard still draws the anchor from the
seed. Does the pine float, fall or vanish? The report has no rule, and the client and the shard must
agree on it or the pine's collider and the pine's picture separate.

### M3 — The handover above the coarsest rung (a planet at 10 000 km in the window)

§5.1 gives `R_L = 786 · 2^L` metres. `block_provenance_collapse.md:953` and
`block_system_design.md:1542` say the ceiling is the body's derived `tier_depth`, about 13 on the starter
world. So the coarsest rung reaches roughly 786 · 2¹² ≈ 3 200 km (ESTIMATED, arithmetic). Beyond that the
realm proxy draws — one fixed tessellation today, `crates/client/src/realm_scene.rs:800-803`.

§5.3's crossfade covers rung `L` against rung `L−1` of the SAME terrain. It says nothing about the
proxy-to-coarsest-rung handover, which is the "arrival pop" and "detail-by-box" seam kinds of SL8, and
which is the FIRST thing a player sees when a hull comes out of warp.

*Game example.* A hull warps in and the moon is 10 000 km away. The client draws the proxy sphere. At
3 200 km the coarsest generated rung takes over. The report sets no tolerance for that frame.

### M4 — Which thread evaluates a chunk, and what its budget is

§1.4 says the client *"evaluates chunks on a worker pool"*. That is the whole statement. There is no
thread count, no per-frame budget and no owner for the pool.

Today `vd-client` spawns no thread (MEASURED: `grep -rn 'std::thread\|thread::spawn\|rayon' crates/client/src/` returns nothing), holds no `rayon`
dependency (MEASURED: `grep -n rayon Cargo.toml` returns nothing), and its clippy fence bans the clock
and `thread::sleep` because *"the lib never sleeps; pacing belongs to the bin's render loop"*
(`crates/client/clippy.toml:6-10`). A worker pool inside a Tier-A crate that must reach 100 % region and
branch coverage (HR5) is a design question, not a detail.

The base's own figure is the reason it matters: `decision_board.md:102` says a 6 800-chunk 100 km view is
**~6 s on one core and 0.75 s on eight** (MEASURED 2026-08-03, M4 Pro). Eight cores is an assumption
about the player's machine that no ruling has made.

*Game example.* A hull descends toward the moon. The client must fill the tier-0 disc — about 505 chunk
columns — before the boots touch. On one core that is seconds. Who runs them, and while what is drawn?

### M5 — Delegated, and named so nobody assumes I checked them

A sub-metre block on a slope, a HUD on a rotating turret, and an edit during a crossing belong to
reports 01, 04, 05 and 06. Report 03 does not carry them, and I do not hold that against it. I record
them so the owner sees they are asked somewhere.

---

## 4. The one-way doors, re-checked

| Door in §9 | Real? | Deadline right? | Retrofit cost stated? |
|---|---|---|---|
| Output frozen at the first saved diff | Yes. `store_stamp.rs:262-267` refuses the file; `:187` says the remedy discards. | Yes, and the report's sharper form ("at the moment the golden literals are committed") is the right one. | Yes. |
| Body definition inside the crate, clause-4 clean | Yes. `crates/physics/src/taxonomy.rs:616` is `powf`. | Yes. | Yes. |
| The concealed tier never enters the grid | Yes. `concealed_resources.md:520` gives the 1 596 × ρ leak. | Yes. | Yes. |
| The public/valuable material line | Yes. | Yes. | Yes. |
| The client-facing tag carrier | Yes, but the deadline is wrong given F3: option (a) is not available at all while `ProtoVersion::CURRENT` is a `const`. The door is "which mechanism", not "when". | Re-state. | Partly. |
| The dependency shape A/B/C | Yes. | Yes. | Yes. |
| **MISSING: the extractor's placement (F1)** | This is the biggest door in the domain and it is not in the table. Moving the extractor into the crate after an Unreal client links the staticlib means a second code base rebuilds and every saved diff is re-validated against a surface that moved. | Before the C surface is frozen. | Not stated. |
| **MISSING: the chunk record's cell width (F2)** | Freezing `uint16_t` and then adding the density byte changes every ABI signature and every golden literal. | Before the ffi crate exists. | Not stated. |

---

## 5. Scale, under SL9

I checked whether any cost in the report grows with the planet's size or with a realm's child count.

- Generation residency does NOT grow with the moon's radius: 505 columns at tier 0 and about 379 per
  rung above, *"independent of the planet's radius"* (`block_system_design.md:4152-4155`). Good, and the
  report failed to claim its own best evidence (F8).
- The golden set grows as `64 × tier_depth`, and `tier_depth` grows as the logarithm of the radius
  (about 13 on the starter world, `block_provenance_collapse.md:953`). Logarithmic is acceptable. But
  §2.4 costs the gate at *"64 × ~8 tiers"* while §2.3 says `64 × tier_depth`. At 13 the leg is 832
  digests, not 512 — a 60 % under-estimate inside one report (MEASURED, arithmetic on the report's own
  two sentences).
- Feature anchors are a bounded per-chunk draw, never a walk of a realm's children. Correct under SL9.

No cost grows with the child count. SL9 holds for this domain.

---

## 6. Verdict

**REFUTED.** Four load-bearing claims are wrong (F1, F2, F7, F8), four break a law (F1, F3, F4, F5,
F11), two state an unmeasured thing as a fact (F7, F10), and four cases the domain needs are absent
(M1–M4).

The report should not be adopted as it stands. F1 alone re-opens the crate boundary, the C surface and
the golden set, because the surface a player stands on is not in any of the three.

The recoverable core is real: one crate, one hash, the four-layer fence, the byte-for-byte gate, the
diff lane split, and an honest UNMEASURED register. Fix the boundary first (F1, F2), then the tag
carriers (F3, F4), then the fence hole (F5, F6), then re-cost the benches (F7, F8, F9).
