# 10 — The law audit: the stale-text register

**Date:** 2026-09-07. **Revision:** 2 (after the law refutation and the feasibility refutation).
**Domain:** the law audit of the investigation base.
**Status:** an input to the synthesis. Nothing here is a ruling. Every row points at a ruling that binds.

## 0. What this report is, and how to read it

The investigation base (`docs/investigation/`, written 2026-08-03/04) predates every ruling from
2026-08-24 on. This report lists every place where the base and a ruling disagree, and every place where
a ruling changes an answer the base gave. The synthesis must not promote a row marked STALE or REFUSED.

Five words carry the verdicts:

- **ANSWERS** — a ruling closes the question. The base's options table is dead.
- **CHANGES** — the question stays open, but a ruling changes the premise, the options or the deadline.
- **REFUSES** — a ruling forbids the base's recommendation by name.
- **REFUSED-UNTIL-ASKED** — the law permits it only after an SL6 ask the owner has not answered.
- **OPEN** — no ruling since 2026-08-24 touches the row.

Every number is marked MEASURED (with how) or ESTIMATED. Every claim about the code cites `file:line`.

**Example, in the game's words.** The base says a rock that flies too fast into a planet's realm is slowed
to that realm's ceiling. The 2026-08-27 ruling says a transfer never changes a speed, because a re-clamp is
a jump and a jump is a seam. The code agrees with the ruling today: `crates/sim/src/stub/transient.rs:232-237`
records the clamp as deleted on 2026-09-05. So the base's sentence is STALE and the synthesis must not
copy it.

**Method.** I read the binding law first (CLAUDE.md HR1–HR6, SL1–SL10; the eleven `owner_decisions_*.md`
files; the DEFERRED rows that name P4/P5/P6, terrain, voxel, `FrameSpace` and the re-anchor). Then I
read `decision_board.md` §1, §2 (Bands A–E), §3, §5, §7; `block_system_design.md` lines 1–110 (the
rulings table), §1, §3.5.4, §3.6–§3.8, §3.10, §4.1, §4.7, §7.6, §7.13, §8.2, §8.6; both addenda; and the
topic files `high_speed_flight_latency.md`, `signal_authority_and_relays.md`, `concealed_resources.md`. I
did not read the whole 17,480-line body; the register rows pointed me at the body sections.

**What revision 2 changed.** Two refuters read revision 1. Both returned REFUTED. Revision 2 corrects
four wrong claims, replaces one unlawful licence with an SL6 ask, and adds nine missing rows. Section 8
lists every finding and what I did with it.

---

## 1. Facts about the code that decide rows below (MEASURED by `grep` and by reading the file on this tree, 2026-09-07)

| Fact | Where | Consequence |
|---|---|---|
| `InterShardFlow` has **44 arms** (counted with `awk` over the enum block) | `crates/wire/src/intershard.rs:124` | Door 38's "30-arm ceiling" is passed. O15's "27 + 3 = 30, no margin" is stale |
| The approval-citation gate is a REAL build test: `every_ledger_minor_from_9_up_carries_an_owner_citation` | `crates/wire/src/version.rs:800` | The successor to the arm count is a test, not a number |
| The saga step ids run 7..18; 19 upward is free | `crates/wire/src/intershard.rs:65-115` | O14's premise holds |
| `ChannelKey` does not exist | `grep -rn ChannelKey crates` → nothing | O12's premise holds |
| `FrameSpace`, `SphericalSpace`, `reanchor()`, `AnchorGen` do not exist as code. **Six doc-comment sites in three files** name them: `crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11,18,35,40`, `crates/sim/src/stub/transient.rs:239` | as cited | D-38 rules the seam is terrain's FIRST slice (owner 2026-09-05). `transient.rs:239` is the site that names the seed-derived analytic gravity P5 owes |
| `VoxelGeometry::{Spherical, Cartesian}` exists as profile data | `crates/sim/src/capability.rs:37-41` | The grid-family split is planted as data, not as code paths. HR4 binds it: the same fixture must pass on a planet realm and a hull realm |
| The movement lane exists: `ChildDrive { push: [i64;3], turn: [i64;3] }` per tick, `ChildFacts { mass_g, drag_micro, declared }` on change, `ReachStated { size_reach_m, light_reach_m }` on change | `crates/wire/src/intershard.rs:520, 538, 603-618, 1377` | The base's `Coupling(ShipThrustPort)` shape is superseded |
| The transient re-clamp is deleted; a held transient keeps its speed | `crates/sim/src/stub/transient.rs:232-237` | The base's speed-ceiling text is stale |
| **The walk's cap RUNS on the live path today.** `walk()` calls `ramp_cap_mps`, then `governed_ceiling_for_frame`, then scales the occupant's stick by `flight::throttle_axes_scale` | `crates/sim/src/stub/dot.rs:457-474`; `crates/core/src/flight.rs:93` (`realm_speed_cap_mps`), `:105` (`approach_ceiling_mps`), `:118` (`ramp_cap_mps`) | The suit ruling S3 deletes the ceiling on the stick; S6 DEFERS the deletion (`owner_decisions_2026-09-05_suit.md:69-71`, ledger D-MOVE-3). The cap is refused by the law and PRESENT in the code |
| `BuiltFacts` and `EngineRating` exist; no suit type exists | `crates/core/src/built.rs:37`, `crates/sim/src/stub/drive.rs:28` | The suit is later than blocks and terrain (S6) |
| The parent-authored marker: `BodyStmt::Marker` is a reserved tombstone; `marker_of` store remains as dead decode | `crates/connection-plane/src/window.rs:157`, `crates/sim/src/stub/window.rs:53,116,497-503` | R2's deletion is built (D-REACH-1 step 5); the base's "realm proxy = the parent's marker" is stale |
| **No realm-inbound message carries a placement or a centre, and a test pins the absence** | the `ChildSceneSet` tombstone `crates/wire/src/intershard.rs:416-440`; the test `no_realm_inbound_payload_carries_a_placement_or_a_centre` (`crates/wire/tests/intershard_closed.rs:485`) | A sibling's placement into a child has NO lane and would turn that test red. See §4 item 7, ask five |
| `RealmInterest` carries M7's one scalar: `how far away the nearest OUTSIDE LOOKER is`, no direction | `crates/wire/src/intershard.rs:1319-1342` | M7 is BUILT. It is a visibility scalar, not a distance between two named realms |
| **One O(children) fold survives.** `rebuild_static_rows_for(parent)` walks `children_of[parent]` in full; it runs on every adoption and every release of a NON-ship child | `crates/sim/src/stub/regions.rs:876`, called at `:752` and `:855` | MEASURED 2026-09-05: a hull's adoption into the galaxy (279,380 children) made the galaxy's slowest tick **60.0 ms** against a 20 ms budget (`DEFERRED.md:5419`). The fix exempted SHIPS only; the same note says "A planet's adoption still rebuilds its parent's layer" (`DEFERRED.md:5425-5427`), and the slowest tick fell to **4.5 ms** (`DEFERRED.md:5431`). Any voxel mechanism that MINTS a non-ship realm inherits the fold |
| `coarsen_level` is a legs-travelled diagnostic on a TOMBSTONED payload | `crates/wire/src/intershard.rs:1185` | O13's correction of Addendum 2 stands and is stronger: the field is dead |
| `pin_abs` and `anchor_epoch` were REMOVED at wire minor 8; the origin is the composed level's `origin, origin_epoch` | `crates/wire/src/version.rs:67-79, 171-173` | O16 / door 51 argue from a field that no longer exists |
| `sky_anchor: Option<StampedPose>` rides the realm datagram | `crates/wire/src/channels.rs:499` | R10 decision 2 is built |
| `BulkKind::{ChunkSnapshot, ChunkDelta, Catalog}` is reserved, unroutable | `crates/wire/src/channels.rs:292-296` | §3.10's carrier premise holds |
| Neither `rapier3d` nor `noise` is in `Cargo.lock` (0 hits each); `noise = "=0.9.0"` is declared in the workspace table and used by no crate | `Cargo.toml:78`, `Cargo.lock` | O7 and O28 are still library adoptions for the owner. This report adopts nothing |
| No generator digest is in the handshake; the handshake folds `coordinate_generation` only | `crates/wire/src/version.rs:350` | SL10 V1.3's "the tag that already refuses a client whose generator disagrees" names a tag I could not find. UNMEASURED which tag the owner means; the synthesis must build one |
| `two_level_clearance_m` / `galaxy_shell_r_m` are deleted | `crates/physics/src/worldgen/forest_query.rs:80` | The 2026-08-15 item-5 shell rule is gone, as R6 of the reach ruling orders |
| `WORLD_SYSTEM_COUNT` is deleted | `crates/physics/src/worldgen/generate.rs:56` | G8 is built |
| `region_verdict` takes `prev_pos` (swept) | `crates/core/src/geometry.rs:1527-1529` | M-D.3 holds; `k_dwell = 5` at `geometry.rs:938` is still sized on the deleted governor |
| The interest (D-9) is built: per-cube bodies to the observers inside the occupant's reach | `docs/design/DEFERRED.md:2925`; `crates/core/src/look.rs:63-70` | The board's "D-9 reshape is a precondition of block edits" is now satisfied |
| The galaxy tick with one looker: 20 µs, 5 candidates, 279,380 children (release; MEASURED 2026-09-05, `measure_the_galaxy_shards_tick_against_its_census`) | `docs/design/DEFERRED.md:5702-5705` | R8 item 1 is closed. The per-tick fold is cheap; the per-EVENT fold above is not |

### 1a. Base text that SURVIVES and matters (the register's other half)

A stale register must also name the base mechanisms a ruling KEEPS. These are live, not stale.

| Base mechanism | Where | Why it survives |
|---|---|---|
| **The `Gf(f64)` newtype** — a generator float that exposes `+ − × ÷`, `sqrt`, `floor`, `abs`, `min`, `max` and nothing else, so a transcendental cannot be reached on the generation path by construction | `docs/investigation/block_system_design.md:3758-3765` | SL10 V1.4 forbids libm `sin/exp/pow`. A byte-for-byte gate DETECTS drift; it does not EXCLUDE it. SL1 clause 5 demands a crate or module rule with an observed-failing control, "never by care". `Gf` is that fence, and the base already wrote it. See §6 |
| The drag term: the parent computes drag from the child's stated mass, cross-section and drag coefficient, semi-implicit | `signal_authority_and_relays.md:1242-1268` | M2a keeps it. Only the CAP half of §6.2 dies |
| The terrain-following autopilot block | `high_speed_flight_latency.md` D-7(b), §6.3(b) | M6: software pressing the stick is lawful; a limiter is not |
| The pop detector as an instrument | Addendum 2 §C.6 | SL8 makes it the measuring tool for every seam kind |
| The ~20 square building shapes and the shape catalogue (§4) | `block_system_design.md` §4 | V2.3 keeps square building blocks |

---

## 2. Table A — rulings → changed text in the base

One row per ruling. The middle column quotes or cites what the base says. The right column says what
replaces it. **REFUSED** rows name the base claim the ruling forbids.

| Ruling | What the base says (where) | New state |
|---|---|---|
| **SL1 rewrite (2026-08-24 A1): a realm is TOLD where it is, one hop; never fold an absolute from the root** | §7.13 step 7: "the hull host ships the authored realm placement down the lane the client already receives (`RealmSnapshotDatagram`, **absolute positions since the A5 flip**)". §7.6.1: "Distance comes from the server-authored **absolute positions** the A5 flip already ships". Board O16 / door 51: "the render origin's absolute-position path (A5 THE FLIP) — committed, PROTO_MINOR 7, nine consumer files". `signal_authority_and_relays.md:1320-1325`: "`uniform_external: DVec3` supplied by the **grandparent**" | **CHANGES §7.13 and door 51; REFUSES §7.6.1's absolute distance and the grandparent field.** There is no absolute position anywhere a realm can reach (SL1 clause 4). A realm holds one stamped, read-only reading of its own placement in its parent's frame (SL1 clause 2). `pin_abs`/`anchor_epoch` are gone (`version.rs:67-79`). Light-lag (§7.6) needs a lawful distance source — see §4 item 3. **Example:** a moon knows its placement in its planet's frame and nothing else; it cannot compute its distance to a station in another star system, and neither can the planet |
| **SL2 clarification (2026-08-24 A2): contacts are realms, people are seen; the gateway composes** | §7.5 / O39(m): "any transmitter beyond the bubble is locatable to ±15,000 km with three posts" treated as a leak to fix. `signal_authority_and_relays.md:1365-1368`: a coarse voxel occupancy "hands your ship's shape to whoever hosts you" | **CHANGES the framing; the sibling-placement half is REFUSED-UNTIL-ASKED.** A person's pose never enters another realm (SL2), so the base's leak worry about PEOPLE is correctly framed. SL2's own words say a ship's systems track "placements the parent authors and may state". That is the owner's INTENT, not a built lane: no realm-inbound message carries a placement or a centre today, and a structural test pins the absence (`intershard.rs:416-440`; `intershard_closed.rs:485`). Stating a SIBLING's placement into a child is therefore new data across a realm boundary and needs an SL6 ask (§4 item 7, ask five). The hull-shape concern is a second SL6 ask, not an SL2 breach |
| **SL3 (a realm draws itself; a parent's per-child message carries a placement only)** + **reach R2 (the marker is deleted)** | Addendum 2 §C.1: "Above the top rung sits the realm's own proxy — which already exists"; board §1 row 7: "the realm proxy becomes the coarsest rung"; DEFERRED D-WINDOW-5 (`DEFERRED.md:6968-6970`): "tier 0 is the parent-authored marker" | **CHANGES.** The coarsest rung is the realm's OWN look statement (`crates/core/src/look.rs`), never a parent's drawing. The marker is deleted (`DEFERRED.md` D-REACH-1 step 5, BUILT 2026-09-04). D-WINDOW-5's own text is stale on tier 0. **Example:** a dormant moon is drawn by nobody; a moon inside somebody's reach boots and ships its own look one hop up |
| **★ NEW ROW — SL3 against SL10: what wakes a realm once the client can draw its shape?** | SL3 (`CLAUDE.md:177-181`): "A realm that is not running cannot be drawn — which is WHY visibility is the spin-up trigger". Reach R2 deletes the marker on exactly that reasoning (`owner_decisions_2026-09-02_reach.md:36-50`) | **A GENUINE CONTRADICTION the new law creates — see §4 item 12.** SL10 V1.1 lets a client draw a moon's hills with no shard for that moon running anywhere. The stated REASON for the spin-up trigger dies for the static half of the picture. **Recommended answer for the owner:** the trigger STAYS and its reason changes — a realm must run so that its DIFF exists (SL10 V1.6: edits, placements, live deposits). **Example:** a pilot warps toward a star system and the client draws the third planet's coastline from the seed before any planet shard boots. If the planet never wakes, the tunnels players dug are not there and the pilot flies over a world nobody edited |
| **SL4 (physics and re-home are separate, one-way)** | Board O29: "the PLANET owns the contact" cites "the standing law that the containing realm authors its children's positions". §3.6: a construction anchor mints a realm "with its own physics authority" | **Consistent.** No change. The crossing path may not name a motion symbol; §7.13's `contact_flags` hazard (`signal_authority_and_relays.md:1332-1338`) is the same rule stated in the base and stands |
| **SL5 (one world; no test-only variant)** + **G11 (raise the count gradually, never with a second world)** | Board O57: "Give the Cartesian profile a **TEST-ONLY generator emitting three boulders**". Addendum 1 §D.3–§D.4: "seed **plus configuration**", "a curated starter world", "an optional authored table … keyed by the body's position in the realm tree". Board O42/O43/O45: the starter world's radius, gravity band, biome set | **REFUSES O57 by name** (a test-only generator is a second world). **CHANGES Addendum 1 §D.3/§D.4 and O42/O43/O45:** the 2026-08-15 item 5(c) rule ("world generation is ALGORITHMS FROM SEED ONLY — any hand-authored table on the generation path is REPORTED to the owner") and G1 ("a value an operator can set is a value that re-rolls the galaxy") stand later than the owner's 2026-08-03 words. G10: the home is FOUND by its property (an Earth-like planet), not placed; the start city is BUILT by hand (live state). See §4 item 2 — the owner said both things and must pick |
| **SL7 (an occupied realm is its occupants' proxy)** + **reach R5** | §3.10: "the same `BulkMsg::ChunkDelta` goes to every session whose AoI includes that chunk" via "the D-9/D-39.5 per-cell reshape" | **ANSWERED by the code.** D-9 is BUILT (`DEFERRED.md:2925`): a body ships to the observers inside its reach, per cube. The board's "D-9 is a precondition of block edits" is satisfied; the edit fan-out reuses it |
| **SL9 (unbounded children; never a per-child per-tick row)** | §7.6.1: light-lag "held at the routing node … the LCA"; O39(o): "a 1,000-child system node performs 1,000 lookups per emission". §7.3's `Neighbourhood { hops: 2 }`. §3.10's "one compact manifest per AoI cell" | **CHANGES the cost rule, and one fold SURVIVES.** Any fold that visits every child is a defect and must be MEASURED on a wide realm (SL9). The galaxy holds 279,380 direct children; a TICK with one looker costs 20 µs through an R*-tree (MEASURED 2026-09-05, release). But `rebuild_static_rows_for` still walks every child on each adoption or release of a non-ship child (`regions.rs:876`), and that cost was MEASURED at 60.0 ms on the galaxy before ships were exempted (`DEFERRED.md:5419-5427`). §3.6 mints a realm when a player places a block on a construction anchor, and a minted station is not a ship — so every such placement pays the fold on its parent. See §4 item 13. The signal router must use the index, never a walk. O39(o)'s "cut to hops = 1" is consistent |
| **★ NEW ROW — HR2 (generic transfer: any entity kind crosses via the `TransferableKind` registry)** | The base has no row for the new kinds V2.4 and V2.5 create | **BINDS the new kinds.** A sub-metre block, a no-volume attachment and a one-record tree are entity kinds. Each crosses a shard through the ONE registry, never through a payload of its own. **Example:** a player chisels a 25 cm panel onto a hull wall and flies that hull from a moon's realm into the star system; the panel crosses inside the hull's transfer, through the same machinery a crate of ore uses. **Owed:** a registry entry per new kind before any of them is saved |
| **★ NEW ROW — HR4 (features once, run anywhere: the identical fixture on ≥2 shard kinds, G-IDENTICAL)** | Board §1 and the base treat the planet grid and the ship grid as one design, but no base row names the gate. `VoxelGeometry::{Spherical, Cartesian}` plants the split as DATA (`crates/sim/src/capability.rs:37-41`) | **BINDS §4 item 5 (two surface extractors).** The extractor choice must be per-CELL, never per shard kind (HR3 forbids a `match` on shard kind in a feature). The gate: ONE fixture that digs a cell and reads the surface back must pass on a PLANET realm and inside a HULL realm. **Example:** the same fixture carves a step into a moon's hillside and into a hull's deck, and reads the same shape back from both |
| **★ NEW ROW — HR5 (Tier-A crates at 100 % region and branch coverage)** | O7: "vendor ~400 lines" of noise "inside the determinism guard" | **BINDS the generator crate.** SL10 V1.2 makes ONE crate that both hosts compile. Vendored noise inside a Tier-A crate reaches 100 % or carries a written exemption (`coverage-exemptions.toml`). The `Gf` newtype helps: every op is a one-line inline shim and is trivially covered (`block_system_design.md:3765`). **Example:** the crate that grows a moon's hill is covered like the crate that ships a hull's placement |
| **★ NEW ROW — HR6 (agent-operable end to end: `vdctl`, input injection, wgpu readback)** | The base has no row; SL10 V1.3 demands a gate that compares a SERVER build against a CLIENT build | **BINDS the no-drift gate's shape.** `vdctl` is the shipped way to drive a client. The gate must build chunks on a client build and on a server build, on x86-64 and on aarch64, and compare byte for byte, driven by an agent with no human hands. **Example:** an agent boots a client, walks a dot to a named chunk on a moon, dumps the derived cells, and diffs them against the moon shard's own dump |
| **Movement contract (2026-08-26 M2/M2a): acceleration + torque per tick in the child's own frame; mass, cross-section, drag on change; VELOCITY NEVER CROSSES UP; the down lane is the stamped placement** | §7.13 step 5–7: `ShipOutputs { thrust, torque, mass, com, inertia }` per tick as `Coupling(ShipThrustPort)`; `HullFeedback { accel, g_load, ambient_density }` back down on the same port. `signal_authority_and_relays.md:1204-1225`: up-flow "thrust, torque — BODY frame, `Fx`", `mass, com, inertia`, `HullShapeSummary` ≈248 B "shipped on change plus a 1 Hz heartbeat", `bounding_radius_m`; down-flow "pose and angular velocity, proper acceleration, local gravity, ambient density, pressure, temperature, wind, composition, illuminance, star direction, contact flags ≈144 B at 20 Hz"; `:1242-1268` drag: parent integrates | **CHANGES §7.13 and the handoff section; REFUSES two items by name.** (1) What crosses up per tick is an ACCELERATION and a torque, six numbers, in the child's frame (`intershard.rs:1388-1389`: "what crosses is an acceleration, never a force in newtons") — thrust in newtons is refused. (2) Mass and drag cross ON CHANGE (`ChildFacts`, `intershard.rs:538`); a 1 Hz heartbeat of what-I-am facts is refused ("never per tick"). (3) The down lane is the placement, stamped, one hop — `HullFeedback`'s ambient density is the REJECTED alternative M2a records ("the parent could state its medium DOWNWARD … refused"). (4) `com`, `inertia`, the six-face shape summary, buoyancy volume are NEW DATA: an SL6 ask each (see §4 item 7). (5) The base's "drag is the parent's arithmetic, semi-implicit" AGREES with M2a and survives. **Example:** a hull sends "4 m/s² along my nose"; the star system rotates it, adds the star's pull and the drag it computes from the hull's stated mass and drag coefficient, integrates, and ships the placement down |
| **Movement answers (2026-08-27 M-A/M-D) + A4 (a parent never sets a speed; the governor is deleted) + suit S3** | Board §1 "Measured": "the maximum flight speed's binding term … **240 m/s**"; O31: "before P4.10's **governor** is written · the target flight speed band"; O39(g): "derived envelope `q = ½ρv²` per hull, plus a 0..1 assist scalar … a flat cap is a magic number"; O39(h): "the hard limiter as a realm property on the physics-authority shard"; `high_speed_flight_latency.md:589-608` (§6.2 the envelope), `:613-622` ("clamps commanded g … clamps speed … refuses a commanded attitude"), D-5/D-6/D-7/D-10 (`:928-933`); O38: a 64 m collider bubble "at 800 m/s"; §3.7.3 rule 5 / §6.4: "the AoI warm ring … the 393 m/s AoI ceiling" | **REFUSES by name:** O31's governor and "target band"; O39(g)'s envelope AS A CAP; O39(h)'s hard limiter; D-6 and D-7(a)'s limiter half; §6.3(a)'s clamp-and-refuse. A player never states a speed, so nothing on the flight path can be clamped or refused (M-D.1–2). **THE CAP IS STILL IN THE CODE:** `dot.rs:457-474` scales the stick by `governed_ceiling_for_frame` and `ramp_cap_mps` every tick (`flight.rs:93,105,118`). S3 deletes it; S6 defers the deletion under D-MOVE-3 (`owner_decisions_2026-09-05_suit.md:69-71`). The terrain and block slices must size NOTHING on those symbols. **CHANGES:** dynamic pressure survives only as DRAG the parent computes (M2a), never as a limit; "slowing down is gameplay" — a ship's own safety block, switchable off (M-D.4); the terrain-following autopilot block (D-7(b), §6.3(b)) survives as software pressing the stick (M6). D-10's "minimum drawn terrain radius ≥ v_max × 4 s" has no v_max: it becomes a residency LEAD grown by closing speed (M-B), which SL10 makes local (the client generates ahead). O38's bubble must hold at any speed via the swept line (M-D.3). The 240 m/s figure is an upload-budget MEASUREMENT of the old vertex format, not a ceiling. `k_dwell = 5` (`geometry.rs:938`) is still sized on the governor — D-MOVE-3 |
| **Interim engine rating (M-C) and the suit (2026-09-05 S2–S6)** | §4.7: "at the **6 m/s reference walk speed**" as a derivation input; `high_speed_flight_latency.md` and O31 size everything to a ship speed band; no suit exists anywhere in the base | **CHANGES.** Every speed is a per-entity RATING stated as a fact (a hull's `EngineRating`, `crates/sim/src/stub/drive.rs:28`; a suit's rating later, S2). No constant tied to the server. The order is binding: foundation → blocks and terrain → the character → the suit (S6). The base's character controller (§4.7) is on-foot movement on a floor, which S2 leaves alone; movement in SPACE needs a suit and the base has no room for one — the character design must take a rating record from the first commit |
| **Seed and secrecy (2026-08-27 S1–S5) + galaxy shape G5 (seed says WHERE TO LOOK, never WHAT YOU FIND) + SL10 V1.6 (live-state deposits cross as a diff)** | §3.8 (`block_system_design.md:4333-4366`): "the generator is split into a public half and a concealed half — `gen_concealed(secret: &ServerSecret, key)`"; R11: "strategic materials are what a rock assays to … a server-held table"; `concealed_resources.md:65-81`: "keep the key AND the working on the server"; O8: "`concealed_key_id: [u8; 8]` on `RealmDescriptor` … master key in the deployment secret store"; O36(b): "the PRF that places concealed ore — HMAC-SHA256"; O44: the enrichment factor 2–4× | **REFUSES by name:** the secret-key placement (`gen_concealed`, `ServerSecret`, `concealed_key_id`, O36(b)). A deposit placed by `f(position, secret)` is a static map, and "a shared static map becomes public knowledge through ordinary play; the seed only gets there faster" (S2). Secrecy is not the cure; what pays must read world state that CHANGES (S5.2). **CHANGES R11:** "what a rock assays to when broken" survives only if the assay reads live state (depletion, what NPC life consumed, what players took — the dormant-world pillar). **CHANGES O9:** the survey is THE mechanic (S5.3) and survives, keyed to live state. **CHANGES O44:** a public indicator material is lawful ("where to look", G5); its 2–4× number was derived against a static map and is UNMEASURED against a live-state one. **OPEN:** whether COMMON ore (iron, copper, coal) is "what pays" — see §4 item 1 |
| **Galaxy shape (2026-08-27 G1–G14): every placement number seed-derived; no config knob; the count is a result; the home is found; the shape is faithful** | Addendum 1 §C.2: `R = 124·m/π` per-body radius ladder; §D.2 `BodyPhysical { radius_step, mass_kg }` "seed-derived by default; every field overridable"; O2 "which planet radii are legal"; O42 "what radius does the starter world get" | **CHANGES.** The ladder step is a generator CONSTRAINT and is lawful; the per-body radius and mass must be DRAWN from the body's own seed stream (G1), never set in a table (see SL5 row). "The starter world" as a curated body is superseded by G10 (found by property, after S12). O2's rule stays a generator constraint |
| **Visibility radius (2026-09-01 V1–V5) + reach (2026-09-02 R1–R10)** | §3.7.1 / Addendum 2 §C.2: tier selection reuses "the locked area-of-interest model: angular size above a threshold"; §3.7.6 "the demand loop already spins a realm up before the player reaches it"; O56: status lights "shipped at the observer's clamped rung"; the window lane's "tier 0 = marker" (D-WINDOW-5) | **CHANGES the wake input.** A realm states ONE reach from its LOOK (size AND brightness) to its parent on change (`ReachStated`, `intershard.rs:603-618`, BUILT); the parent tests its direct children by an index; visibility is a radius, not a tree walk. Tier selection on the client stays a local screen-space choice (SL10 lets the client hold the shape) and is not the wake rule. O56 is CHANGED: a lit status light is part of the realm's own LOOK and therefore of its reach; nothing is "clamped" by a parent (R8: a realm may state a smaller reach, a parent may never clamp it). The galaxy's star field is shipped ONCE and placed by the client at the sky anchor (`channels.rs:499`) — the ONE thing the client places from a catalogue |
| **SL8 seamless (eleven seam kinds: jump, black frame, flicker, detail-by-box, brightness pop, arrival pop, tick hitch, re-state rate, tier refusal, sprite cull, lane flood)** + **V2.9** | Addendum 2 §C.6 "No pop, ever"; §3.7.4 the crossfade direction rule; O20 "automatic exposure makes stars vanish on warp arrival"; O23 impostors; O59 AA/120 Hz; the board's "warm ring" | **Consistent with the BASE, and now NAMED. NOT complete against SL10 — a twelfth kind is owed.** Each base mechanism maps to a seam kind: the crossfade → detail-by-box and flicker; the pyramid → detail-by-box (a tunnel that vanishes at 200 m); exposure → brightness pop; impostor cull → sprite cull; the residency governor → tick hitch; a tier the server refuses → tier refusal. **SL10 creates a seam the eleven do not name: the DIFF LAG.** The client derives the static shape at once and the owning realm's diff arrives afterwards; the order is visible. The nearest kind, "arrival pop", is about a realm arriving, not about a surface correcting itself under a standing player. **Proposed twelfth kind for the owner — the diff lag,** with a PHYSICAL tolerance: the diff for every cell inside the character's own reach is applied BEFORE the derived shape for that cell is first drawn; the gate measures the frame count between the two and it must be zero. **Example:** a player lands on a moon and walks toward a hill the client drew from the seed. Three frames later the moon's shard sends the diff for a tunnel another player dug last week, and the ground opens under the player's feet. Tolerances are PHYSICAL and must be MEASURED per seam kind; the base's "pop detector" is the instrument |
| **SL10 — the seed-shaped world (2026-09-07 V1.1–V1.8)** | Commitment 2 (`block_system_design.md:366-369`): "terrain never crosses the network; only the seed and the cells players changed travel, and everything else is recomputed identically on every machine"; commitment 5 (`:389-396`): "whole-number arithmetic … behind a type that cannot reach the maths functions that differ between machines"; commitment 6 (`:398-405`); §3.7.3 rule 2 "regenerated on demand — legal only because generation is Category-A"; §3.10 "a client that already has the generator only ever receives deltas"; O7 "vendor ~400 lines … inside the determinism guard"; O3's option (A) "angle-proportional" (a `tan()` warp) against the JCGT polynomial "using only + − ×"; §4.7 "the one transcendental, `atan`, evaluated once per realm at activation" | **ANSWERS the base's biggest premise, which the 2026-08-24 Q4 and S9 Q2 rulings had REFUSED in between** ("the client may not derive … that door does not close again"). SL10 supersedes those for the STATIC SHAPE ONLY: `f(seed, address)`, never of time or state. **CHANGES commitment 5:** floating add, subtract, multiply, divide and square root are allowed (V1.4); integer hashing for draws; no libm `sin/exp/pow/tan`; no FMA CONTRACTION; no fast-math. So O3's `tan()` option is refused unless the crate carries its own deterministic `tan`; the JCGT polynomial is lawful. Commitment 5's "behind a type that cannot reach the maths functions" is the `Gf` newtype and it SURVIVES — see §6. **CHANGES O7:** whatever the noise source, it must pass the byte-for-byte gate on x86-64 AND aarch64, server build against client build (V1.3); the library adoption stays the owner's. **CHANGES commitment 2's carrier:** everything the seed does not decide crosses as a one-hop DIFF from the owning realm (V1.6) — edits, placed blocks, sub-metre blocks, attachments, growth stages, damage, live deposits. The star field is NOT derived by the client (V1.8): it stays shipped once. `atan` at activation (§4.7) is solver configuration, not shape, and is lawful. **★ SL10 OPENS A DOOR — see the door table.** **Example:** a client draws a hill from the moon's seed while the hull descends; the moon's shard evaluates the same crate for the chunk under the player's boots; a tunnel dug last week arrives as a diff and BOTH hosts compose it over the derived hill before either draws or collides |
| **V2.1 smooth, realistic terrain built of voxels; mine like Minecraft; a placed terrain voxel reshapes the surface** | "What we are not building" (`block_system_design.md:454-460`): "**No smooth or melted terrain.** No marching cubes, no surface nets, no dual contouring. The world is blocky, softened only at exposed edges"; door 13: "explicit named shapes instead of a smooth density field — before P4"; board row 7: "seams closed by skirts, which work **precisely because the geometry is blocky**"; §5.6 the un-square edge; O21 the crumbling-edge amplitude; O39(d) soft blending "on a surface that visibly steps in one-metre cubes" | **REFUSES by name:** "No smooth or melted terrain" and door 13's "instead of a smooth density field", for TERRAIN. **CHANGES:** the terrain mesher, the skirt argument, the crumble (O21 dissolves for terrain; it existed to soften cubes), O39(d) (the blend premise), the pyramid entry (O4's octant mask summarises blocky sub-boxes; a smooth surface needs a height or density summary — UNMEASURED cost), and the collision shape (SL10 V1.5: the server collides on the SAME surface, which is the derived shape COMPOSED WITH THE DIFF; O28's voxel-cube collider premise changes). V2.3 keeps SQUARE building blocks with the agreed ~20 shapes, so the base's shape catalogue (§4) survives for constructions. **Open:** one mesher or two — see §4 item 5 |
| **V2.2 trees and large vegetation are ONE object on the surface, rendered by the client from seed and height parameters in the block; the server knows the shape for collision** | R4 (OPEN): "placing several tree blocks together yields a real tree … all of it must collide"; R13/R14: bushes and canopy are real CELLS with empty colliders; O25: "≈998 cells" per tree; O26: "composites are RECOGNISED, never overlaid … the server has no meshes"; O22: trees "bought greybox"; O23 impostors for a "12 m tree vanishing at the cull radius"; `stunning_look_plan.md` §3 the composite subsystem | **CHANGES R4, R13, R14, O22, O24, O25, O26.** A tree is ONE record whose geometry is `f(params in the record)` evaluated by the one generator crate on both hosts (SL10 V1.1: "the geometry of seed-placed features"); the server evaluates the same crate for collision (V1.5), so "the server has no meshes" (O26) is answered. The multi-block pattern grammar, the 998-cell count, the canopy-as-cells rule and the bought-greybox tree are all premises V2.2 removes. Trunk collides / canopy passes (R14) is a shape question inside the one object and is the owner's. **A felled tree needs a NEGATIVE diff entry — see §7 case 4.** Trees are designed LATER (V2.2); the foundation keeps room for "other large landscape parts" |
| **V2.4 blocks smaller than one metre that fit inside the 1 m cell** | Door 12: "one geometry per cell — no sub-cell subdivision — a finer chisel layer must be strictly additive; making the base cell subdividable is the door a comparable title's sequel walked through"; `:461-462`: "No sub-block chisel grid in the first release"; the collision rule (board §1): "Below the catalogue's smallest extent (Panel 0.125 m, Nub 0.5 m) there is no cell and therefore no collider, ever — the base cell is never subdividable"; O39(c): the collision threshold is representability at 0.125 m | **CHANGES door 12, the collision rule and O39(c).** The owner asks for sub-metre blocks that "should fit into 1 m space of the bigger blocks to simplify collision" (V2.4, "if possible"). The base's own escape — a finer layer strictly ON TOP of the 1 m cell — is the shape that fits the words. The saved record (O5, R1) must hold a sub-cell position; the 14 reserved bits are UNMEASURED against that need. **V2.1 and V2.4 collide on a smooth slope — see §4 item 11.** See §4 item 4 |
| **V2.5 sub-blocks (attachments) that take no space: HUDs, joints, rails, pistons** | §6.9 the block-cover (a HUD panel on a block face; Addendum 1 §B.3 "the cover table: face + a reference"); §7.16 mechanical blocks "kinematic by default"; door 37 "articulated machines transfer as one entity or as sub-assemblies"; O55 the panel reduction rung | **Consistent for HUDs; CHANGES joints and rails; and a JOINT REACHES THE `FrameSpace` SEAM.** The cover table already is a no-volume attachment on a face. A rotation joint "between two blocks" that turns what is built on it is a no-volume attachment too; the base's mechanical blocks are CELLS. The record needs an attachment list per cell (the cover table generalised). **A turning sub-assembly is a MOVING FRAME inside one realm:** the cells built on it no longer sit at fixed addresses, which breaks the pyramid's address AND breaks SL10's `f(seed, address)` for anything riding it. That is what `FrameSpace` is for, and D-38 rules the seam to be terrain's FIRST slice (owner 2026-09-05). Door 37 sits beside it and stays open. See §7 case 3 |
| **V2.6 one block type with parameters (durability, mass, style); style is client-side, not stored** | Commitment 3 / §2.1: "a block's identity is one small number naming a combination somebody approved, and every quantity is looked up from it rather than typed"; R1: "identity is ONE combined number, never independent substance/shape fields"; Addendum 1 §B.3: changing state in side tables | **CHANGES lightly; one question is open.** If "params" means a substance-table ROW (a hull-steel row with its mass and durability), the base's model already holds it. If it means PER-INSTANCE parameters on a placed block, the record needs room (R1's reserved bits) — see §4 item 4. Client-side STYLE from the connected mesh is lawful under SL10 (derived, never stored) and matches R12's "deliberately different per client" precedent |
| **V2.7 themes beyond space (registry room)** | Door 8: "numbering dense, append-only, never reused"; §2.1 caps: 512 substances, 256 shapes, 65,536 block types | **Consistent.** The caps must be re-checked against themes (ESTIMATED: several hundred substance rows per theme). Open, cheap |
| **V2.8 an Unreal client is investigated elsewhere; the foundation is renderer-agnostic; nothing may assume Bevy** | Board §1 "Verified engine and hardware facts" (wgpu 27.0.1 sparse residency, `MAX_CASCADES_PER_LIGHT = 4`, `VISIBILITY_RANGE_DITHER`, clustered decals off on macOS); O33 (shadow cascades), O59 (engine version, AA, 120 Hz), O27 (the packed vertex format "through the standard engine pipeline"), O19/O20 (materials, display transform), §6.7 "Rendering it in Bevy 0.18", the CDLOD crossfade "costing one pipeline bit" | **CHANGES the placement of every renderer fact.** These facts bind the Bevy renderer BEHIND the render seam; they may not shape the foundation's formats or the client library (which stays engine-free). O27's vertex bytes are a renderer decision; the seam defines the mesh OUTPUT. O59's engine-version row is also bound by M-E (no Bevy 0.18→0.19 until a ship realm is flown in a window — flown 2026-09-02 per R9; whether S11/S12 are done is UNMEASURED here). **SL10 V1.2 pulls the generator crate ACROSS the seam:** an Unreal client links it through a C interface, so the crate may hold no engine type at all |
| **2026-08-24 Q5/Q8 (galaxies; the tunnel; a player who creates a realm makes the cluster start processes — cap where realms are STARTED)** | R18: "the dock … spins up a realm belonging to you"; §3.6: "placing a block on a construction anchor mints a new realm … per-account concurrent anchors are capped"; O37: "the payment is the anti-abuse brake on realm spin-up" | **Consistent on the cap; CHANGED on the cost.** The cap belongs where realms are started (the orchestrator), not only where they are built. `construction_and_spawn.md` §4.3 already notes the payment brake collides with the economy-overlay law; a material or count cap, not a transaction, is the lawful brake. **A mint is an ADOPTION and pays the O(children) fold** (`regions.rs:752`) — see §4 item 13 |
| **S9 Q5 (every generator takes the instant; a realm was alive while nobody was there)** + **SL10 V1.1 (the static shape has NO time argument)** | O11: "(b) seasonal via an integer season index; (c) continuous time — refused"; door 26: "whether unobserved worlds keep evolving — before P6b"; O34 world simulation's phase; R7/R8 regrowth by a binomial draw over elapsed sleep | **ANSWERS door 26** (yes, they evolve; the pattern is `f(seed, universe_tick)`, `crates/physics/src/celestial.rs:3`). **CHANGES O11:** the client-derived static shape takes no time argument; anything that changes with time (snow cover, growth, a season's look) is NOT static shape and crosses as a diff or is a live-state field computed on the owning realm by `f(seed, at)`. O11's "(c) continuous time refused" contradicts Q5 only if it is read as refusing the instant on the SERVER side; on the client side SL10 refuses it. R7/R8's memoryless draw is exactly Q5's shape and survives |
| **2026-08-15 item 7 (the approval-citation gate on every wire arm from minor 9)** + **SL6** | O15 / door 38: "the closed set passes thirty entries and the structural review … stops being reviewable"; "there is no margin left at all" | **CHANGES.** The set has 44 arms (MEASURED). The review mechanism is not a count: every arm from minor 9 carries an owner citation enforced by a build test (`crates/wire/src/version.rs:800`). O15's discriminator-inside-an-arm advice is still good hygiene; its ceiling arithmetic is dead |
| **2026-08-16 window lane Q2 (the parent relays its children's self-authored statements VERBATIM)** + **D-WINDOW-5** | §3.10: "`tier_floor: u8` on the client's AoI interest report … the server CLAMPS it … EGRESS SELECTION ONLY"; O13's three-ladder rider | **CHANGES O13's second field.** The owner-approved lane carries the tier as a plain number on the WINDOW subscription ("serve tier 2"), selected at the gateway, which knows every observer's angular size (D-WINDOW-5, `DEFERRED.md:6971-6974`). The base's `tier_floor` on the interest report is a second road to the same place; take the window lane's. The `tier: u8` on the block-edit request and delta payload (door 45) is untouched |

### 2a. The base's own rulings table (R1–R20), re-validated

| # | Base ruling | Verdict under the rulings since 2026-08-24 |
|---|---|---|
| R1 | 8-byte record, 14 reserved bits rejected on decode | **CHANGES:** V2.4/V2.5/V2.6/V2.7 and V3.2 (the owner freezes "the saved block record" himself). Re-derive the width against sub-cell position, attachments, params and themes before anything is frozen |
| R2, R3, R5 | light level + colour as signals; colour where seen, brightness where simulated | OPEN, unchanged; V2.8 moves the renderer-light half behind the seam |
| R4 | composites (OPEN) | **Superseded by V2.2** (a tree is one object). Do not design the pattern grammar |
| R6 | three-state provenance; base terrain never collapses | OPEN, unchanged. V2.1's "a placed terrain voxel reshapes the surface" is `Placed` provenance on a smooth surface — compatible |
| R7, R8 | regrowth both ways; growth stage stored, sparse | **Consistent with S9 Q5 and SL10 V1.6** (growth stages cross as a diff) |
| R9, R10 | terrain-instability flag reserved; axial-free support flood | OPEN, unchanged |
| R11 | the strategic tier is what a rock assays to; server-held table | **Half REFUSED** (a keyed placement is a static map — seed ruling S2/S5); the assay-on-break idea survives only tied to live state |
| R12 | grass and bushes pure decoration, client-derived | **Strengthened by SL10** (the client derives static decoration from the one crate; the server samples the same crate) |
| R13, R14 | bushes and leaves are cells with empty colliders | **CHANGED by V2.2** for trees and large vegetation; open for small plants |
| R15–R19 | signal authorisation, scoped keys, relays, blueprints at a dock | OPEN, P9; no ruling touches them. R18's realm spin-up must be capped at the orchestrator (2026-08-24 Q5) |
| R20 | armour-tier materials answer griefing; the starter spaceport unbreakable | OPEN; G10 makes the starter city a hand-BUILT place, which fits |

### 2b. The base's "measured" facts

| Fact | Verdict |
|---|---|
| Noise: 12.19 ns/eval, 0.885 ms/chunk (MEASURED, Apple M4 Pro, one core, scalar f64, no transcendental) | Still a measurement of THAT bench. **It must now become a CLIENT frame-budget requirement** (SL10 moves the work onto the client — see §7 case 5). SL10 V1.3 also owes the cross-target byte-for-byte gate (x86-64 and aarch64, server build against client build) under no-FMA-contraction and no-fast-math. Both UNMEASURED |
| Crossfade band f = 0.15 passes at 7,353 resident chunks; 0.20 fails (MEASURED by the residency model) | A renderer-side number (V2.8); re-measure per engine |
| The flight ceiling 240 m/s from the terrain-upload budget | A measurement of the OLD vertex format's upload cost; **REFUSED as a ceiling** (M-D). Restate as a residency-lead measurement |
| The galaxy: 233,220 rows + markers per keep-alive, 22 MB (MEASURED 2026-09-02) | Fixed: a window ships the children in range (R10 decision 3); the marker is deleted |
| "150,000 rows per player per tick" | **NOT base text and NOT stale.** MEASURED by `grep`: the phrase appears at `CLAUDE.md:158`, inside SL1's own reasoning for the 2026-08-24 reversal, and again in M7's reasoning (`owner_decisions_2026-08-26_movement.md:213`). It is a LIVE law's justification and it stands. Revision 1 called it "the base's historical fear"; that was wrong |

---

## 3. Table B — the open register (Bands A–E), row by row

Columns: row · the old question · the new state (ANSWERS / CHANGES / REFUSES / OPEN) · what the synthesis
must do.

### Band A — before the first world is generated or saved

| Row | Old question | New state |
|---|---|---|
| **O1** | What shape is a planet's grid, and how far up the ladder do blocks go (five tier bits)? | **CHANGES.** The grid family stays the owner's (V2.3 keeps square building blocks; V2.1 makes the TERRAIN surface smooth over the same cells). The ladder's top is no longer a parent proxy: it hands to the realm's OWN look (SL3, R2). SL10 makes every rung client-derivable. The key-width half stays a one-way door |
| **O2** | Which planet radii are legal (the divisibility rule)? | **OPEN as a generator constraint.** G1: the radius itself is drawn from the body's seed, never set. The starter-world clause is CHANGED by G10 (home found, not curated) |
| **O3** | Which curve wraps the cube onto the sphere? | **CHANGES.** SL10 V1.4 forbids libm `tan`; option (A) is lawful only with the crate's own deterministic curve; the JCGT polynomial (+ − × only) is lawful as written. The deadline sharpens to the first no-drift gate pin |
| **O4** | What is in one edit-pyramid entry; is the pyramid stored? | **CHANGES.** V3.2 names "the edit-pyramid entry" as one of the three formats the owner freezes. V2.1 changes what an entry summarises (a smooth surface, not an octant mask of cubes) — every number in O4 (504 entries, 3.9 KiB, 5.85 µs) is UNMEASURED for a smooth terrain. SL10 V1.6: entries cross as diffs from the owning realm. **A DELETION must be representable** (a felled tree, a removed seed-placed rock) — see §7 case 4. (a) stored-vs-recomputed stays the owner's |
| **O5** | The saved record's exact field order | **CHANGES.** V3.2 makes it the owner's frozen format; V2.4 (sub-cell position), V2.5 (attachments), V2.6 (params/variants), V2.7 (themes) all need room. The sub-cell SEAT has two candidate shapes and they are different formats (§4 item 11). The 14 reserved bits are not known to suffice. Re-derive, then freeze |
| **O6** | The seven storage-format freezes | **CHANGES item 3** (prune-on-equality must compare against the STATIC shape `f(seed, address)` only; growth and other timed facts are diffs — SL10 V1.1/V1.6, S9 Q5). Prune-on-equality is also what makes the generator's version WORLD IDENTITY — see the new door row. Items 1, 2, 4–7 OPEN |
| **O7** | Where does the terrain noise come from; does the sampler take an octave count? | **CHANGES.** SL10 V1.4 binds the arithmetic (integer hashing, no libm, no FMA contraction); V1.3 binds the gate (byte-for-byte on both targets, both hosts); §6 binds the FENCE (`Gf`); HR5 binds the coverage of anything vendored. `noise` is declared and unused (`Cargo.toml:78`; not in `Cargo.lock`). The library adoption stays the owner's (standing rule). The octave count as a parameter survives, but an added octave moves a built world — see the new door row |
| **O8** | Which materials are concealed, and how is placement conditioned? | **REFUSED by name** (seed ruling S2/S5; SL10 V1.6): no secret-keyed placement, no `concealed_key_id`, no master key. Valuable deposits are LIVE STATE of the owning realm. What stays open: which materials "pay" — §4 item 1 |
| **O9** | Does the survey instrument ship with concealment? | **CHANGES.** The survey is THE mechanic (S5.3) and ships with live-state deposits, not with concealment |
| **O10** | Does the generator emit a non-cube surface skin? | **ANSWERED by V2.1:** the terrain surface is smooth from the first world; a `Cube` default skin is the refused shape. The vehicle half (hover first) is CHANGED by the rating law: a vehicle states its rating as a hull does |
| **O11** | Is the climate baseline seasonal? | **CHANGES.** The static shape has no time argument (SL10 V1.1); a seasonal look is a live-state field or a diff (`f(seed, at)` on the owning realm, S9 Q5). The determinism DoD in `roadmap.json` is superseded by SL10's gate |

### Band B — before the wire arm exists

| Row | Old question | New state |
|---|---|---|
| **O12** | The scoped channel key in S0.4 | **OPEN.** No `ChannelKey` exists (MEASURED). The S6 order (blocks and terrain before the character) does not forbid planting a constant |
| **O13** | The requested-tier byte and the tier descriptor | **CHANGES.** Take the window lane's tier number (2026-08-16 Q2, D-WINDOW-5) instead of a `tier_floor` on the interest report; `coarsen_level` is now a tombstoned diagnostic (`intershard.rs:1185`). Door 45's request/payload tier stays |
| **O14** | The four saga step-id constants in one file | **OPEN.** Ids 7..18 taken; 19+ free (MEASURED `intershard.rs:65-115`) |
| **O15** | The signal message family's shape; the 30-arm ceiling | **CHANGES.** 44 arms today (MEASURED); the ceiling is passed; the review is the approval-citation gate (`version.rs:800`). The "ride inside an existing arm" hygiene stays |
| **O16** | The render origin becomes a rigid pose | **ANSWERED by the code.** The origin is the composed level's `origin, origin_epoch` (`version.rs:171-173`); `pin_abs`/`anchor_epoch` are removed; the placement carries `orient: DQuat` (`crates/core/src/pose.rs:919-924`). Door 51's premise is gone |
| **O17** | Widen the chunk-state wire entry | **OPEN** |
| **O18** | Does distance delay a signal; how fast is the relay plane? | **CHANGES.** The light-lag distance has no lawful source under SL1 (no absolutes; one hop) — §4 item 3. The bubble/horizon numbers are ESTIMATED and untouched |

### Band C — before art is authored

| Row | Old question | New state |
|---|---|---|
| **O19** | Texture tile, resolution, roughness bake | **OPEN**, renderer-side (V2.8) |
| **O20** | Display transform, exposure, look statement | **OPEN**; SL8 names the brightness-pop seam its manual-exposure rule prevents; renderer-side |
| **O21** | The crumbling-edge amplitude | **CHANGES / likely dissolves.** It softened cubes; V2.1's terrain is smooth. Constructed blocks stay "exactly zero" |
| **O22** | Where prop and composite art comes from | **CHANGES.** Trees are generated from parameters in the record (V2.2) — option (D) for trees. Rocks and small props OPEN |
| **O23** | Impostors for distant props | **OPEN**, renderer-side; SL8's sprite-cull seam binds the tolerance |
| **O24** | R14's deferred collider pass, two halves due at P4.2 | **CHANGES.** The canopy-as-cells premise is gone (V2.2); the face-hiding/AO halves apply to small plants only |
| **O25** | Pin the composite tree's cell count | **REFUSED as a question:** a tree is one object (V2.2). The economy yield figure must be re-derived from the object's parameters |
| **O26** | R4's composite subsystem | **Superseded by V2.2 + SL10 V1.5** (the server evaluates the same generator for the shape). The emit-exclusion hook at P4.2 is not needed |

### Band D — before the subsystem is built

| Row | Old question | New state |
|---|---|---|
| **O27** | The mesh vertex format and its deadline | **CHANGES.** A renderer decision behind the seam (V2.8); the foundation fixes the seam's mesh output. The deadline dispute is moot for the foundation |
| **O28** | The physics and collision spike | **CHANGES.** SL10 V1.5: the server collides on the SAME surface the client draws — and that surface is the derived shape **composed with the owning realm's diff** (V1.6: "the client applies the diff over the shape it derived"), never the generator's output alone. The composition happens BEFORE the collider is built, on both hosts. `rapier3d` is not in `Cargo.lock` (MEASURED) — the adoption is the owner's. The hull collider is an SL6 ask (§4 item 7). The scripted topple (no envelope) is consistent with the movement contract. **Measurement owed:** the client's drawn surface and the shard's collider surface agree on a chunk that CARRIES A DIFF, not only on a virgin chunk |
| **O29** | The run-anywhere gate for geometry code | **CHANGES.** The seam's timing is ruled: `FrameSpace` is terrain's FIRST slice (D-38, owner 2026-09-05). "The planet owns the contact" matches SL4 and the frame law. **The seam is also what V2.5's rotating joint needs** (§7 case 3). HR4 makes the gate concrete: one fixture, a planet realm and a hull realm. Split-gate stays the owner's |
| **O30** | The in-realm authorisation model | **OPEN** (P9) |
| **O31** | The target flight speed band and the terrain radius floor | **REFUSED** (M-D: no governor, no cap, no target band). Restate as a residency LEAD grown by closing speed (M-B) and measured. The refused cap is still IN THE CODE under D-MOVE-3 |
| **O32** | The residency governor's fields | **CHANGES.** SL10 makes generation cost CLIENT cost, so the tick-hitch seam is now a CLIENT seam as well as a server one; V2.8 puts the knob behind the render seam. A frame budget and a thread are owed — §7 case 5 |
| **O33** | Shadow distance of built structures | **OPEN**, renderer-side (V2.8) |
| **O34** | Where world simulation lives; which six subsystems | **CHANGES.** S9 Q5 makes temporality a day-one argument on every generator; the phase question stays the owner's |
| **O35** | A temperature per block | **OPEN** |
| **O36** | Cryptographic primitives | **(b) REFUSED** with concealment; (a), (c), (d) OPEN |
| **O37** | Blueprint paste policy | **OPEN**; the realm spin-up cap belongs at the orchestrator (2026-08-24 Q5), and a mint pays the O(children) fold (§4 item 13) |
| **O38** | Deck normal; edit-rate cap; collider merge; bubble radius | **CHANGES the bubble** (no speed cap; swept line, M-D.3). The deck normal is the "felt-acceleration" open question in D-45 — §4 item 7. The other two OPEN |
| **O39 a** | Write our own mesher | **CHANGES:** the mesher must produce a smooth terrain surface AND square building blocks (V2.1, V2.3); "binary greedy meshing" alone no longer fits terrain |
| **O39 b** | Which shapes are walkable | **OPEN** (V2.3 keeps the ~20 shapes) |
| **O39 c** | The collision threshold = representability at 0.125 m | **CHANGES** (V2.4: sub-metre blocks must collide) |
| **O39 d** | Soft blending vs hard edges | **CHANGES:** V2.1 asks for realistic blending; the "steps in one-metre cubes" counter-argument is gone for terrain |
| **O39 e** | Is radio interceptable | **OPEN** |
| **O39 f** | Cockpit free-look | **OPEN**; R1 already treats the camera as client-local ("parallax is the ordinary camera") |
| **O39 g** | Derived speed envelope | **REFUSED as a cap** (M-D); survives as the parent's drag term (M2a) |
| **O39 h** | Flight computer split | **Limiter half REFUSED** (M-D); the autopilot block survives as software pressing the stick (M6, A5) |
| **O39 i** | Energy-based collision destruction | **OPEN**; collisions have their mass but no solver (M2) |
| **O39 j** | Client replay of own craft | **REFUSED** already by the no-prediction law; the base agrees. SL10 does NOT reopen it: V1.7 says the client never derives a pose, a velocity or any state |
| **O39 k** | A translucent preview block | **OPEN — needs the owner's ruling**, as the base says |
| **O39 l** | Amend the no-prediction wording (view transform client-local) | **OPEN — the owner's** (a law wording). R1's camera sentence is the nearest ruling |
| **O39 m** | The relay plane's eight rows | **OPEN** (P9) |
| **O39 n** | Six ship-physics rows | **Mostly REFUSED/CHANGED by the movement contract:** the grandparent's 24 B field (SL6 default NO; and no lane exists — SL1's one-hop shape is why); a per-tick 128 B aero struct (facts cross on change only); lift per block survives (an acceleration the child sums before it speaks); quantise at the producer is BUILT (`ChildDrive` is `[i64; 3]`); buoyancy volume is an SL6 ask; the inertial body-centred frame is consistent with D-MOVE-4 |
| **O39 o** | Two signal specification defects | **OPEN** (P9); hops = 1 agrees with SL9 |

### Band E — cheap, answerable any time

| Row | Old question | New state |
|---|---|---|
| **O40** snow | OPEN; V2.1 (smooth) and V2.4 (sub-metre) change the substrate premise |
| **O41** surface marks | OPEN; a mark is live state → a diff (SL10 V1.6) |
| **O42, O43, O45** the starter world | **CHANGED by G10/G14** (home FOUND after S12; the city BUILT by hand) and by SL5/G1/item 5(c) (no curated table). See §4 item 2 |
| **O44** enrichment factor | **CHANGED by G5** ("where to look" is lawful); the number is UNMEASURED against live-state deposits |
| **O46** provenance residuals | OPEN |
| **O47** hiding in grass | OPEN; the server samples the same crate (SL10) — consistent |
| **O48** body bend, abundance radius | OPEN |
| **O49** who works out the grass | **ANSWERED by SL10** (client-derived from the one crate; the server samples the same crate) |
| **O50** healing, chemistry | OPEN |
| **O51** scripting | OPEN |
| **O52** voice | OPEN |
| **O53** atlas tool | OPEN, renderer-side |
| **O54** compression | OPEN |
| **O55** panel reduction rung | OPEN; V2.5's "widget from a predefined set" agrees |
| **O56** status-light range | **CHANGED by reach R3/R8:** a lit light is part of the realm's own look and reach; no parent clamp |
| **O57** a test-only generator for the Cartesian profile | **REFUSED by name** (SL5; G11) |
| **O58** the pyramid budget | OPEN; the numbers are UNMEASURED for smooth terrain |
| **O59** render taste rows | OPEN, renderer-side (V2.8); the engine-version row is bound by M-E |

### One-way doors — the ones a ruling moves, and the one a ruling OPENS

Every row carries three fields the base gave its doors and revision 1 dropped: the new state, the
DEADLINE (which freeze or which slice), and the RETROFIT COST (what it costs to change afterwards).

| Door | New state | Deadline | Retrofit cost if changed later |
|---|---|---|---|
| **NEW — the generator crate's arithmetic profile and version** | **OPENS with SL10.** Once a player builds on ground the client derived, the crate's version and the target's arithmetic profile become WORLD IDENTITY. One added octave or one rounding fix moves the ground under every placed block and under every saved diff, because a diff is stored against the derived shape (O6 item 3, prune-on-equality). There is no re-roll: G1 forbids a knob that moves a star anybody has seen | **Before the first world any player builds on is saved** | **The whole edit corpus.** Every diff is re-based against a new shape, or the old shape is kept forever as a second generator — which SL5 forbids. **Recommended answer:** freeze the arithmetic profile and adopt an append-only octave rule (a new octave may only add detail BELOW the existing surface's stated tolerance), and MEASURE the rule with a gate that generates the old and the new crate and reads the maximum surface displacement. **Example:** a player builds a landing pad on a hill on a moon; six months later the generator gains one octave, the hill is 40 cm higher, and the pad floats |
| 12 (no sub-cell subdivision) | CHANGED by V2.4 — re-design as a strictly additive finer layer | **The same sitting as the saved block record (V3.2)** | The record's field order. A sub-cell seat added after the freeze re-writes every saved record |
| 13 (named shapes, no density field) | REFUSED for terrain by V2.1; kept for building blocks by V2.3 | Before the mesher slice | The mesher and every stored chunk summary |
| 17 (mineral locations computable from the seed) | ANSWERED: common materials may be seed-decided (SL10 V1.1); what pays may not (S5, G5) | Before the deposit slice | The deposit records and the survey mechanic |
| 24 | shut, as the board says | — | — |
| 26 (unobserved worlds evolve) | ANSWERED yes (S9 Q5) | — | — |
| 37 (articulated machines transfer as one entity or as sub-assemblies) | OPEN, and now linked to V2.5's joint and to the `FrameSpace` seam (§7 case 3) | Before the attachment slice | The transfer registry entry and the pyramid's address |
| 38 (30-arm ceiling) | passed at 44 arms (MEASURED); the review is the citation gate (`version.rs:800`) | — | — |
| 42 (max flight speed) | **REFUSED as a door — no cap may bind a stick (M-D, S3). The cap is STILL IN THE CODE as a placeholder** (`dot.rs:457-474`, `flight.rs:93,105,118`) and its deletion is the owner-deferred D-MOVE-3 | The deletion lands with the suit (S6), AFTER blocks and terrain | **Low if nothing sizes on it; high if something does.** The terrain and block slices must anchor no number on `tau_s`, `approach_ceiling_mps` or `ramp_cap_mps`. `k_dwell = 5` (`geometry.rs:938`) already does |
| 51 (render origin pose) | its premise (`pin_abs`) is gone; the origin is composed | — | — |
| 52 (surface-skin field) | dissolved by V2.1 | — | — |
| 56 (legal radius set) | stays a generator constraint | Before the first galaxy is saved | Every body's radius, so every placement — the same cost as the new door above |

---

## 4. The genuine contradictions that remain open for the owner

1. **Is common ore "what pays"?** The seed ruling S5.1 says a block's substance may not be a pure function
   of (position, seed). SL10 V1.1 (newer) allows "the seed-decided common materials". R11 and O8 keep iron,
   copper and coal seed-derived and visible. If iron pays, S5.2 says it must read live state. **Recommended
   answer:** common building stock is seed-decided static shape; anything with an economy price reads live
   depletion. **Example:** a cliff shows iron-bearing rock from the seed; how much iron the rock still
   yields is the moon's live state and crosses as a diff.
2. **A curated starter world.** The owner said "seed + configuration for size + biome types" on 2026-08-03
   (Addendum 1 §D.3). On 2026-08-15 (item 5c) and 2026-08-27 (G1, G10, G11) the owner ruled no hand-authored
   table on the generation path, no config knob, the home FOUND by property. Newest wins, but the base
   carries the older words as a ruling. **Recommended answer:** no override table; the start is a found
   Earth-like planet, and the city there is built by hand as live state.
3. **Light-lag needs a distance between two realms.** §7.6.1 takes it from absolute positions, which SL1
   forbids. **M7 approved ONE scalar for VISIBILITY ONLY** — the distance from the nearest outside LOOKER
   to this realm, with no direction (`owner_decisions_2026-08-26_movement.md:200,245`; BUILT at
   `crates/wire/src/intershard.rs:1319-1342`). It is not a distance between two NAMED realms and it cannot
   carry light-lag. A routing node that is the LCA holds its direct children's placements only (one hop),
   so a distance between two GRANDCHILDREN cannot be computed by anyone without new data. **Recommended
   answer:** in-system light-lag only between direct children of one realm (lawful in that realm); deeper
   endpoints ride the relay plane with its own constant per hop; or ask under SL6 for a per-hop distance
   fold. Do not build §7.6 as written. **Example:** a radio call goes from a station in one star system to
   a hull in another. M7 tells the hull's realm how far the nearest looker is. It tells nobody how far the
   station is.
4. **Sub-metre blocks and the base cell.** V2.4 asks for blocks smaller than 1 m inside the 1 m cell. Door 12
   and the collision rule say the base cell is never subdividable. **Recommended answer:** the base's own
   escape — a finer layer strictly additive on top of the cell, with its own collider, addressed by (cell,
   sub-position); the 1 m cell stays the unit of matter and of the pyramid. Item 11 decides the address's
   shape.
5. **Smooth terrain and square blocks in ONE mesher (HR3, commitment 1).** V2.1 refuses "no smooth terrain";
   V2.3 keeps square building blocks. The base's whole appearance section assumes one blocky mesher with
   skirts. **Recommended answer:** one block field, one ladder, two surface EXTRACTORS selected by a
   per-cell shape class (terrain cell → smooth surface; construction cell → catalogue shape), never by
   realm kind; the collider comes from the same extractor applied to the derived shape **composed with the
   diff** (SL10 V1.5 with V1.6). HR4 binds the gate: one fixture must pass on a planet realm and inside a
   hull realm. Measure the seam where a placed block meets smooth ground before freezing the pyramid entry.
6. **A tree as one object: cell or attachment?** V2.2 makes a tree one record with parameters. Is it a CELL
   (occupies a 1 m cell, so nothing else can be placed there) or a no-volume attachment (V2.5)? The trunk
   collides, so a cell fits better. Trees are designed later; the record must reserve the parameter room now,
   and the diff format must be able to DELETE a seed-placed tree (§7 case 4).
7. **FIVE SL6 asks the base assumes silently** (default NO until the owner rules):
   - **ask 1 — felt acceleration DOWN to a child.** A walker inside a hull under 3 g needs "down"; D-45
     names it as the open question.
   - **ask 2 — the hull's collision SHAPE UP to its parent.** The planet owns the contact; today the parent
     holds mass, drag, reach and extent only.
   - **ask 3 — buoyancy volume and centre of buoyancy UP on change.**
   - **ask 4 — `com` and `inertia` UP on change** (§7.13 step 5).
   - **★ ask 5 — ONE SIBLING'S PLACEMENT DOWN into a child, so a ship can hold a contact list.**
     **The data:** one sibling realm's placement in the shared parent's frame, on change.
     **From which realm to which:** the star system down to the hull.
     **Why the receiver cannot compute it:** a child holds no placement but the stamped reading of its own
     (SL1 clause 2), and folding an absolute is refused (SL1 clause 4).
     **What doing without costs:** a ship's contact list is empty, so targeting, docking approach and
     collision warning have no source at all.
     **What it breaks if granted:** `no_realm_inbound_payload_carries_a_placement_or_a_centre`
     (`crates/wire/tests/intershard_closed.rs:485`) and the `ChildSceneSet` tombstone's stated protection
     (`crates/wire/src/intershard.rs:416-440`). SL2's own words ("placements the parent authors and may
     state") are the owner's INTENT; the lane does not exist and a passing test forbids it, so the ask is
     REFUSED-UNTIL-ASKED, not lawful.
     **Example:** a hull's targeting computer wants the station in the next orbit. The station is the hull's
     sibling under the same star system. Today the star system authors both placements and states neither
     one sideways.

   The grandparent's 24 B uniform-gravity field (`signal_authority_and_relays.md:1320`) is REFUSED as an
   SL6 ask nobody has made — the ground is SL6 (new data across a boundary, default NO) and SL2, not SL1
   clause 4, because a uniform acceleration is not a placement. Naming the right law matters: the owner can
   lawfully approve an SL6 ask and cannot lawfully approve an SL1 breach. Accept the SOI-crossing energy
   error as a game rule and MEASURE it, as the base itself offers
   (`signal_authority_and_relays.md:1327-1328`).
8. **The translucent preview block (O39 k) and the no-prediction wording (O39 l).** Both are the owner's;
   no ruling since 2026-08-24 touches them.
9. **Per-instance block parameters (V2.6).** A table row or a per-block record? The record width (item 4)
   depends on it.
10. **Which generator tag refuses a client?** SL10 V1.3 says one already exists. The handshake folds
    `coordinate_generation` only (`crates/wire/src/version.rs:350`). If the owner means that fold, it must
    grow the generator crate's version and the target's arithmetic profile; if another tag is meant, I could
    not find it. UNMEASURED.
11. **★ NEW — a sub-metre block on a smooth slope.** V2.1 makes the terrain surface smooth. V2.4 asks for
    sub-metre blocks that seat inside the 1 m cell. On a smooth slope the surface does not follow a cell
    face, so a block placed on a sub-cell LATTICE either floats above the ground or sinks into it. The two
    candidate answers are two different saved formats:
    - (a) a LATTICE OFFSET inside the cell (three small integers) — cheap, and it floats on a slope;
    - (b) a SURFACE-RELATIVE SEAT (a position on the derived surface plus an orientation) — it sits flush,
      and it re-binds if the generator's version ever moves the surface (the new door row).
    **Recommended answer:** (a) for the address, plus a client-and-server-shared SEATING rule that drops the
    block onto the derived-plus-diff surface inside its cell, so the seat is DERIVED and never stored.
    **Example:** a player sets a 25 cm lamp post on a moon's hillside; the lamp's cell is a lattice address,
    and both the client and the moon's shard drop it onto the same slope by the same rule.
12. **★ NEW — what wakes a realm, once the client can draw its shape?** SL3 says a realm that is not running
    cannot be drawn, and that is WHY visibility spins it up. SL10 V1.1 lets the client draw the static shape
    with no shard running. **Recommended answer:** the trigger stays and its stated reason changes to "so
    its DIFF exists" — a realm must run to state the edits, the placements and the live deposits the seed
    does not decide. **The seam to measure:** a client that draws seed shape with no diff yet applied (the
    proposed twelfth seam kind, the diff lag).
13. **★ NEW — a mint pays an O(children) fold on its parent.** §3.6 mints a realm when a player places a
    block on a construction anchor. A mint is an ADOPTION, and `rebuild_static_rows_for` walks every child
    of the parent (`crates/sim/src/stub/regions.rs:876`, called at `:752`). Ships are exempt; a station is
    not (`DEFERRED.md:5425-5427`). **Recommended answer:** either make a minted realm a mover for the
    placement layer (the ship's cure, generalised), or replace the vector copy with a per-row edit, which
    the code's own cost note already proposes. **The measurement owed:** mint a realm on a wide parent and
    read the pace line, BEFORE the anchor design is promoted. **Example:** a player places the first block
    of a station on an anchor in a busy star system; the star system copies its whole static row vector to
    admit the new realm. On the galaxy that same event MEASURED 60.0 ms against a 20 ms budget.
14. **★ NEW — `mul_add` on the generation path.** SL10 V1.4 forbids "fused multiply-add contraction", which
    is the compiler fusing `a*b + c` without being asked. The base states the explicit `f64::mul_add` call
    is bit-exact and keeps it (`block_system_design.md:3748-3751`). These are different things and the two
    texts disagree. **Recommended answer:** keep `mul_add` OFF the `Gf` surface until the byte-for-byte gate
    MEASURES it on x86-64 and on aarch64, because `mul_add` lowers to a hardware instruction on one target
    and to a library call on the other. UNMEASURED today. **Example:** the crate that grows a moon's hill
    uses one `mul_add` in the warp; the moon's shard and the client run on different targets and the hill
    differs by one bit.

---

## 5. Claims the rulings REFUSE by name (the do-not-promote list)

- "No smooth or melted terrain" (`block_system_design.md:454`) — V2.1.
- "The base cell is never subdividable" / door 12 / "no cell below 0.125 m" — V2.4.
- `gen_concealed(secret: &ServerSecret, …)` (`:4338`), `concealed_key_id`, O36(b), "keep the key on the server" — seed ruling S2/S5, SL10 V1.6.
- "A test-only generator emitting three boulders" (O57) — SL5, G11.
- The governor, the target speed band, the envelope-as-cap, the hard limiter, D-6, D-7(a), D-10 (O31, O39 g/h, `high_speed_flight_latency.md:589-622, 928-933`) — M-D, A4. **The code still holds the cap; do not size on it, and do not claim it is gone.**
- "Absolute positions since the A5 flip" (§7.13 step 7, §7.6.1) — SL1 clause 4.
- `uniform_external` from the grandparent (`signal_authority_and_relays.md:1320`) — SL6 (default NO) and SL2.
- **"The server collides on the generator's shape."** The collision surface is the derived shape COMPOSED WITH THE DIFF, on both hosts, before either draws or collides — SL10 V1.5 with V1.6.
- **"A parent may state a sibling's placement to a child" as if it were built law** — REFUSED-UNTIL-ASKED (SL6 ask five; the wire test pins the absence).
- Thrust in newtons per tick; `HullFeedback` with ambient density down; a 1 Hz heartbeat of what-I-am facts; a per-tick aero struct (§7.13, `signal_authority_and_relays.md:1204-1225`) — M2, M2a.
- "The realm proxy is the parent's marker" / "tier 0 is the parent-authored marker" (Addendum 2 §C.1, D-WINDOW-5) — SL3, R2.
- "27 + 3 = 30, no margin" (O15, board §5) — MEASURED 44 arms.
- The composite pattern grammar, the 998-cell tree, the bought-greybox tree (R4, O25, O26, O22) — V2.2.
- "The client may not derive; that door does not close again" (2026-08-24 Q4, S9 Q2) — SL10, for the static shape only; the star field stays shipped once.
- **A prose rule as the determinism guard.** SL1 clause 5 demands a crate or module fence with an observed-failing control, never care — see §6.

---

## 6. The determinism fence — the base mechanism SL10 needs and the register nearly lost

SL10 V1.4 states the arithmetic rules and V1.3 states the gate. **A gate is a DETECTOR, not an exclusion.**
The gate compares the chunks it generates; it cannot exclude a target-dependent divergence at an address it
never generated. SL1 clause 5 already says how this project does exclusion: a crate or module rule with an
observed-failing control, "never by care". The conventions section uses that shape for I/O (a clippy
`disallowed-methods` list).

The base wrote the right fence and the synthesis must promote it, not bury it:

- **`Gf(f64)`** (`docs/investigation/block_system_design.md:3758-3765`) — a newtype for the generator's
  float that exposes `+ − × ÷`, `sqrt`, `floor`, `abs`, `min`, `max`, `from_i64`, `to_f64` and nothing else.
  A transcendental is unreachable on the generation path by CONSTRUCTION. The base names the precedent in
  this tree: `EffectFree` in `sim/src/coupling.rs`. Every op is an inline one-liner, so HR5's 100 % coverage
  is cheap.

Three arithmetic items the ruling does not name and the register must carry:

| Item | State | What to do |
|---|---|---|
| **The float remainder `%`** | `%` on `f64` lowers to the platform's `fmod`. V1.4 names `sin`, `exp` and `pow` and does not name it. ESTIMATED risk: real, because `fmod` is a library call on some targets | `Gf` excludes it by construction; a prose rule does not. Do not add `Rem` to `Gf` |
| **`powi`** | It expands to repeated multiplication and is safe. Nothing in revision 1 said so | State it: a reader who takes "no `pow`" literally strikes a lawful operation. `powi` may be a `Gf` method written as explicit multiplication |
| **`mul_add`** | The base and the ruling disagree — see §4 item 14. UNMEASURED | Keep it off `Gf` until the gate measures it on both targets |

**The gate must be OBSERVED FAILING.** Delete one octave from the crate, run the gate, watch it go red, put
the octave back. A gate nobody has seen fail is an argument, not a measurement (the standing rule).

**Example, in the game's words.** The moon's shard and the client both evaluate the one crate for the chunk
under the player's boots. One of them was built for a machine whose `fmod` rounds differently. The gate
generated ten thousand chunks and never reached that branch. The player's boots sink one centimetre into a
slope on some machines and not on others. The newtype makes that branch impossible to write.

---

## 7. Five cases the domain must answer, and the base never asks

1. **An edit during a crossing.** A player digs inside a hull while the hull hands over from the star system
   to the galaxy. Who owns the diff for those ticks? How does the diff's fence order against the transfer
   saga? The transfer machinery is the most proven part of this tree, and the voxel diff is a new payload
   riding beside it. **Owed:** a stated owner for the diff across the saga's steps, and a fixture that digs
   during a hand-over. **Example:** a player carves a doorway in a hull's wall at the exact tick the hull
   leaves System 7; the doorway must exist once, in the galaxy's copy, with no lost tick and no duplicate.
2. **A sub-metre block on a smooth slope.** See §4 item 11. It decides a field in a format V3.2 freezes.
3. **A HUD on a rotating turret.** V2.5's joint "turns what is built on it". A turning sub-assembly is a
   MOVING FRAME inside one realm: the cells built on it no longer sit at fixed addresses, which breaks the
   pyramid's address and breaks SL10's `f(seed, address)` for anything riding it. That is what `FrameSpace`
   is for, and D-38 makes the seam terrain's FIRST slice. Door 37 (articulated machines transfer as one
   entity or as sub-assemblies) sits beside it. **Owed:** the attachment design must state whether a
   rotating sub-assembly is a frame, and the frame seam must land before the joint does. **Example:** a
   player mounts a HUD panel on a turret's face; the turret turns; the panel's cell address must still name
   the same panel.
4. **A tree on a mined edge.** A tree is one seed-placed record whose geometry the client derives (V2.2).
   A player fells it. The felling is live state, so the diff must carry a **deletion tombstone against
   derived shape**, and the client must not re-grow the tree when it re-derives the chunk. **Owed:** the
   diff format needs a NEGATIVE entry, and the diff is one of the three formats V3.2 freezes. **Example:**
   a player cuts down an oak on a moon; a week later the client re-derives that chunk from the seed and the
   oak must stay down.
5. **The client's frame budget, and the thread.** SL10 moves generation cost onto the CLIENT for the first
   time. The tick-hitch seam is now a CLIENT seam. The base's one relevant measurement — 0.885 ms per chunk
   on an Apple M4 Pro, one core, scalar f64 (MEASURED, that bench) — must become a REQUIREMENT with a stated
   budget and a stated thread. A hull landing at flight speed crosses chunk after chunk; a planet 10,000 km
   away in the window needs coarse chunks over its whole visible face. **Owed:** the frame budget in
   milliseconds, the thread the crate runs on, and a MEASURED stutter count on a landing. **Example:** a
   hull descends toward a moon at flight speed and the client must derive the hills ahead in time; if that
   work runs on the frame thread, the picture stutters, and a stutter is a seam.

---

## 8. Revision log

Each finding from the two verdicts, and what revision 2 did.

### From the law refuter (`verdicts/laws_law.md`)

| # | Finding | What I did |
|---|---|---|
| 1 | **WRONG** — door 42 says "no cap exists"; the cap runs on the live path | **FIXED.** Verified: `crates/sim/src/stub/dot.rs:457-474` scales the stick through `flight::throttle_axes_scale` after `governed_ceiling_for_frame` and `ramp_cap_mps` (`crates/core/src/flight.rs:93,105,118`). Rewrote door 42 with the code citation and the D-MOVE-3 deferral; added a fact row in §1; added the same warning to the movement-answers row in Table A, to O31 and to the §5 list |
| 2 | **BREAKS_LAW (SL6)** — a sibling's placement declared lawful with no ask | **FIXED.** Verified the `ChildSceneSet` tombstone (`crates/wire/src/intershard.rs:416-440`) and the test `no_realm_inbound_payload_carries_a_placement_or_a_centre` (`crates/wire/tests/intershard_closed.rs:485`). Rewrote the SL2 row as REFUSED-UNTIL-ASKED, added the verdict word to §0, added ask five to §4 item 7 in the full SL6 shape, added a fact row, and added the claim to the §5 do-not-promote list |
| 3 | **MISSING** — SL10 against SL3: what wakes a realm once a client can draw it? | **ADDED.** A new Table A row and §4 item 12, with the recommended answer (the trigger stays; the reason becomes "so its DIFF exists") and the seam to measure |
| 4 | **MISSING** — no row for HR2, HR4, HR5, HR6 | **ADDED.** Four new Table A rows, each naming the base text it binds and the gate it owes. HR4's gate is also written into §4 item 5 and into O29 |
| 5 | **MISSING** — SL10 creates a seam the eleven kinds do not name | **ADDED.** The SL8 row now proposes a twelfth kind, the DIFF LAG, with a physical tolerance and a gate |
| 6 | **WRONG, narrow** — M7 cited as a general parent→child distance | **FIXED.** Verified `owner_decisions_2026-08-26_movement.md:200,245` and the built field `crates/wire/src/intershard.rs:1319-1342`. §4 item 3 now says M7 is a VISIBILITY scalar only and cannot carry light-lag; added a fact row |
| 7 | **WRONG, cosmetic** — "two doc comments" | **FIXED.** MEASURED six sites in three files; the row now lists them and names `transient.rs:239` as the one that carries P5's analytic-gravity note |

### From the feasibility refuter (`verdicts/laws_feasibility.md`)

| # | Finding | What I did |
|---|---|---|
| 2.1 | **WRONG** — the collision surface is the derived shape PLUS the diff | **FIXED.** Verified SL10 V1.5 and V1.6 in `owner_decisions_2026-09-07_voxels.md:16-64`. Corrected O28, the V2.1 row, the SL10 row and §4 item 5; added the composition to the §5 list and stated the measurement owed (a chunk that CARRIES a diff) |
| 2.2 | **MISSING** — SL10 opens a one-way door the report does not register | **ADDED.** A new first row in the door table: the generator crate's arithmetic profile and version become world identity, deadline "before the first world any player builds on is saved", retrofit cost "the whole edit corpus", with the append-only octave rule as the recommended answer |
| 2.3 | **MISSING** — the door table has no deadline and no retrofit cost | **FIXED.** The door table now carries three fields per row: the new state, the deadline, and the retrofit cost. Door 12 is pinned to the same sitting as the saved block record (V3.2) |
| 2.4 | **MISSING** — no structural fence for the determinism rules; the surviving base fence goes unlisted | **ADDED.** New §6. Promotes `Gf(f64)` (`block_system_design.md:3758-3765`) as the fence SL1 clause 5 demands, adds the `%`/`powi`/`mul_add` rows, and states the gate must be OBSERVED failing. Added a new §1a table for base text that SURVIVES |
| 2.5 | **UNMEASURED_AS_FACT** — one O(children) fold survives | **ADDED.** Verified `rebuild_static_rows_for` at `crates/sim/src/stub/regions.rs:876`, called at `:752` and `:855`, and the measurement at `DEFERRED.md:5419-5431` (60.0 ms on the galaxy at a hull's adoption; ships exempted; a planet's adoption still rebuilds; slowest tick then 4.5 ms). Added a fact row, rewrote the SL9 row, added §4 item 13 with the measurement owed, and linked O37 and the Q5/Q8 row |
| 2.6 | **MISSING** — five cases the domain needs | **ADDED.** New §7: the edit during a crossing, the sub-metre block on a smooth slope (also §4 item 11), the HUD on a rotating turret (also the V2.5 row and O29), the felled tree's negative diff entry (also O4 and §4 item 6), and the client's frame budget and thread (also O32 and §2b) |
| 2.7 | **WRONG (small)** — "150,000 rows" belongs to a live law | **FIXED.** MEASURED: the phrase is at `CLAUDE.md:158` inside SL1's own reasoning, and again in M7's reasoning. The §2b row now says it is a LIVE law's justification and stands. The old clause is deleted from the galaxy-tick fact row |
| 2.8 | **WRONG (small)** — the grandparent's gravity field is refused by SL6, not SL1 clause 4 | **FIXED** in §4 item 7, in the §5 list and in O39(n). SL1's one-hop shape is why no lane exists; SL6 and SL2 are the refusal ground |
| 2.9 | **WRONG (small)** — the doc-comment count | **FIXED** with finding 7 above |

**No disputes.** I re-measured every finding against the tree and every one held. Revision 2 adopts all
sixteen.

**Example, to close.** A player lands a hull on a moon. Under the base, the moon's shard clamps the hull to
the moon's ceiling, the client draws blocky ground with skirts, a tree is 998 cells recognised by a pattern,
and the iron is where a secret key put it. Under the rulings, the hull keeps its speed and the parent adds
only drag; the client and the moon's shard evaluate one crate, compose the same tunnel diff over the same
derived hill, and agree byte for byte on what the boots stand on; the tree is one record the client grows
from its parameters and can be felled by a negative diff entry; and what the iron rock still yields is the
moon's live state, arriving as a diff. The cap the moon once used is still in the code, waiting for the
suit to delete it.
