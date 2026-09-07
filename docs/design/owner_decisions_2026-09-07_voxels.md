# Owner decisions — 2026-09-07 — THE VOXEL FOUNDATION: the seed-shaped world, and the owner's requirements

★ **LATER THAN EVERY DESIGN DOC AND EVERY EARLIER RULING, including 2026-09-05.** Where a plan
disagrees with this file, this file wins. Read it before you touch the voxel, terrain, block or
client-derivation design. The investigation base in `docs/investigation/` is NOT binding and was written
before the rulings of 2026-08-24 and later; this file lists the requirements the re-investigation must
serve.

The owner stated the requirements on 2026-09-07, agreed that a new law is needed for the seed-shaped
world, and said: *"let's re run the investigation on the voxels based on the new requirements I've
provided. Also we should not break laws, but I agree that we need to have a separate law for the
seed-based terrain."*

---

## V1. SL10 — THE SEED-SHAPED WORLD: ONE GENERATOR, TWO HOSTS, NO DRIFT

**Owner:** *"client should be able to render it based on the seed without drift from server, as server
will calculate collisions on what client will render."*

1. **The client MAY derive the world's STATIC SHAPE from the seed.** The static shape is what a fixed
   seed and an address decide and nothing else decides: the terrain surface, the seed-decided common
   materials, and the geometry of seed-placed features (a tree's trunk and canopy from its seed
   parameters). It is a function of `(seed, address)`. It is never a function of time, of a tick, or of
   any live state.
2. **ONE GENERATOR.** The generator is one Rust crate. The server compiles it. Every client compiles the
   SAME crate as a library and links it. A port to another language is FORBIDDEN, because a port is a
   second implementation and drift is a defect (the FINAL-backend law). A client on another engine links the crate
   through a C interface.
3. **NO DRIFT is a MEASUREMENT, never an argument.** A gate generates the same chunks on the server build
   and on the client build, on every target the game ships on (x86-64 and aarch64), and compares them
   byte for byte. The gate is red on one differing byte. The world-generation tag today refuses a PEER on
   the mesh handshake and a FILE on the store stamp; the CLIENT handshake carries no generator tag
   (MEASURED 2026-09-07: `crates/wire/src/version.rs:337-351` carries `coordinate_generation` only, and
   `crates/client` has no `world_generation` hit). The client refusal must be BUILT, and it names the
   generator crate's version and the target's arithmetic profile. (This clause first said the client
   refusal "already exists"; the re-investigation found that wrong, and the text is corrected here.)
4. **Determinism rules inside the generator.** Integer hashing for every random draw. Fixed evaluation
   order. No fast-math flag. No fused multiply-add contraction. No call into the platform's transcendental
   functions (`sin`, `exp`, `pow` from libm); where a curve is needed, the crate carries its own
   deterministic implementation or an integer table. Floating-point add, subtract, multiply, divide and
   square root are IEEE-exact on every target and are allowed.
5. **The server computes collision on the SAME shape.** The realm's shard evaluates the same crate for the
   same address and gets the same surface. Collision, the character controller and the sweep test read
   that surface. What the player sees is what the player stands on.
6. **Everything the seed does NOT decide crosses the wire as a DIFF from the owning realm**, one hop, from
   the realm that owns the cell: player edits, placed blocks, sub-metre blocks, attachments, growth
   stages, damage, and every deposit that live state decides (the 2026-08-27 seed ruling: a valuable
   substance is never a pure function of position and seed). The client applies the diff over the shape
   it derived.
7. **This is NOT a relaxation of "the server does all math; the client only renders".** The client still
   receives every placement from the server. The client never derives a pose, a velocity, or any state of
   any entity or realm. The exception is exactly the static shape of the seed, because that shape is not
   state. It is the same for every observer and for all time.
8. **Supersedes** the 2026-08-27 seed-and-secrecy ruling's S6 ("NO client-side derivation") for the
   world's static shape ONLY. The galaxy's star field stays as ruled on 2026-09-02: shipped once, placed
   by the client at the galaxy's composed placement.

**Example.** A player lands on a moon and walks to a hill. The client evaluated the moon's seed for the
hill's chunks while the hull was still descending, so the hill was drawn from far away, at a coarse
level, and grew detail without a pop. The moon's shard evaluates the same crate for the chunk the player
stands on and finds the same slope, so the character controller keeps the boots on the surface the
player sees. A tunnel another player dug last week arrives as a diff from the moon's shard, and the
client cuts it into the derived hill.

---

## V2. THE OWNER'S REQUIREMENTS FOR THE VOXEL FOUNDATION

Recorded from the owner's words on 2026-09-07. Not designed here. The re-investigation designs against
them; nothing in the foundation may shut a door on any of them.

- **V2.1 Smooth, realistic terrain, still built of voxels.** *"I'd like to try to implement the smooth
  realistic terrain, but still build with voxels. We should be able to mine, same as in minecraft, but
  the world around should look a bit more realistic. And when we place voxels, the terrain should change
  accordingly."* Mining removes what a cell holds. Placing a terrain voxel reshapes the surface around it.
- **V2.2 Trees and large vegetation are ONE object on the surface.** *"they should be placed as one
  object/block on the surface, but simply rendered on the client as a tree (with different height and
  structure depends on the seed and height params inside this block), but then the collisions are
  calculated on the server, so server somehow should know the shape."* The design keeps room for this and
  for other large landscape parts. Trees themselves are designed later.
- **V2.3 Building blocks stay square, with the agreed ~20 shapes**, so constructions and ships need not be
  boxes.
- **V2.4 Blocks smaller than one metre.** *"I'd like to have smaller blocks than 1m, as with them we
  should be able to build little details. But those smaller blocks should fit into 1m space of the bigger
  blocks to simplify collision and space taking (this is if possible)."*
- **V2.5 Sub-blocks (attachments) that take no space.** *"mostly joints and HUDs. … you would be able to
  put such sub-block to any squared voxel and it should not take space or prevent placing another voxel
  on top."* Examples the owner gave: a HUD on any block face that listens to a signal and shows it with a
  widget from a predefined set; a rotation joint between two blocks that turns what is built on it by a
  signal (a manipulator, a remote-controlled turret); rails, pistons.
- **V2.6 Block parameters and client-side style.** *"we might have different block types for different
  hull types, but it can be one block type with different params — durability, mass, and style. Style
  can be picked up by the client, so when same blocks of that type are connected and construct the mesh,
  it will render small additional details, that are not stored on the BE."* Reserve the room for
  parameters and variants from the first record.
- **V2.7 Themes beyond space.** Stylised blocks, plants, clothes and weapons for other settings on other
  planets. Not built now; the registry and the record must leave room.
- **V2.8 The client is replaceable.** The game continues on Bevy for now. The requirement stays: fully
  server-authoritative, an engine-free client library, and a render seam that any engine can consume, so
  a later client on another engine needs no change to the world. Nothing in the voxel design may assume
  Bevy. (Owner 2026-09-07: *"we will continue with Bevy for now, so just remove it from plans, but the
  requirements stay — fully server authoritative, so later on we can change clients."*)
- **V2.9 Seamless, always (SL8).** Landing, walking, the landing sequence, every realm in the window at
  any distance: no blank spot, no loading, no pop. A seam is a defect.

---

## V3. WHAT THE RE-INVESTIGATION MUST PRODUCE

1. A design for the voxel foundation that serves V2.1–V2.9 under SL10 and every earlier law.
2. The three formats the owner freezes together, each with its one-way doors named: the grid family, the
   saved block record, and the edit-pyramid entry.
3. The implementation sequence, slice by slice, each slice with its gate and its measurement.
4. A register of the decisions still open for the owner, with a recommended answer for each.

---

## V4. THE OWNER'S REVIEW OF THE PROPOSAL (2026-09-07, evening)

The owner read `docs/investigation/2026-09-07/00a_walkthrough.md` step by step and answered each step.
These answers are RULINGS. Where a step says "discuss", nothing is decided and no code may start on it.

| Step | Ruling |
|---|---|
| 1 The address | **Accepted.** |
| 2 Two forms | **Accepted, with a requirement:** *"we just need to be sure that it's super performant and looks beautiful and believable, especially considering LOD."* Performance and the look at every detail level are gated, not assumed. |
| 3 Small blocks | **Accepted.** |
| 4 Attachments | **Accepted.** |
| 5 The one generator | **Accepted.** |
| 6 The record | **Open.** *"Need to discuss when time comes."* The twelve-byte layout is a proposal, not a freeze. |
| 7 The pyramid | **Accepted, with two requirements:** *"We need to see from very far, and probably client will need to rebuild the LODs for the edits also."* The far view must hold at the largest distances a window reaches. The CLIENT folds edits into its own coarse rungs as well; the design must let the client rebuild coarse levels from the diffs it holds, by the same fold rule as the server. |
| 8 The diff lane | **Accepted, minus one item.** *"cross-shard placements probably should be quite rare, so I'd postpone it, but do all other proposed things here."* The cross-realm edit forward and its ack (R-1, R-2) are POSTPONED. Cross-realm edits are refused in v1 (V94). Everything else in step 8 is approved. |
| 9 Trees and decoration | **REFUSED as proposed; redesign owed.** *"I would not build it with primitives as trees most probably will not look good enough. I want to have a realistic graphics, meaning we will need to use assets. Same goes for grass, and some other decorative objects. Primitives should not be used for decoration. We should blend in the assets dynamically. But I like the modularity you've proposed, so maybe we can figure out something in between?"* Rulings: (a) no procedural primitive mesh is ever DRAWN for a tree, grass or a decorative object; (b) the look comes from ART ASSETS, blended in dynamically; (c) the modular record (one object on one cell, seed and stage parameters, a server-side skeleton for collision) is KEPT; (d) the middle path is owed as a design: the skeleton selects and places asset MODULES (trunk, branch, crown, tuft) from an asset kit per kind, so the collider is derived from the skeleton and the picture from the kit. Grass and decoration stay client-derived and never collide (R12), and they are drawn from assets. |
| 10 Render seam | **Accepted.** |
| 11 Physics | **Accepted.** |
| 12 The landing | **Accepted.** |
| 13 The formats | **One correction, the rest open.** *"Remove is not Air — Air block means atmosphere. When we remove in space — it's simply EmptyBlock."* The removal record's kind is **Empty**. `Air` is a SUBSTANCE (an atmosphere) and never the removal marker. Whether a removed cell on a planet fills with the realm's atmosphere is the realm's medium, not the record's kind. |
| 14 The sequence | **Accepted as an order, with a gate:** *"we need to discuss IN DETAIL each of the topics before we start any of them."* No slice starts before its own detailed discussion with the owner. |
| 15.1 Kind capacity | Owner asks: *"do we have enough if the amount of block types will significantly grow?"* Answer recorded in the reply of 2026-09-07: a 16-bit kind id holds 65 536 kinds and a 6-bit variant holds 64 styles per kind; 21 reserved bits remain. The owner has not yet said whether that is enough. |
| 15.2 Terrain in a built realm | **YES: a hull or a station may hold terrain cells.** And a new requirement: *"In the future I might want to add O'Neill Cylinder as one of the additional Realms. That mean gravity will point from center and it will require soil."* A realm's gravity is a FUNCTION the realm states (toward a centre, away from an axis, or a constant), never an assumption of the grid. The flat grid must serve a cylinder whose "down" points outward from its axis. |
| 15.2 (added) Gravity in a station or a hull | *"For the stations we can do a levitation, unless they wear magnet boots (if station don't have rotating parts). So on station people can walk on the ceiling or walls when needed. Actually the same for ships if easy to implement. We can have a graviblock or something similar to 'simulate' normal gravity inside the ship hull (or station hull). So again — the function."* **Confirmed: gravity is a FUNCTION the realm states.** A station or a hull without rotating parts has no gravity by default: an occupant floats, and magnet boots hold a character to any wall or ceiling. A GRAVIBLOCK is a future functional block that states a gravity field inside a hull or a station, so the function can come from a block. Nothing in the grid, the collider or the character controller may assume one "down". |
| 15.3 Common ore public? | *"Ideally this info is not public. But if that will simplify the architecture, I might accept it."* **Default: NOT public** — the seed ruling's S5.1 stands, and common ore is live state shipped as a diff. The door stays open only if the design shows a real simplification, stated to the owner. |
| 15.4 Block parameters | *"there can be additional parameters of the blocks, that will change its behaviours, like liquidity, max temperature, low temperature, flammable, etc."* These are per-KIND properties in the registry's substance row, never per-cell bits. The registry must have room for many behaviour parameters per substance, added over time without a record change. |
| 15.5 The store file name | *"Maybe name + address, something unique."* The realm store's file name must be unique across galaxies: the realm id PLUS its address in the tree (its parent chain), or another unique identity. The design proposes the exact form. |
| 15.6 Canopy fold, removal encoding, body id, seating | **Open.** *"Don't know, we should discuss when time comes."* |
| 15.7 The SL6 table | **Open.** *"I'd like to discuss separately in details."* No row is YES yet. Slice 4 may plant nothing until that discussion. |
| 16 Open risks | *"We can address them later one by one."* |
| The other engine | **Removed from every plan.** See V2.8 as rewritten. |

**What the owner has NOT decided yet, so no code may start:** the record layout (6, 15.6), the pyramid's
reserved bits (15.6), every SL6 row (15.7), the tree and decoration asset design (9), and each slice's
detailed discussion (14).

---

## V5. TOPIC 1 REVIEW — THE ADDRESS AND THE GRID (2026-09-07, night)

The owner read `docs/investigation/2026-09-07/topic_01_address_and_grid.md` and answered.

| Item | Ruling |
|---|---|
| The grid family is PER-BODY DATA, never per realm kind | *"I would argue about asteroid — if big enough, it can be similar to the planet in the future — own realm, move way faster. Players might land on it and build bases if they want. Or mine."* The cube-sphere grid goes to a body the seed makes large and round enough to be a world; the flat grid goes to a small or irregular body. The choice is a seed-derived property of the body, read from its size and shape, NEVER from the realm kind. Every asteroid is its own realm, moves as its parent authors, and a player lands, builds and mines on it whichever grid it has. The code's `asteroid = Cartesian` profile is the SMALL case only. |
| Caves and the underground (V2.1 extended) | *"we need to add caves, possibility to explore the underground. Depends on the planet type of course, but earthlike planet should have earthlike structure. But we need to be smart to not to render or 'activate' all non-visible voxels that are lying underground and will not be visible at all. But when we are flying over the cave and there is enough light, we should see what's inside (real life seamless)."* Requirements: (a) caves and strata per planet type, from the seed; (b) NOTHING underground is generated, meshed, collided or held in memory unless a surface crossing lies inside an observer's band — solid rock costs zero; (c) a cave seen from above through its mouth draws its interior with no pop and no load, under the light that reaches it. |
| The word "warp" | The owner asked what "warp" means here. It is the mathematical bend that maps a flat cube face onto the sphere. It has nothing to do with warp travel. **From now on every voxel document calls it THE FACE BEND.** |
| LOD in this topic | The rung is part of the address, so the levels exist from the first record. The rules of the detail ladder (which rung at which distance, the crossfade, the chunk order) are the render-side topic; this topic is the shared address in the core crate, used by the server AND the client. |
| D11 small blocks on planets, and all the like | *"All allowed everywhere, no edge cases."* Small blocks are legal on every body. No feature of the grid is refused on one body kind and allowed on another. |
| Areas (D4a, D4b) — CONFIRMED and EXTENDED into a generic rule | *"areas should somehow repeat the model of the parent (ideally without breaking any laws, we need a very smart generic solution, as physics in those realms will be absolute copy of the parent — area on station will copy its physics, in the Cylinder — its physics, planet — its physics; same can be with the blocks curvature.)"* **Rule: an Area realm has NO grid and NO physics function of its own. It borrows its parent's grid mapping and its parent's gravity function through its told berth, which it reads as an instrument (SL1 clause 2).** Its cells ARE the parent's cells at the parent's bend, so the block forms match the parent's exactly, a vast city follows the planet's curvature by construction, and no flat build exists on a round world. What is area-local is only the ORIGIN of its addresses (an integer offset, subtracted in the parent when the area is made) and the ownership of the cells inside its box (D4b). An area on a station borrows the station's flat grid and its absence of gravity; an area in a cylinder borrows the cylinder's grid and its outward gravity. Two ways to make an area are both supported: (1) place the area first, then build inside it; (2) build on the planet first, then PROMOTE the box: the planet re-bases the records inside the box to the new area's origin and hands them to the new shard, with no change of form or size because both use the same grid. Promotion is designed in the area topic. An area whose box crosses a face edge keeps the parent's face in its address and uses the parent's seam table. |
| The band rule for the underground | The client extracts every chunk inside the observer's band that holds a surface crossing, a sealed hole included; the engine culls what the camera cannot see; nothing outside the band is ever touched; the coarse rungs fold small holes away. **Owner: option A first** (extract everything in the band), with the band's extraction cost on an earthlike crust with caves MEASURED; option B (reachable-air flood fill, extraction at the moment of the dig) lands only if that number is too high. |
| Realistic terrain and the storage, the ladder, the client load and the delta | The owner asked whether the smooth, realistic terrain changes them. Answer recorded in the reply of 2026-09-07: yes, and it was designed in — a density byte in every terrain record (the record grew from 8 to 12 bytes for it), a fill summary and a surface-height datum in the coarse entry, a brush edit that ships a handful of density rows, the extractor at every rung, and a client that meshes only chunks with a surface crossing. |

---

## V6. SLICE 0 — THE FORMAT SITTING, ANSWERED (2026-09-07, night)

The owner read `docs/investigation/2026-09-07/topic_00_format_sitting.md` and said: *"The rest is
approved."* Every recommendation in that document is now a RULING, with one exception (R-20) and one
number still owed (D-1's tolerance).

**Part A, the address — all five approved:** A1 the HR4 gate splits (G-IDENTICAL above the seam;
G-MAPPING-TABLE exhaustive at N = 62 and G-MAPPING-ROUNDTRIP at the largest legal N on the seam);
A2 HR4's wording changes from the stateful `FrameSpace` to the stateless `GridMapping` seam, any physics
anchor sits below it; A3 a built realm's slot is a BOX; A4 the top rung is the one at which a face is at
most 64 × 64 chunks; A5 the generator always places a landform at all eight cube corners.

**Part B, the cell record — B-1 to B-13 approved:** twelve bytes fixed; a 16-bit kind with the registry
capped at the full width; the density byte as the signed radial gap at the cell centre in 1/128 cell,
present on every terrain cell and kept under a placed block; 6-bit rotation with the sixth bit zero;
the tree's shape byte in the record and its growth stage in a side row; the removal = kind `Empty`,
provenance `Placed`; the attachment key = (block address, face, slot byte); a second typed registry
table for attachment kinds under one digest; body 0 = the realm's own grid and a never-reused persisted
counter for the rest; the small-block seat DERIVED by one shared rule; several slots per face; a PREFIX
registry digest at the client handshake plus a per-chunk refusal; the digest covers only what changes a
saved record's meaning, never the style count.

**Part C, the pyramid entry — C-1 to C-5 approved:** prune on equality with the generator's coarse
answer; no surface-height delta by default (decided with the renderer before the first world); no sticky
bit by default, provisional on the fly-away measurement; the canopy fold in a SECOND store family under
the same key; the pyramid persisted at the checkpoint under a watermark.

**Part D, the world identity — D-1 to D-6 approved:** the identity is frozen as a document beside A, B
and C with an APPEND-ONLY octave rule (**the tolerance in metres is still OWED by the owner; asked at
the generator topic**); the generator crate owns a body's radius and the forest reads it; the cave
lattice step is decided on pictures before the seal; the noise is ~400 vendored lines on our integer
hash inside the fenced float type, and the unused `noise` pin is deleted; the x86-64 leg runs emulated
as a smoke test now and on a real x86-64 machine before SL10 V1.3 is called satisfied; the world tag has
two halves (declared on the stamp and the handshake, measured on live handshakes only).

**Part E, the SL6 table:**

| Row | Answer |
|---|---|
| R-3 chunk diff rows on a paced reliable class | **YES** |
| R-5 a reliable world action from the client | **YES** |
| R-7 an exact cheap coarse fold inside the generator (a code ask) | **YES** — the bench decides whether it exists |
| R-8 the realm's own surface tag in its look bag | **YES** |
| R-9 the client's generator handshake | **YES** |
| R-11 a body's pose in its realm's own row | **YES** |
| R-13 the new tags inside the chunk diff | **YES** |
| R-15 the canopy fold in the coarse rung | **YES** |
| R-20 felt acceleration down to a child | **YES ON TRIAL, MARKED "TO TEST".** Owner: *"probably that makes sense, so we can try it out, but mark as to test — I'm afraid it might affect the performance. In theory the Ship Shard calculates the forces and passes them to the Parent, so theoretically speaking we can calculate how all occupants will be affected, but it will not be accurate near the planets, as weight can be different then. Probably planets/stars is the only place where it changes. So let's have it first for testing, but if performance or scalability is affected, we will calculate internally without any crossing."* The fallback is stated now: the hull derives its occupants' felt acceleration from the forces it already states, with no crossing, and accepts the error near a planet or a star where the parent's gravity differs. The trial's gate: the crossing's bytes per tick and its tick cost with 1, 100 and 600 hulls, against the shard's stated physics budget. |
| R-1, R-2 the cross-realm edit forward and its ack | **NO** for v1 (postponed by the owner) |
| R-6 a landed hull's collider surface up to its parent | **NO** for v1; decided at the collider topic (V60 option c preferred) |
| R-10 a mesh peer's measured arithmetic profile | **NO** |
| R-12 a HUD's live value | **NO** until signals land |
| R-14 a falling crown | **NO** for v1; the tree topic answers the black-frame seam |
| R-16 the planet's diff under an area | **NO**; an area's floor is built |
| R-17 a sibling's placement down for a contact list | **NO** for the voxel foundation |
| R-18 a warm lead per realm | **NO** unless the measurement says the reach plus the buffer is not enough |
| R-19 one interest radius per window | **NO**, held in reserve |
| R-21 buoyancy volume up | **NO** until oceans |
| R-4 a rung floor pushed down | **WITHDRAWN** |

Slice 4 may plant exactly the YES rows and R-20 behind its trial flag, and nothing else.

**Slice 0b — CLOSED AS ALREADY LANDED (measured 2026-09-07 at the owner's "go").** The "15,550-line sim
shard file" of the decision board (2026-08-04) was split on 2026-08-21 in commit `40b5ce2` ("Four files
become sixty"). `crates/sim/src/stub/` holds 26 modules (15,883 lines; the largest, `drive.rs`, is
2,182) plus 14 test files (20,939 lines). No file in the workspace exceeds 11,452 lines, and that one
is a test file. The "engine arity workarounds" the gate wanted gone are bevy's bundled tuple
`SystemParam`s under its 16-parameter ceiling (`aoi.rs:199`, `window.rs:539`), the idiomatic shape, not
a defect. The proposal repeated the board's number without measuring; the sequence's next slice is 1.

**Slice 1 — LANDED 2026-09-07 (the owner's "go").** The store seam gains `get` (the point read of one
committed key) and `range` (the bounded half-open range read: `from <= key < to`, ascending, at most
`limit`, inverted = empty, never a panic), on the memory twin and on the disk store; the contract items 5
and 6 are written into the trait. The redb-versus-memory parity test compares both reads after every
commit window and across a reopen. The per-realm handle already existed (`RealmStore`). MEASURED: 643 sim
unit tests pass; Tier-A coverage PASS at 100 %; io-prod regions 94.99 % / 95.17 % against the floor of
94; the sim suite's median wall time 115.75 s after against 115.76 s before (three runs each, idle
machine), so the fast-suite property holds. Next: slice 2, the geometry seam.
